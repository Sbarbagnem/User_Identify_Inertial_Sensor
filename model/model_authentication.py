from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
import random
import h5py
from pprint import pprint
from pathlib import Path
from sklearn import utils as skutils
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine

from model.resNet182D.resnet18_2D import resnet18 as resnet2D
from util.tf_metrics import custom_metrics
from util.utils import mapping_act_label, plot_pred_based_act, delete_overlap, normalize_data, mapping_act_label
from util.eer import calculate_eer
from util.data_augmentation import magnitude_warp, time_warp


class ModelAuthentication():
    def __init__(self, path_data, name_dataset, name_model, overlap=None, colab_path=''):
        if colab_path == '':
            self.path_save_model = f'./saved_model/{name_dataset}/'
            self.path_data = path_data
        else:
            self.path_save_model = f'{colab_path}saved_model/{name_dataset}/'
            self.path_data = colab_path + path_data
        if name_dataset.lower() == 'ouisir':
            self.batch_size = 64
        else:
            self.batch_size = 128
        self.init_lr = 0.001
        self.epochs = 100
        self.name_model = name_model
        self.best_model = None
        self.name_dataset = name_dataset
        self.overlap = overlap

        if not os.path.exists(self.path_save_model):
            os.makedirs(self.path_save_model)

    def load_data(self):
        data_dict = dict.fromkeys(
            ['data', 'user_label', 'act_label', 'id', 'gender', 'sessions'])
        for key in list(data_dict.keys()):
            if Path(f'{self.path_data}{key}.npy').is_file():
                data_dict[key] = np.load(f'{self.path_data}{key}.npy')

        if self.name_dataset.lower() in 'ouisir':
            data_dict['act_label'] = np.zeros_like(data_dict['id']) 

        # remove key with None value
        self.data_dict = {k: v for k, v in data_dict.items() if v is not None}
        if 'gender' not in self.data_dict.keys():
            print('There aren\'t gender information for this dataset')
        if 'sessions' not in self.data_dict.keys():
            print('There aren\'t sessions information for this dataset')
        self.win_len = self.data_dict['data'].shape[1]
        self.axis = self.data_dict['data'].shape[2]

        print('{} total window in dataset'.format(
            self.data_dict['data'].shape))

    def split_user(self):
        """
        Split user in two subset: 70% are used to train CNN for feature extraction,
        30% of users are used for authentication evaluation.
        """
        self.classifier = dict.fromkeys(list(self.data_dict.keys()))
        self.auth = dict.fromkeys(list(self.data_dict.keys()))

        # divide based on gender if given
        if 'gender' in list(self.data_dict.keys()):
            male = np.unique(
                self.data_dict['user_label'][np.where(self.data_dict['gender'] == 0)])
            female = np.unique(
                self.data_dict['user_label'][np.where(self.data_dict['gender'] == 1)])
            mask_male_class = np.isin(
                self.data_dict['user_label'], male[:round(len(male)*0.7)])
            mask_female_class = np.isin(
                self.data_dict['user_label'], female[:round(len(female)*0.7)])
            mask_male_auth = np.isin(
                self.data_dict['user_label'], male[round(len(male)*0.7):])
            mask_female_auth = np.isin(
                self.data_dict['user_label'], female[round(len(female)*0.7):])

            # classifier user
            for key in list(self.data_dict.keys()):
                self.classifier[key] = np.concatenate(
                    (self.data_dict[key][mask_male_class], self.data_dict[key][mask_female_class]), axis=0)

            # authentication user
            for key in list(self.data_dict.keys()):
                self.auth[key] = np.concatenate(
                    (self.data_dict[key][mask_male_auth], self.data_dict[key][mask_female_auth]), axis=0)
        else:
            users = np.unique(self.data_dict['user_label'])
            mask_class = np.isin(
                self.data_dict['user_label'], users[:round(len(users)*0.7)])
            mask_auth = np.isin(
                self.data_dict['user_label'], users[round(len(users)*0.7):])

            # classifier user
            for key in list(self.data_dict.keys()):
                self.classifier[key] = self.data_dict[key][mask_class]

            # authentication user
            for key in list(self.data_dict.keys()):
                self.auth[key] = self.data_dict[key][mask_auth]

        # mapping user label
        old_label_user = np.unique(self.classifier['user_label'])
        new_label_user = np.arange(len(old_label_user))

        mapping_user_label = {k: v for k, v in zip(
            old_label_user, new_label_user)}

        self.classifier['user_label'] = [mapping_user_label[user]
                                         for user in self.classifier['user_label']]
        self.num_user_classifier = len(
            np.unique(self.classifier['user_label']))

        print('{} users for train feature extraction'.format(
            len(np.unique(self.classifier['user_label']))))
        print('{} users for authentication evaluation'.format(
            len(np.unique(self.auth['user_label']))))
        print('{} window for train classifier'.format(
            self.classifier['data'].shape))
        print('{} window for evaluate authentication'.format(
            self.auth['data'].shape))

    def split_train_test_classifier(self):
        """
        From numpy to tensorflow data to train. Divide data for train classifier in 80-20
        """
        
        # split data balance based on user and act
        if 'sessions' not in self.classifier.keys():
            data_train, data_val, label_user_train, label_user_val, id_window_train, id_window_val = self.split_train_val_classifier(
                self.classifier['data'], self.classifier['user_label'], self.classifier['act_label'], self.classifier['id'], None, train_size=0.8)
        else:
            data_train, data_val, label_user_train, label_user_val, id_window_train, id_window_val = self.split_train_val_classifier(
                self.classifier['data'], self.classifier['user_label'], self.classifier['act_label'], self.classifier['id'], self.classifier['sessions'], train_size=0.8)            

        print(f'Train window before delete overlap sequence: {data_train.shape[0]}')

        # delete overlap sequence
        if self.overlap == 0.5:
            distance_to_delete = [1]
        elif self.overlap == 0.75:
            distance_to_delete = [1,2,3]
        invalid_idx = delete_overlap(id_window_train, id_window_val, distance_to_delete)
        data_train = np.delete(data_train, invalid_idx, axis=0)
        label_user_train = np.delete(label_user_train, invalid_idx, axis=0)

        print(f'Train window after delete overlap sequence: {data_train.shape[0]}')
        print(f'Validation set: {data_val.shape[0]}')

        self.train = data_train
        self.train_user = label_user_train
        self.val = data_val
        self.val_user = label_user_val
    
    def normalize_data(self):
        self.train, self.val, _, self.mean, self.std = normalize_data(self.train, self.val, return_mean_std=True)

    def augment_train_data(self):

        functions = {
            #'jitter': jitter,
            #'scaling': scaling,
            'magnitude_warp': magnitude_warp,
            'time_warp': time_warp,
            #'random_sampling': random_sampling,
            #'permutation': permutation 
        }

        data_aug = []
        label_aug = []

        print(f'Shape train before augment: {self.train.shape[0]}')

        data_aug_temp = np.empty_like(self.train)
        label_aug_temp = self.train_user
        for i,cycle in enumerate(self.train):
            random_func = random.sample(list(functions.keys()), 2)
            data_aug_temp[i,:,[0,1,2]] = self.apply_aug_function(cycle[:,[0,1,2]], random_func, functions).T
            data_aug_temp[i,:,-1] = np.sqrt(np.sum(np.power(data_aug_temp[i,:,[0,1,2]], 2), 0, keepdims=True))[0]       
        data_aug.extend(data_aug_temp)
        label_aug.extend(label_aug_temp)

        data_aug_temp = np.empty_like(self.train)
        label_aug_temp = self.train_user
        for i,cycle in enumerate(self.train):
            data_aug_temp[i,:,[0,1,2]] = functions['magnitude_warp'](cycle[:,[0,1,2]]).T
            data_aug_temp[i,:,-1] = np.sqrt(np.sum(np.power(data_aug_temp[i,:,[0,1,2]], 2), 0, keepdims=True))[0]
        data_aug.extend(data_aug_temp)
        label_aug.extend(label_aug_temp)

        data_aug_temp = np.empty_like(self.train)
        label_aug_temp = self.train_user
        for i,cycle in enumerate(self.train):
            data_aug_temp[i,:,[0,1,2]] = functions['time_warp'](cycle[:,[0,1,2]]).T
            data_aug_temp[i,:,-1] = np.sqrt(np.sum(np.power(data_aug_temp[i,:,[0,1,2]], 2), 0, keepdims=True))[0]
        data_aug.extend(data_aug_temp)
        label_aug.extend(label_aug_temp)

        data_aug = np.asarray(data_aug)

        self.train = np.concatenate((self.train, data_aug), axis=0)
        self.train_user = np.concatenate((self.train_user, label_aug))
        print(f'Shape train after augment: {self.train.shape[0]}')

    def apply_aug_function(self, x, index_f, funcs):
        f1 = funcs[index_f[0]]
        f2 = funcs[index_f[1]]
        x = f2(f1(x))
        return x
    
    def create_dataset_classifier(self):
        # train
        train_tf = tf.data.Dataset.from_tensor_slices(
            (self.train, self.train_user))
        train_tf = train_tf.shuffle(
            buffer_size=self.train.shape[0], reshuffle_each_iteration=True)
        self.train_tf = train_tf.batch(self.batch_size, drop_remainder=True)

        # val
        val_tf = tf.data.Dataset.from_tensor_slices((self.val, self.val_user))
        self.val_tf = val_tf.batch(self.val.shape[0])

    def build_model(self, stride=1, fc=False):
        if self.name_dataset.lower() != 'ouisir':
            self.feature_extractor = resnet2D(
                multi_task=False, num_act=0, num_user=self.num_user_classifier, stride=stride, fc=fc, flatten=False)
        else:
            self.feature_extractor = resnet2D(
                multi_task=False, num_act=0, num_user=self.num_user_classifier, stride=2, fc=fc, flatten=False)
        self.feature_extractor.build(
            input_shape=(None, self.win_len, self.axis, 1))
        self.feature_extractor.build_graph(
            (self.win_len, self.axis, 1)).summary()

    def loss_opt_metric(self):
        """
        Define metric, optimizer and loss function.
        """
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr)
        self.metric_loss = tf.keras.metrics.Mean()
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(self, batch, label_user, num_user):

        with tf.GradientTape() as tape:
            predictions_user = self.feature_extractor(batch, training=True)
            loss_batch = self.loss(y_true=label_user, y_pred=predictions_user)
        gradients = tape.gradient(
            loss_batch, self.feature_extractor.trainable_variables)
        self.optimizer.apply_gradients(
            grads_and_vars=zip(
                gradients,
                self.feature_extractor.trainable_variables))
        self.metric_loss.update_state(values=loss_batch)
        self.accuracy.update_state(y_true=label_user, y_pred=predictions_user)

        # confusion matrix on batch
        cm = tf.math.confusion_matrix(label_user, tf.math.argmax(
            predictions_user, axis=1), num_classes=num_user)

        return cm

    @tf.function
    def valid_step(self, batch, label_user, num_user):

        predictions_user = self.feature_extractor(batch, training=False)

        loss_batch = self.loss(y_true=label_user, y_pred=predictions_user)

        self.metric_loss.update_state(values=loss_batch)
        self.accuracy.update_state(y_true=label_user, y_pred=predictions_user)

        # calculate precision, recall and f1 from confusion matrix
        cm = tf.math.confusion_matrix(label_user, tf.math.argmax(
            predictions_user, axis=1), num_classes=num_user)

        return cm

    @tf.function
    def feature_generation(self, batch):
        features = self.feature_extractor(
            batch, training=False)
        return features

    def train_model(self, log):

        drop_factor = 0.25
        drop_patience = 5
        early_stop = 6

        best_seen = {
            'epoch': 0,
            'loss': 10,
            'model': None,
            'time_not_improved': 0
        }

        for epoch in range(1, self.epochs + 1):

            self.metric_loss.reset_states()
            self.accuracy.reset_states()
            cm = tf.zeros(shape=(self.num_user_classifier,
                                 self.num_user_classifier), dtype=tf.int32)

            # train
            for batch, label_user in self.train_tf:
                cm_batch = self.train_step(
                    batch, label_user, self.num_user_classifier)
                cm = cm + cm_batch

            metrics = custom_metrics(cm)

            if log:
                mean_loss = self.metric_loss.result().numpy()
                accuracy = self.accuracy.result().numpy()
                print("TRAIN epoch: {}/{}, loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, rec :{:.4f}, f1: {:.4f}".format(
                    epoch, self.epochs, mean_loss, accuracy, metrics[
                        'macro_precision'], metrics['macro_recall'], metrics['macro_f1']
                ))

            self.metric_loss.reset_states()
            self.accuracy.reset_states()
            cm = tf.zeros(shape=(self.num_user_classifier,
                                 self.num_user_classifier), dtype=tf.int32)

            # validation
            for batch, label_user in self.val_tf:
                cm_batch = self.valid_step(
                    batch, label_user, self.num_user_classifier)
                cm = cm + cm_batch
            metrics = custom_metrics(cm)

            if log:
                mean_loss = self.metric_loss.result().numpy()
                accuracy = self.accuracy.result().numpy()
                print("VAL epoch: {}/{}, loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, rec :{:.4f}, f1: {:.4f}".format(
                    epoch, self.epochs, mean_loss, accuracy, metrics[
                        'macro_precision'], metrics['macro_recall'], metrics['macro_f1']
                ))

            # update best seen and reduce lr
            if self.metric_loss.result().numpy() < best_seen['loss']:
                best_seen['loss'] = self.metric_loss.result().numpy()
                best_seen['epoch'] = epoch
                best_seen['model'] = self.feature_extractor
                best_seen['time_not_improved'] = 0
            else:
                best_seen['time_not_improved'] += 1
                if best_seen['time_not_improved'] >= early_stop and epoch > 20:
                    print('early stop')
                    break
                elif best_seen['time_not_improved'] == drop_patience:
                    new_lr = self.optimizer.learning_rate * drop_factor
                    if new_lr < 0.000001:
                        print('min lr reached')
                        break
                    self.optimizer.learning_rate.assign(new_lr)
                    print(f'reduce learning rate on plateau to {new_lr}')

        self.feature_extractor = best_seen['model']

    def save_model(self):

        print('Save model')
        self.feature_extractor.save_weights(
            self.path_save_model + self.name_model + '.h5')

        print('Mean and std')
        np.save(self.path_save_model + 'mean.npy', self.mean)
        np.save(self.path_save_model + 'std.npy', self.std)

    def load_model(self):

        print('Load model')
        if self.name_dataset.lower() != 'ouisir':
            self.feature_extractor = resnet2D(
                False, 0, self.num_user_classifier, feature_generator=True)
        else:
            self.feature_extractor = resnet2D(
                False, 0, self.num_user_classifier, stride=2, feature_generator=False)
        self.feature_extractor.build((None, self.win_len, self.axis, 1))
        self.feature_extractor.load_weights(
            self.path_save_model + self.name_model + '.h5', by_name=True)

        print('Load mean and std')
        self.mean = np.load(self.path_save_model + 'mean.npy')
        self.std = np.load(self.path_save_model + 'std.npy')

    def generate_features(self, split_probe_gallery=''):

        if not os.path.exists(self.path_data + 'gallery_probe/'):
            os.makedirs(self.path_data + 'gallery_probe/')

        path_probe_gallery = self.path_data + 'gallery_probe/'

        # sort data based on id window (time sorted)
        idx_sorted = np.argsort(self.auth['id'])
        self.auth['data'] = self.auth['data'][idx_sorted]
        self.auth['user_label'] = self.auth['user_label'][idx_sorted]
        self.auth['act_label'] = self.auth['act_label'][idx_sorted]
        self.auth['sessions'] = self.auth['sessions'][idx_sorted]
        self.auth['id'] = self.auth['id'][idx_sorted]

        # case1: probe-gallery for every session
        if split_probe_gallery == 'intra_session':
            # make_dir session_dependent
            path_probe_gallery = os.path.join(
                path_probe_gallery, 'intra_session/')
            if not os.path.exists(path_probe_gallery):
                os.makedirs(path_probe_gallery)
            # make_dir for every session session_dependent/session_{n_session}
            for session in np.unique(self.auth['sessions']):
                path_session = os.path.join(
                    path_probe_gallery, f'session_{session}/')
                if not os.path.exists(path_session):
                    os.makedirs(path_session)
                if not os.path.exists(os.path.join(path_session, 'gallery/')):
                    os.makedirs(os.path.join(path_session, 'gallery/'))
                if not os.path.exists(os.path.join(path_session, 'probe/')):
                    os.makedirs(os.path.join(path_session, 'probe/'))
                # take data from n_session
                idx_session = np.where(self.auth['sessions'] == session)
                data = self.auth['data'][idx_session]
                user_label = self.auth['user_label'][idx_session]
                act_label = self.auth['act_label'][idx_session]
                ID = self.auth['id'][idx_session]
                # divide in two balanced subset (user-act)
                gallery, user_label_gallery, act_label_gallery, probe, user_label_probe, act_label_probe = self.balance_gallery_probe(
                    data, user_label, act_label, ID
                )
                print(f'Session {session}')
                print(f'Gallery shape: {gallery.shape}')
                print(f'Probe shape: {probe.shape}')
                # save gallery and probe in every session respectively
                self.generate_save_feaures(
                    gallery, user_label_gallery, act_label_gallery, probe, user_label_probe, act_label_probe, path_session)
        else:
            # case2: there are more session, first for gallery and other for probe
            if split_probe_gallery == 'extra_session':
                path_probe_gallery = path_probe_gallery + 'extra_session/'
                if not os.path.exists(path_probe_gallery + 'gallery/'):
                    os.makedirs(path_probe_gallery + 'gallery/')
                if not os.path.exists(path_probe_gallery + 'probe/'):
                    os.makedirs(path_probe_gallery + 'probe/')
                idx_gallery = np.where(self.auth['sessions'] == 0)
                idx_probe = np.where(self.auth['sessions'] == 1)
                gallery = np.expand_dims(self.auth['data'][idx_gallery], 3)
                user_label_gallery = self.auth['user_label'][idx_gallery]
                act_label_gallery = self.auth['act_label'][idx_gallery]
                probe = np.expand_dims(self.auth['data'][idx_probe], 3)
                user_label_probe = self.auth['user_label'][idx_probe]
                act_label_probe = self.auth['act_label'][idx_probe]
            # case3: random split between sessions
            elif split_probe_gallery == 'random':
                path_probe_gallery = path_probe_gallery + 'random/'
                if not os.path.exists(path_probe_gallery + 'gallery/'):
                    os.makedirs(path_probe_gallery + 'gallery/')
                if not os.path.exists(path_probe_gallery + 'probe/'):
                    os.makedirs(path_probe_gallery + 'probe/')
                gallery, user_label_gallery, act_label_gallery, probe, user_label_probe, act_label_probe = self.balance_gallery_probe(
                    self.auth['data'], self.auth['user_label'], self.auth['act_label'], self.auth['id'])

            print(f'Gallery shape: {gallery.shape}')
            print(f'Probe shape: {probe.shape}')
            self.generate_save_feaures(gallery, user_label_gallery, act_label_gallery,
                                       probe, user_label_probe, act_label_probe, path_probe_gallery)

    def generate_save_feaures(self, gallery, user_label_gallery, act_label_gallery, probe, user_label_probe, act_label_probe, path_to_save):

        # add normalization gallery and probe
        gallery = np.expand_dims(((gallery[:,:,:,0] - self.mean)/self.std), 3)
        probe = np.expand_dims(((probe[:,:,:,0] - self.mean)/self.std), 3)

        # extracts features
        print('Generate features')
        gallery_tf = tf.data.Dataset.from_tensor_slices(
            (gallery)).batch(1)
        probe_tf = tf.data.Dataset.from_tensor_slices(
            (probe)).batch(1)

        gallery_feature = []
        probe_feature = []

        for data in gallery_tf:
            gallery_feature.append(self.feature_generation(data))
        for data in probe_tf:
            probe_feature.append(self.feature_generation(data))

        gallery_feature = np.concatenate(gallery_feature, axis=0)
        probe_feature = np.concatenate(probe_feature, axis=0)

        # l2 normalization on features
        gallery_feature = self.normalize_feature_vector(gallery_feature)
        probe_feature = self.normalize_feature_vector(probe_feature)

        print(f'Shape gallery feature{gallery_feature.shape}')
        print(f'Shape probe feature{probe_feature.shape}')

        # save features dataset
        print('Save features')
        np.save(path_to_save + 'gallery/gallery.npy', gallery_feature)
        np.save(path_to_save + 'gallery/user_label_gallery.npy',
                user_label_gallery)
        np.save(path_to_save + 'gallery/act_label_gallery.npy',
                act_label_gallery)
        np.save(path_to_save + 'probe/probe.npy', probe_feature)
        np.save(path_to_save +
                'probe/user_label_probe.npy', user_label_probe)
        np.save(path_to_save + 'probe/act_label_probe.npy', act_label_probe)

    def normalize_feature_vector(self, features):
        """
        Normalize with L2 norm the features vectors
        """

        features_normalized = features / np.repeat(np.linalg.norm(features, ord=2, axis=1).reshape(
            (features.shape[0], 1)), features.shape[1], axis=1)

        return features_normalized

    def distance_prob(self, distance):
        """
        From distance to prob, after this pos in eer is 1
        """

        distance = 1 - (distance/np.max(distance))

        return distance

    def compute_distance_gallery_probe(self, split_gallery_probe, action_dependent=True, preprocess=False):

        path_probe_gallery = os.path.join(
            self.path_data, f'gallery_probe/{split_gallery_probe}/')
        path_distance_txt = os.path.join(
            self.path_data, f'distance/{split_gallery_probe}/')
        if not os.path.exists(path_distance_txt):
            os.makedirs(path_distance_txt)

        if split_gallery_probe == 'intra_session':
            for dir_session in os.listdir(path_probe_gallery):
                path_session = os.path.join(
                    path_probe_gallery, f'{dir_session}/')
                path_distance_session = path_distance_txt + f'{dir_session}/'
                if not os.path.exists(path_distance_session):
                    os.makedirs(path_distance_session)
                gallery = np.load(path_session + 'gallery/gallery.npy')
                user_gallery = np.load(
                    path_session + 'gallery/user_label_gallery.npy')
                act_gallery = np.load(path_session +
                                      'gallery/act_label_gallery.npy')
                probe = np.load(path_session + 'probe/probe.npy')
                user_probe = np.load(path_session +
                                     'probe/user_label_probe.npy')
                act_probe = np.load(path_session +
                                    'probe/act_label_probe.npy')
                self.compute_distance(gallery, user_gallery, act_gallery, probe,
                                      user_probe, act_probe, preprocess, action_dependent, path_distance_session)
        else:
            gallery = np.load(path_probe_gallery + 'gallery/gallery.npy')
            user_gallery = np.load(
                path_probe_gallery + 'gallery/user_label_gallery.npy')
            act_gallery = np.load(path_probe_gallery +
                                  'gallery/act_label_gallery.npy')
            probe = np.load(path_probe_gallery + 'probe/probe.npy')
            user_probe = np.load(path_probe_gallery +
                                 'probe/user_label_probe.npy')
            act_probe = np.load(path_probe_gallery +
                                'probe/act_label_probe.npy')
            path_to_save = path_distance_txt
            self.compute_distance(gallery, user_gallery, act_gallery, probe,
                                  user_probe, act_probe, preprocess, action_dependent, path_to_save)

    def compute_distance(self, gallery, user_gallery, act_gallery, probe, user_probe, act_probe, preprocess, action_dependent, path_to_save):

        # preprocess features
        if preprocess:
            print(f'Shape gallery before processing {gallery.shape}')
            gallery, user_gallery, act_gallery, probe, user_probe, act_probe = self.process_feature(
                gallery, user_gallery, act_gallery, probe, user_probe, act_probe)
            print(f'Shape gallery before processing {gallery.shape}')

        # compute distances 
        if action_dependent:
            if not os.path.exists(os.path.join(path_to_save, 'action_dependent/')):
                os.makedirs(os.path.join(
                    path_to_save, 'action_dependent/'))
            for act in np.unique(act_gallery):
                distances = []
                idx_gallery = np.where(act_gallery == act)
                idx_probe = np.where(act_probe == act)
                gallery_filtered = gallery[idx_gallery]
                user_gallery_filtered = user_gallery[idx_gallery]
                probe_filtered = probe[idx_probe]
                user_probe_filtered = user_probe[idx_probe]
                print(f'Gallery shape: {gallery_filtered.shape}')
                print(f'Probe shape: {probe_filtered.shape}')
                for seq1, user1 in zip(gallery_filtered, user_gallery_filtered):
                    for seq2, user2 in zip(probe_filtered, user_probe_filtered):
                        dist = np.linalg.norm(seq1 - seq2)
                        same = 1 if user1 == user2 else 0
                        distances.append([dist, same, user1])
                distances = np.array(distances)
                np.savetxt(
                    path_to_save + f'action_dependent/distance_act_{act}.txt', distances, delimiter='\t', fmt="%.3f %d %d")
        # compute distances and evaluate eer without filter for action
        else:
            if not os.path.exists(os.path.join(path_to_save, 'action_independent/')):
                os.makedirs(os.path.join(
                    path_to_save, 'action_independent/'))
            distances = []
            for seq1, user1 in zip(gallery, user_gallery):
                for seq2, user2 in zip(probe, user_probe):
                    dist = np.linalg.norm(seq1 - seq2)
                    same = 1 if user1 == user2 else 0
                    distances.append([dist, same, user1])
            distances = np.array(distances)
            np.savetxt(
                path_to_save + 'action_independent/distance.txt', distances, delimiter='\t', fmt="%.3f %d %d")

    def compute_eer(self, split_gallery_probe, action_dependent=True):
        path_distance_txt = os.path.join(
            self.path_data, f'distance/{split_gallery_probe}/')

        if split_gallery_probe == 'intra_session':
            for dir_session in os.listdir(path_distance_txt):
                print(f'Session {dir_session.split("_")[1]}')
                _, _ = self.eer(
                    path_distance_txt + f'{dir_session}/', action_dependent)
        else:
            _, _ = self.eer(path_distance_txt, action_dependent)

    def eer(self, path_score_txt, action_dependent):
        path_save_eer = path_score_txt.split(
            'distance')[0] + 'eer' + path_score_txt.split('distance')[1]
        if action_dependent:
            path_save_eer = path_save_eer + 'action_dependent/'
        else:
            path_save_eer = path_save_eer + 'action_independent/'
        if not os.path.exists(path_save_eer):
            os.makedirs(path_save_eer)

        # load txt based on action dependent or independent
        if action_dependent:
            # read score for every act
            eers = []
            for act in np.unique(self.data_dict['act_label']):
                score_txt = np.loadtxt(
                    path_score_txt + f'action_dependent/distance_act_{act}.txt')
                score = score_txt[:, 0]
                label = score_txt[:, 1]
                eer, threshold = calculate_eer(label, score, pos=0)
                eers.append([act, eer*100])
                print('Act {} EER: {:.3f} Trhesh: {:.3f}'.format(
                    act, eer*100, threshold))
            # save result
            with open(f"{path_save_eer}eer.txt", "w") as f:
                for el in eers:
                    f.write('{} EER: {:.3f}%\n'.format(mapping_act_label(
                        self.name_dataset.lower())[el[0]], el[1]))
        else:
            score_txt = np.loadtxt(
                path_score_txt + 'action_independent/distance.txt')
            score = score_txt[:, 0]
            label = score_txt[:, 1]
            eer, threshold = calculate_eer(label, score, pos=0)
            print('EER: {:.3f}'.format(eer*100))
            # save result
            with open(f"{path_save_eer}eer.txt", "w") as f:
                f.write('EER: {:.3f}%'.format(eer*100))

        return eer, threshold

    def balance_gallery_probe(self, data, users, activities, id_window):

        if self.overlap == 0.5:
            to_del = [1]
        elif self.overlap == 0.75:
            to_del = [1,2,3]

        gallery = []
        probe = []
        users_gallery = []
        users_probe = []
        act_gallery = []
        act_probe = []

        for user in np.unique(users):
            for act in np.unique(activities):

                # filter for user and activity
                idx = np.where((users == user) & (activities == act))
                samples = len(idx[0])
                data_temp = data[idx]
                id_temp = id_window[idx]

                # split 50% gallery and 50% probe
                gallery_temp = data_temp[:int(samples/2)]
                gallery_id = id_temp[:int(samples/2)]
                probe_temp = data_temp[int(samples/2):]
                probe_id = id_temp[int(samples/2):]

                # delete possible overlap between gallery and probe (delete from probe)
                invalid_idx = delete_overlap(probe_id, gallery_id, to_del)
                probe_temp = np.delete(probe_temp, invalid_idx, axis=0)             

                gallery.append(gallery_temp)
                probe.append(probe_temp)
                users_gallery.extend(
                    [user for _ in range(gallery_temp.shape[0])])
                users_probe.extend(
                    [user for _ in range(probe_temp.shape[0])])
                act_gallery.extend([act for _ in range(gallery_temp.shape[0])])
                act_probe.extend([act for _ in range(probe_temp.shape[0])])

        gallery = np.expand_dims(np.concatenate(gallery, axis=0), 3)
        probe = np.expand_dims(np.concatenate(probe, axis=0), 3)
        users_gallery = np.asarray(users_gallery)
        users_probe = np.asarray(users_probe)
        act_gallery = np.asarray(act_gallery)
        act_probe = np.asarray(act_probe)

        return gallery, users_gallery, act_gallery, probe, users_probe, act_probe

    def process_feature(self, gallery, user_gallery, act_gallery, probe, user_probe, act_probe):

        # mean 0 variance 1

        _mean = np.mean(gallery, axis=0)
        _std = np.std(gallery, axis=0)
        gallery = (gallery - _mean) / _std
        probe = (probe - _mean) / _std

        # PCA for dimension reduction ?
        #pca = PCA()
        pca = KernelPCA(kernel='rbf', gamma=0.0001)
        pca.fit(gallery)
        gallery_pca = pca.transform(gallery)
        probe_pca = pca.transform(probe)

        explained_variance = np.var(gallery_pca, axis=0)
        explained_variance_ratio = explained_variance / \
            np.sum(explained_variance)
        print('Cumsum explained variance of gallery PCA (of first 20 component)')
        print(np.cumsum(explained_variance_ratio)[20])

        explained_variance = np.var(probe_pca, axis=0)
        explained_variance_ratio = explained_variance / \
            np.sum(explained_variance)
        print('Cumsum explained variance of probe PCA (of first 20 component)')
        print(np.cumsum(explained_variance_ratio)[20])

        gallery = gallery_pca[:, :20]
        probe = probe_pca[:, :20]

        return gallery, user_gallery, act_gallery, probe, user_probe, act_probe

    def split_train_val_classifier(self, data, users, activities, id_window, sessions, train_size=0.8):

        data_train = []
        data_val = []
        label_user_train = []
        label_user_val = []
        id_window_train = []
        id_window_val = []

        for user in np.unique(users):
            for act in np.unique(activities):
                idx = np.where((users == user) & (activities == act))
                data_temp = data[idx]
                user_temp = np.array(users)[idx]
                id_temp = np.array(id_window)[idx]
                train = int(len(data_temp)*train_size)
                data_train.append(data_temp[:train])
                data_val.append(data_temp[train:])
                label_user_train.append(user_temp[:train])
                label_user_val.append(user_temp[train:])
                id_window_train.append(id_temp[:train])
                id_window_val.append(id_temp[train:])

        data_train = np.concatenate(data_train, axis=0)
        data_val = np.concatenate(data_val, axis=0)
        label_user_train = np.concatenate(label_user_train, axis=0)
        label_user_val = np.concatenate(label_user_val, axis=0)
        id_window_train = np.concatenate(id_window_train, axis=0)
        id_window_val = np.concatenate(id_window_val, axis=0)

        return data_train, data_val, label_user_train, label_user_val, id_window_train, id_window_val
