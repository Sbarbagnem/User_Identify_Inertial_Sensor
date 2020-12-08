import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import sys
import random
from pprint import pprint

from model.resNet182D.resnet18_2D import resnet18
from model.model_paper.model import ModelPaper
from util.utils import split_data_train_val_test_gait, normalize_data, delete_overlap, to_delete
from util.tf_metrics import custom_metrics
from util.data_augmentation import jitter, scaling, magnitude_warp, time_warp, permutation, random_sampling


class ModelGait():
    def __init__(self, config, colab_path):
        if colab_path != '':
            self.path_data = colab_path + \
                ''.join(config['ouisir']['PATH_DATA'].split('.')[1:])
        else:
            self.path_data = colab_path + config['ouisir']['PATH_DATA']
        self.num_user = config['ouisir']['NUM_CLASSES_USER']
        self.best_model = None

    def load_data(self, denoise, filter_num_user=None, method='cycle_based', window_len=100, overlap=50, autocorr=False):

        if method == 'cycle_based':
            if denoise:
                self.path_data = self.path_data + 'cycle_based/denoise/'
            else:
                self.path_data = self.path_data + 'cycle_based/no_denoise/'
            if autocorr:
                self.path_data = self.path_data + 'autocorr/'
            else:
                self.path_data = self.path_data + 'no_autocorr/'
            self.id = None
            self.sessions = None
        elif method == 'window_based':
            self.path_data = self.path_data + \
                f'window_based/{window_len}/{overlap}/'
            self.id = np.load(self.path_data + 'id.npy')
            self.sessions = np.load(self.path_data + 'sessions.npy')

        self.data = np.load(self.path_data + 'data.npy')
        self.label = np.load(self.path_data + 'user_label.npy')
        self.axis = self.data.shape[2]
        self.window_sample = self.data.shape[1]
        self.overlap = overlap

        print(
            f'Found {self.data.shape[0]} cycles for {np.unique(self.label).shape[0]} users')

        if filter_num_user != None:
            idx = np.isin(self.label, np.arange(filter_num_user))
            self.data = self.data[idx]
            self.label = self.label[idx]
            self.num_user = np.unique(self.label).shape[0]
            if method == 'window_based':
                self.id = self.id[idx]
                self.sessions = self.sessions[idx]
            print(f'Filter for first {np.unique(self.label).shape[0]} user')

    def split_train_test(self, method='cycle_based', overlap=None, split=None):

        if method == 'cycle_based':
            self.train, self.val, self.test, self.train_label, self.val_label, self.test_label = split_data_train_val_test_gait(
                data=self.data, label_user=self.label, id_window=None, sessions=None, method=method, overlap=None, split=split)
        else:
            self.train, self.val, self.test, self.train_label, self.val_label, self.test_label = split_data_train_val_test_gait(
                data=self.data, label_user=self.label, id_window=self.id, sessions=self.sessions, method=method, overlap=overlap, split=None)

        print(f'{self.train.shape[0]} gait cycles for train')
        print(f'{self.val.shape[0]} gait cycles for val')
        print(f'{self.test.shape[0]} gait cycles for test')

    def augment_train_data(self, methods):

        functions = {
            'jitter': jitter,
            'scaling': scaling,
            'magntiude_warp': magnitude_warp,
            'time_warp': time_warp,
            'random_sampling': random_sampling,
            #'permutation': permutation 
        }

        print(f'Shape train before augment: {self.train.shape[0]}')

        data_aug = np.empty_like(self.train)
        label_aug = self.train_label
        for i,cycle in enumerate(self.train):
            random_func = random.sample(list(functions.keys()), 3)
            data_aug[i,:,[0,1,2]] = self.apply_aug_function(cycle[:,[0,1,2]], random_func, functions).T
            data_aug[i,:,-1] = np.sqrt(np.sum(np.power(data_aug[i,:,[0,1,2]], 2), 0, keepdims=True))[0]       


        self.train = np.concatenate((self.train, data_aug), axis=0)
        self.train_label = np.concatenate((self.train_label, label_aug))
        print(f'Shape train after augment: {self.train.shape[0]}')

    def apply_aug_function(self, x, index_f, funcs):
        f1 = funcs[index_f[0]]
        f2 = funcs[index_f[1]]
        f3 = funcs[index_f[2]]
        x = f3(f2(f1(x)))
        return x

    def normalize_data(self):
        self.train, self.val, self.test = normalize_data(
            self.train, self.val, self.test)

    def create_tf_dataset(self, batch_size=128):

        # train
        train_tf = tf.data.Dataset.from_tensor_slices((self.train, self.train_label))
        train_tf = train_tf.shuffle(buffer_size=self.train.shape[0], reshuffle_each_iteration=True)
        self.train_tf = train_tf.batch(batch_size, drop_remainder=True)

        # val
        val_tf = tf.data.Dataset.from_tensor_slices((self.val, self.val_label))
        self.val_tf = val_tf.batch(self.val.shape[0])

        # test
        test_tf = tf.data.Dataset.from_tensor_slices((self.test, self.test_label))
        self.test_tf = test_tf.batch(self.test.shape[0])

    def build_model(self, stride, fc, flatten, summary=False, name='our'):
        if name == 'our':
            self.model = resnet18(False, 1, self.num_user, stride, fc, flatten)
        elif name == 'paper':
            self.model = ModelPaper(self.num_user)
        self.model.build(input_shape=(None, self.window_sample, self.axis, 1))
        if summary:
            self.model.summary()

    def loss_metric(self, init_lr=0.001):
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
        self.metric_loss = tf.keras.metrics.Mean()
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(self, batch, label_user, num_user):

        with tf.GradientTape() as tape:
            predictions_user = self.model(batch, training=True)
            loss_batch = self.loss(y_true=label_user, y_pred=predictions_user)
        gradients = tape.gradient(loss_batch, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            grads_and_vars=zip(
                gradients,
                self.model.trainable_variables))
        self.metric_loss.update_state(values=loss_batch)
        self.accuracy.update_state(y_true=label_user, y_pred=predictions_user)

        # confusion matrix on batch
        cm = tf.math.confusion_matrix(label_user, tf.math.argmax(
            predictions_user, axis=1), num_classes=num_user)

        return cm

    @tf.function
    def valid_step(self, batch, label_user, num_user):

        if self.best_model is not None:
            predictions_user = self.best_model(batch, training=False)
        else:
            predictions_user = self.model(batch, training=False)

        loss_batch = self.loss(y_true=label_user, y_pred=predictions_user)

        self.metric_loss.update_state(values=loss_batch)
        self.accuracy.update_state(y_true=label_user, y_pred=predictions_user)

        # calculate precision, recall and f1 from confusion matrix
        cm = tf.math.confusion_matrix(label_user, tf.math.argmax(
            predictions_user, axis=1), num_classes=num_user)

        return cm

    def train_model(
            self,
            epochs=100,
            drop_factor=0.25,
            drop_patience=5,
            early_stop=6,
            log=False):

        best_seen = {
            'epoch': 0,
            'loss': 10,
            'model': None,
            'time_not_improved': 0
        }

        for epoch in range(1, epochs + 1):

            self.metric_loss.reset_states()
            self.accuracy.reset_states()
            cm = tf.zeros(shape=(self.num_user, self.num_user), dtype=tf.int32)

            # train
            for batch, label_user in self.train_tf:
                cm_batch = self.train_step(batch, label_user, self.num_user)
                cm = cm + cm_batch

            metrics = custom_metrics(cm)

            if log:
                mean_loss = self.metric_loss.result().numpy()
                accuracy = self.accuracy.result().numpy()
                print(
                    f"TRAIN epoch: {epoch}/{epochs}, loss: {mean_loss}, acc: {accuracy}, prec: {metrics['macro_precision']}, rec :{metrics['macro_recall']}, f1: {metrics['macro_f1']}")

            self.metric_loss.reset_states()
            self.accuracy.reset_states()
            cm = tf.zeros(shape=(self.num_user, self.num_user), dtype=tf.int32)

            # validation
            for batch, label_user in self.val_tf:
                cm_batch = self.valid_step(batch, label_user, self.num_user)
                cm = cm + cm_batch
            metrics = custom_metrics(cm)

            if log:
                mean_loss = self.metric_loss.result().numpy()
                accuracy = self.accuracy.result().numpy()
                print(
                    f"VALIDATION epoch: {epoch}/{epochs}, loss: {mean_loss}, acc: {accuracy}, prec: {metrics['macro_precision']}, rec :{metrics['macro_recall']}, f1: {metrics['macro_f1']}")

            # update best seen and reduce lr
            if self.metric_loss.result().numpy() < best_seen['loss']:
                best_seen['loss'] = self.metric_loss.result().numpy()
                best_seen['epoch'] = epoch
                best_seen['model'] = self.model
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

        self.best_model = best_seen['model']

    def test_model(self, plot_cm=False):

        self.metric_loss.reset_states()
        self.accuracy.reset_states()

        cm = tf.zeros(shape=(self.num_user,
                             self.num_user), dtype=tf.int32)

        for batch, label_user in self.test_tf:
            cm_batch = self.valid_step(
                batch, label_user, self.num_user)
            cm = cm + cm_batch
        metrics = custom_metrics(cm)

        print(
            "\nTEST FINAL loss: {:.5f}, acc: {:.5f}, macro_precision: {:.5f}, macro_recall: {:.5f}, macro_f1: {:.5f}".format(
                self.metric_loss.result().numpy(),
                self.accuracy.result().numpy(),
                metrics['macro_precision'],
                metrics['macro_recall'],
                metrics['macro_f1']))

        # confusion matrix
        if plot_cm:
            df_cm = pd.DataFrame(
                cm.numpy(), index=[
                    str(i) for i in range(
                        0, self.num_user)], columns=[
                    str(i) for i in range(
                        0, self.num_user)])
            plt.figure(figsize=(30, 21))
            sn.heatmap(df_cm, annot=True)
            plt.show()
