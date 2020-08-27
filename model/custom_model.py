from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn import utils as skutils
import math
import datetime
import json
import matplotlib.pyplot as plt
import sys
import pprint
import pickle
from sklearn import utils as skutils
from sklearn.utils import class_weight

from model.resNet182D.resnet18_2D import resnet18 as resnet2D
from model.resnet18_multibranch.resnet_18_multibranch import resnet18MultiBranch
from model.resNet18LSTM_parallel.resnet_18_lstm import resnet18_lstm as parallel
from model.resNet18LSTM_consecutive.resnet_18_lstm import resnet18_lstm as consecutive
from model.resNet181D.resnet18_1D import resnet18 as resnet1D
from model.resnet182D_multitask.resnet182D_multitask import resne18MultiTask
from util.dataset import Dataset
from util.tf_metrics import custom_metrics
from util.data_augmentation import random_transformation
from util.data_augmentation import random_guided_warp_multivariate
from util.utils import samples_to_down
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd


class Model():
    def __init__(self, dataset_name, configuration_file, multi_task, lr, model_type, fold_test=0, fold_val=None, save_dir='log', outer_dir='OuterPartition/', overlap=5.0, magnitude=False, log=False):
        self.dataset_name = dataset_name
        self.configuration = configuration_file
        self.multi_task = multi_task
        self.lr = lr
        self.overlap = overlap
        self.model_type = model_type
        self.epochs = configuration_file.EPOCHS
        self.num_act = configuration_file.config[dataset_name]['NUM_CLASSES_ACTIVITY']
        self.num_user = configuration_file.config[dataset_name]['NUM_CLASSES_USER']
        self.batch_size = configuration_file.BATCH_SIZE
        self.latent_dim = configuration_file.LATENT_DIM
        self.epochs_gan = configuration_file.EPOCHS_GAN
        self.model_type = model_type
        self.sensor_dict = configuration_file.config[dataset_name]['SENSOR_DICT']
        self.fold_test = fold_test
        self.fold_val = fold_val
        self.log = log
        self.magnitude = magnitude
        if magnitude:
            self.axes = self.configuration.config[self.dataset_name]['WINDOW_AXES'] + len(
                list(self.configuration.config[self.dataset_name]['SENSOR_DICT'].keys()))
        else:
            self.axes = self.configuration.config[self.dataset_name]['WINDOW_AXES']
        # to see performance on user identification based on activity done
        self.history_act_true = []
        self.history_act_pred = []
        self.history_user_true = []
        self.history_user_pred = []
        self.outer_dir = outer_dir
        self.magnitude = magnitude
        self.train_log_dir = "{}/{}/{}/{}/batch_{}/lr_{}/over_{}/fold_{}/{}/train".format(save_dir,
                                                                                          self.model_type,
                                                                                          self.dataset_name,
                                                                                          'multi_task' if self.multi_task else 'single_task',
                                                                                          self.batch_size,
                                                                                          self.lr,
                                                                                          str(
                                                                                              overlap),
                                                                                          self.fold_test,
                                                                                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.val_log_dir = "{}/{}/{}/{}/batch_{}/lr_{}/over_{}/fold_{}/{}/val".format(save_dir,
                                                                                      self.model_type,
                                                                                      self.dataset_name,
                                                                                      'multi_task' if self.multi_task else 'single_task',
                                                                                      self.batch_size,
                                                                                      self.lr,
                                                                                      str(overlap),
                                                                                      self.fold_test,
                                                                                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.train_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        self.final_pred_right_act = [0 for _ in np.arange(0, self.num_act)]
        self.final_pred_wrong_act = [0 for _ in np.arange(0, self.num_act)]
        self.final_pred_right_user = [0 for _ in np.arange(0, self.num_user)]
        self.final_pred_wrong_user = [0 for _ in np.arange(0, self.num_user)]

    def create_dataset(self):
        if self.magnitude:
            channel = self.configuration.config[self.dataset_name]['WINDOW_AXES'] + len(
                list(self.configuration.config[self.dataset_name]['SENSOR_DICT'].keys()))
        else:
            channel = self.configuration.config[self.dataset_name]['WINDOW_AXES']

        self.dataset = Dataset(path=self.configuration.config[self.dataset_name]['PATH_OUTER_PARTITION'],
                                name=self.dataset_name,
                                channel=channel,
                                winlen=self.configuration.config[self.dataset_name]['WINDOW_SAMPLES'],
                                user_num=self.configuration.config[self.dataset_name]['NUM_CLASSES_USER'],
                                act_num=self.configuration.config[self.dataset_name]['NUM_CLASSES_ACTIVITY'],
                                outer_dir=self.outer_dir,
                                config_file=self.configuration)

    def load_data(self, only_acc=False, normalize=True, delete='delete'):
        # gat data [examples, window_samples, axes, channel]
        TrainData, TrainLA, TrainLU, TestData, TestLA, TestLU, ValidData, ValidLA, ValidLU = self.dataset.load_data(
            step_test=self.fold_test, step_val=self.fold_val, overlapping=self.overlap, delete=delete, magnitude=self.magnitude)

        if only_acc:
            if self.magnitude:
                TrainData = TrainData[:, :, [0, 1, 2, 3]]
                TestData = TestData[:, :, [0, 1, 2, 3]]
                if ValidData is not None:
                    ValidData = ValidData[:, :, [0, 1, 2, 3]]
                self.axes = 4
            else:
                TrainData = TrainData[:, :, [0, 1, 2]]
                TestData = TestData[:, :, [0, 1, 2]]
                if ValidData is not None:
                    ValidData = ValidData[:, :, [0, 1, 2]]
                self.axes = 3
        self.dataset._channel = self.axes

        self.train = TrainData
        self.train_user = TrainLU
        self.train_act = TrainLA
        self.test = TestData
        self.test_user = TestLU
        self.test_act = TestLA

        if ValidData is not None:
            self.val = ValidData
            self.val_user = ValidLU
            self.val_act = ValidLA
        else:
            self.val = None
            self.val_user = None
            self.val_act = None

    def normalize_data(self):
        # normalize data
        self.train, self.test, self.val = self.dataset.normalize_data(
            self.train, self.test, self.val)

    def tf_dataset(self, method, weighted):
        if weighted == 'no':
            self.create_tensorflow_dataset()
        else:
            if method == 'act':
                datasets, weights = self.create_dataset_for_act(weighted)
            if method == 'subject':
                datasets, weights = self.create_dataset_for_subject()
            if method == 'act_subject':
                datasets, weights = self.create_dataset_for_act_subject()
            dataset_weighted = tf.data.experimental.sample_from_datasets(
                datasets, weights)
            dataset_weighted = dataset_weighted.shuffle(
                buffer_size=self.train.shape[0], reshuffle_each_iteration=True)
            dataset_weighted = dataset_weighted.batch(
                self.batch_size, drop_remainder=True)
            self.train_data = dataset_weighted

        TestData = tf.data.Dataset.from_tensor_slices(self.test)
        TestLA = tf.data.Dataset.from_tensor_slices(self.test_act)
        TestLU = tf.data.Dataset.from_tensor_slices(self.test_user)
        test_data = tf.data.Dataset.zip((TestData, TestLA, TestLU))
        self.test_data = test_data.batch(1)

        if self.val is not None:
            ValData = tf.data.Dataset.from_tensor_slices(self.val)
            ValLA = tf.data.Dataset.from_tensor_slices(self.val_act)
            ValLU = tf.data.Dataset.from_tensor_slices(self.val_user)
            test_data = tf.data.Dataset.zip((ValData, ValLA, ValLU))
            self.val_data = test_data.batch(1)

    def create_tensorflow_dataset(self):

        TrainData = tf.data.Dataset.from_tensor_slices(self.train)
        TrainLA = tf.data.Dataset.from_tensor_slices(self.train_act)
        TrainLU = tf.data.Dataset.from_tensor_slices(self.train_user)
        train_data = tf.data.Dataset.zip((TrainData, TrainLA, TrainLU))
        train_data = train_data.shuffle(
            buffer_size=self.train.shape[0], reshuffle_each_iteration=True)
        train_data = train_data.batch(self.batch_size, drop_remainder=True)
        self.train_data = train_data

    def create_dataset_for_act_subject(self):
        datasets = []
        act_user_sample_count = []
        for user in np.unique(self.train_user):
            idx_user = np.where(self.train_user == user)
            for act in np.unique(self.train_act):
                idx = np.intersect1d(idx_user, np.where(self.train_act == act))
                dataset = tf.data.Dataset.from_tensor_slices(
                    (self.train[idx], self.train_act[idx], self.train_user[idx]))
                datasets.append(dataset)
                act_user_sample_count.append(len(idx))

        weights = np.repeat(1., len(act_user_sample_count)) / act_user_sample_count

        print(f'Weight samples based on act and subject frequency:  {weights}')

        return datasets, weights

    def create_dataset_for_subject(self):
        datasets = []
        for user in np.unique(self.train_user):
            idx = np.where(self.train_user == user)
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.train[idx], self.train_act[idx], self.train_user[idx]))
            datasets.append(dataset)

        user_sample_count = [np.where(self.train_user == user)[
            0].shape[0] for user in np.unique(self.train_user)]

        # to have a balance number of samples for every user in batch
        weights = np.repeat(1., len(user_sample_count)) / user_sample_count
        
        print(f'Weight samples based on subject:  {weights}')

        return datasets, weights

    def create_dataset_for_act(self, method='balance'):
        '''
            Weight samples in dataset based on inverse activity frequency
        '''
        datasets = []

        for act in np.unique(self.train_act):
            idx = np.where(self.train_act == act)
            temp_d = self.train[idx]
            temp_a = self.train_act[idx]
            temp_u = self.train_user[idx]
            dataset = tf.data.Dataset.from_tensor_slices(
                (temp_d, temp_a, temp_u))
            datasets.append(dataset)

        # Compute samples weight to have batch sample distribution like train set
        activities_sample_count = [np.where(self.train_act == act)[
            0].shape[0] for act in np.unique(self.train_act)]

        
        # to have balance samples in batch
        if method == 'balance':
            weights = np.repeat(1., len(activities_sample_count)
                            ) / activities_sample_count
        
        # for have the same distribution of train in every batch 
        if method == 'train_set':
            n = np.sum(activities_sample_count)
            weights = activities_sample_count / np.repeat(n, len(activities_sample_count))
        
        print(f'Weight samples based on activity:  {weights}')

        return datasets, weights

    def augment_data(self, augmented_par=[], compose=False, plot_augmented=False):

        shape_original = self.train.shape[0]

        train_augmented, label_user_augmented, label_act_augmented = self.dataset.augment_data(
            self.train, self.train_user, self.train_act, self.magnitude, augmented_par, compose, plot_augmented)

        self.train = train_augmented
        self.train_user = label_user_augmented
        self.train_act = label_act_augmented

        print('data before augmented {}, data after augmented {}'.format(
            shape_original, train_augmented.shape[0]))

    def build_model(self):
        # create model
        if self.model_type == 'resnet18_2D':
            self.model = resnet2D(
                self.multi_task, self.num_act, self.num_user, self.axes
            )
        if self.model_type == 'resnet18_multi_branch':
            self.model = resnet18MultiBranch(
                self.sensor_dict, self.num_user, self.magnitude
            )
        if self.model_type == 'resnet18_lstm_parallel':
            self.model = parallel(
                self.multi_task, self.num_act, self.num_user, self.axes
            )
        if self.model_type == 'resnet18_lstm_consecutive':
            self.model = consecutive(
                self.multi_task, self.num_act, self.num_user, self.axes
            )
        if self.model_type == 'resnet18_1D':
            self.model = resnet1D(
                self.multi_task, self.num_act, self.num_user, self.axes
            )
        if self.model_type == 'resnet18_2D_multitask':
            self.model = resne18MultiTask(
                self.num_act, self.num_user
            )

        samples = self.configuration.config[self.dataset_name]['WINDOW_SAMPLES']
        self.model.build(input_shape=(None, samples, self.axes, 1))

    def print_model_summary(self):
        self.model.summary()

    def loss_opt_metric(self):
        # define loss and optimizer
        self.loss_act = tf.keras.losses.SparseCategoricalCrossentropy()
        self.loss_user = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # performance on train
        self.train_loss_activity = tf.keras.metrics.Mean(
            name='train_loss_activity')
        self.train_loss_user = tf.keras.metrics.Mean(name='train_loss_user')
        self.train_accuracy_activity = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy_activity')
        self.train_accuracy_user = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy_user')
        self.train_precision_user = tf.keras.metrics.Precision()
        self.train_recall_user = tf.keras.metrics.Recall()

        # performance on val
        self.valid_loss_activity = tf.keras.metrics.Mean(
            name='valid_loss_activity')
        self.valid_loss_user = tf.keras.metrics.Mean(name='valid_loss_user')
        self.valid_accuracy_activity = tf.keras.metrics.SparseCategoricalAccuracy(
            name='valid_accuracy_activity')
        self.valid_accuracy_user = tf.keras.metrics.SparseCategoricalAccuracy(
            name='valid_accuracy_user')
        self.val_precision_user = tf.keras.metrics.Precision()
        self.val_recall_user = tf.keras.metrics.Recall()

    @tf.function
    def train_step(self, batch, label_activity, label_user, num_user):
        with tf.GradientTape() as tape:
            if self.multi_task:
                predictions_act, predictions_user = self.model(
                    batch, training=True)
                loss_a = self.loss_act(
                    y_true=label_activity,
                    y_pred=predictions_act)
                loss_u = self.loss_user(
                    y_true=label_user,
                    y_pred=predictions_user)
                penality = sum(tf.nn.l2_loss(tf_var)
                               for tf_var in self.model.trainable_variables)
                loss_global = loss_a + loss_u #+ 0.003*penality
            else:
                predictions_user = self.model(batch, training=True)
                loss_u = self.loss_user(
                    y_true=label_user, y_pred=predictions_user)
                penality = sum(tf.nn.l2_loss(tf_var)
                               for tf_var in self.model.trainable_variables)
                loss_global = loss_u #+ 0.003*penality

        gradients = tape.gradient(loss_global, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            grads_and_vars=zip(
                gradients,
                self.model.trainable_variables))
        if self.multi_task:
            self.train_loss_activity.update_state(values=loss_a)
            self.train_accuracy_activity.update_state(
                y_true=label_activity, y_pred=predictions_act)
        self.train_loss_user.update_state(values=loss_u)
        self.train_accuracy_user.update_state(
            y_true=label_user, y_pred=predictions_user)

        # confusion matrix on batch
        cm = tf.math.confusion_matrix(label_user, tf.math.argmax(
            predictions_user, axis=1), num_classes=num_user)

        return cm

    @tf.function
    def valid_step(self, batch, label_activity, label_user, num_user):
        if self.multi_task:
            predictions_act, predictions_user = self.model(
                batch, training=False)
            loss_a = self.loss_act(y_true=label_activity,
                                   y_pred=predictions_act)
        else:
            predictions_user = self.model(batch, training=False)
        loss_u = self.loss_user(y_true=label_user, y_pred=predictions_user)
        if self.multi_task:
            self.valid_loss_activity.update_state(values=loss_a)
            self.valid_accuracy_activity.update_state(
                y_true=label_activity, y_pred=predictions_act)
        self.valid_loss_user.update_state(values=loss_u)
        self.valid_accuracy_user.update_state(
            y_true=label_user, y_pred=predictions_user)

        # calculate precision, recall and f1 from confusion matrix
        cm = tf.math.confusion_matrix(label_user, tf.math.argmax(
            predictions_user, axis=1), num_classes=num_user)

        return cm, tf.math.argmax(predictions_user, axis=1)

    def distribution_act_on_batch(self, label_act):
        distribution = {act: np.count_nonzero(label_act == act) for act in np.unique(label_act)}
        pprint.pprint(distribution)

    def train_model(self):
        if self.model_type == 'resnet18_2D_multitask':
            self.train_multi_task()
        elif self.multi_task:
            self.train_multi_task()
        else:
            self.train_single_task()

    def train_single_task(self):
        for epoch in range(1, self.epochs + 1):
            cm = tf.zeros(shape=(self.dataset._user_num,
                                 self.dataset._user_num), dtype=tf.int32)

            ### PERFORMANCE ON TRAIN AFTER EACH EPOCH ###

            for batch, label_act, label_user in self.train_data:
                #self.distribution_act_on_batch(label_act)
                cm_batch = self.train_step(
                    batch, None, label_user, self.dataset._user_num)
                cm = cm + cm_batch
            metrics = custom_metrics(cm)
            if self.log:
                print("TRAIN: epoch: {}/{}, loss_user: {:.5f}, acc_user: {:.5f}, macro_precision: {:.5f}, macro_recall: {:.5f}, macro_f1: {:.5f}".format(
                    epoch,
                    self.epochs,
                    self.train_loss_user.result().numpy(),
                    self.train_accuracy_user.result().numpy(),
                    metrics['macro_precision'],
                    metrics['macro_recall'],
                    metrics['macro_f1']))

            with self.train_writer.as_default():
                tf.summary.scalar(
                    'loss_user', self.train_loss_user.result(), step=epoch)
                tf.summary.scalar(
                    'accuracy_user', self.train_accuracy_user.result(), step=epoch)
                tf.summary.scalar(
                    'macro_precision_user', metrics['macro_precision'], step=epoch)
                tf.summary.scalar(
                    'macro_recall_user', metrics['macro_recall'], step=epoch)
                tf.summary.scalar(
                    'macro_f1_user', metrics['macro_f1'], step=epoch)
            self.train_loss_user.reset_states()
            self.train_accuracy_user.reset_states()

            cm = tf.zeros(shape=(self.dataset._user_num,
                                 self.dataset._user_num), dtype=tf.int32)

            ### PERFORMANCE ON VALIDATION AFTER EACH EPOCH ###

            for batch, label_act, label_user in self.test_data:
                if epoch == self.epochs:
                    cm_batch, predictions_user = self.valid_step(
                        batch, label_act, label_user, self.dataset._user_num)
                    cm = cm + cm_batch
                    if self.val is None:
                        self.update_pred_based_on_act(
                            predictions_user, label_user, label_act)
                        self.update_pred_based_on_user(
                            predictions_user, label_user)
                else:
                    cm_batch, _ = self.valid_step(
                        batch, label_act, label_user, self.dataset._user_num)
                    cm = cm + cm_batch
            metrics = custom_metrics(cm)
            if self.log:
                print(
                    "VALIDATION: epoch: {}/{}, loss_user: {:.5f}, acc_user: {:.5f}, macro_precision: {:.5f}, macro_recall: {:.5f}, macro_f1: {:.5f}".format(
                        epoch,
                        self.epochs,
                        self.valid_loss_user.result().numpy(),
                        self.valid_accuracy_user.result().numpy(),
                        metrics['macro_precision'],
                        metrics['macro_recall'],
                        metrics['macro_f1']))
            with self.val_writer.as_default():
                tf.summary.scalar(
                    'loss_user', self.valid_loss_user.result(), step=epoch)
                tf.summary.scalar(
                    'accuracy_user', self.valid_accuracy_user.result(), step=epoch)
                tf.summary.scalar(
                    'macro_precision_user', metrics['macro_precision'], step=epoch)
                tf.summary.scalar(
                    'macro_recall_user', metrics['macro_recall'], step=epoch)
                tf.summary.scalar(
                    'macro_f1_user', metrics['macro_f1'], step=epoch)
            self.valid_loss_user.reset_states()
            self.valid_accuracy_user.reset_states()

            if self.lr == 'dynamic':
                new_lr = self.decay_lr(epoch=epoch)
                self.optimizer.learning_rate.assign(new_lr)
                with self.train_writer.as_default():
                    tf.summary.scalar("learning_rate", new_lr, step=epoch)

        ### PERFORMANCE ON TEST AFTER EPOCHS LOOP ###
        if self.val is not None:
            cm = tf.zeros(shape=(self.dataset._user_num,
                                 self.dataset._user_num), dtype=tf.int32)
            for batch, label_act, label_user in self.val_data:
                if epoch == self.epochs:
                    cm_batch, predictions_user = self.valid_step(
                        batch, label_act, label_user, self.dataset._user_num)
                    cm = cm + cm_batch
                    self.update_pred_based_on_act(
                        predictions_user, label_user, label_act)
                    self.update_pred_based_on_user(
                        predictions_user, label_user)
                else:
                    cm_batch, _ = self.valid_step(
                        batch, label_act, label_user, self.dataset._user_num)
                    cm = cm + cm_batch
            metrics = custom_metrics(cm)
            if self.log:
                print(
                    "\nTEST FINAL: epoch: {}/{}, loss_user: {:.5f}, acc_user: {:.5f}, macro_precision: {:.5f}, macro_recall: {:.5f}, macro_f1: {:.5f}".format(
                        epoch,
                        self.epochs,
                        self.valid_loss_user.result().numpy(),
                        self.valid_accuracy_user.result().numpy(),
                        metrics['macro_precision'],
                        metrics['macro_recall'],
                        metrics['macro_f1']))

        ### CONFUSION MATRIX ON TEST ###
        df_cm = pd.DataFrame(cm.numpy(), index=[str(i) for i in range(0, self.dataset._user_num)],
                             columns=[str(i) for i in range(0, self.dataset._user_num)])
        plt.figure(figsize=(30, 21))
        sn.heatmap(df_cm, annot=True)
        plt.show()

        # save dict metrics object
        with open('metrics_original_unimib.pickle', 'wb+') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)

    def train_multi_task(self):
        for epoch in range(1, self.epochs + 1):
            cm = tf.zeros(shape=(self.dataset._user_num,
                                 self.dataset._user_num), dtype=tf.int32)
            if self.multi_task:
                for batch, label_act, label_user in self.train_data:
                    cm_batch = self.train_step(
                        batch, label_act, label_user, self.dataset._user_num)
                    cm = cm + cm_batch
                metrics = custom_metrics(cm)
                if self.log:
                    print(
                        "TRAIN: epoch: {}/{}, loss_act: {:.5f}, loss_user: {:.5f}, "
                        "acc_act: {:.5f}, acc_user: {:.5f}, macro_precision: {:.5f}, macro_recall: {:.5f}, macro_f1: {:.5f}".format(
                            epoch,
                            self.epochs,
                            self.train_loss_activity.result().numpy(),
                            self.train_loss_user.result().numpy(),
                            self.train_accuracy_activity.result().numpy(),
                            self.train_accuracy_user.result().numpy(),
                            metrics['macro_precision'],
                            metrics['macro_recall'],
                            metrics['macro_f1']))
                with self.train_writer.as_default():
                    tf.summary.scalar(
                        'loss_activity', self.train_loss_activity.result(), step=epoch)
                    tf.summary.scalar('accuracy_activity',
                                      self.train_accuracy_activity.result(), step=epoch)
                    tf.summary.scalar(
                        'loss_user', self.train_loss_user.result(), step=epoch)
                    tf.summary.scalar(
                        'accuracy_user', self.train_accuracy_user.result(), step=epoch)
                    tf.summary.scalar(
                        'macro_precision', metrics['macro_precision'], step=epoch)
                    tf.summary.scalar(
                        'macro_recall', metrics['macro_recall'], step=epoch)
                    tf.summary.scalar(
                        'macro_f1', metrics['macro_f1'], step=epoch)
                self.train_loss_activity.reset_states()
                self.train_loss_user.reset_states()
                self.train_accuracy_activity.reset_states()
                self.train_accuracy_user.reset_states()
                cm = tf.zeros(shape=(self.dataset._user_num,
                                     self.dataset._user_num), dtype=tf.int32)

                for batch, label_act, label_user in self.test_data:
                    if epoch == self.epochs:
                        cm_batch, predictions_user = self.valid_step(
                            batch, label_act, label_user, self.dataset._user_num)
                        cm = cm + cm_batch
                        self.update_pred_based_on_act(
                            predictions_user, label_user, label_act)
                    else:
                        cm_batch, _ = self.valid_step(
                            batch, label_act, label_user, self.dataset._user_num)
                        cm = cm + cm_batch
                metrics = custom_metrics(cm)
                with self.val_writer.as_default():
                    tf.summary.scalar(
                        'loss_activity', self.valid_loss_activity.result(), step=epoch)
                    tf.summary.scalar('accuracy_activity',
                                      self.valid_accuracy_activity.result(), step=epoch)
                    tf.summary.scalar(
                        'loss_user', self.valid_loss_user.result(), step=epoch)
                    tf.summary.scalar(
                        'accuracy_user', self.valid_accuracy_user.result(), step=epoch)
                    tf.summary.scalar(
                        'macro_precision', metrics['macro_precision'], step=epoch)
                    tf.summary.scalar(
                        'macro_recall', metrics['macro_recall'], step=epoch)
                    tf.summary.scalar(
                        'macro_f1', metrics['macro_f1'], step=epoch)
                if self.log:
                    print(
                        "VALIDATION: epoch: {}/{}, loss_act: {:.5f}, loss_user: {:.5f}, "
                        "acc_act: {:.5f}, acc_user: {:.5f}, macro_precision: {:.5f}, macro_recall: {:.5f}, macro_f1: {:.5f}".format(
                            epoch,
                            self.epochs,
                            self.valid_loss_activity.result().numpy(),
                            self.valid_loss_user.result().numpy(),
                            self.valid_accuracy_activity.result().numpy(),
                            self.valid_accuracy_user.result().numpy(),
                            metrics['macro_precision'],
                            metrics['macro_recall'],
                            metrics['macro_f1']))
                self.valid_loss_activity.reset_states()
                self.valid_loss_user.reset_states()
                self.valid_accuracy_activity.reset_states()
                self.valid_accuracy_user.reset_states()

                if self.lr == 'dynamic':
                    new_lr = self.decay_lr(epoch=epoch)
                    self.optimizer.learning_rate.assign(new_lr)
                    with self.train_writer.as_default():
                        tf.summary.scalar("learning_rate", new_lr, step=epoch)

    def decay_lr(self, init_lr=0.001, drop_factor=0.50, drops_epoch=10, epoch=0):

        exp = np.floor((1 + epoch) / drops_epoch)
        alpha = init_lr * (drop_factor ** exp)
        return float(alpha)

    def plot_distribution_data(self, test=True, title=''):

        if test:
            r = 3
        else:
            r = 1
        c = 2

        plt.figure(figsize=(12, 3))
        plt.style.use('seaborn-darkgrid')
        plt.title(f'Distribuzione dati in train e test {title}')

        ### distribution user ###

        user_distributions = []
        for user in np.arange(self.num_user):
            plt.subplot(r, c, 1)
            plt.title('Train user')
            number_user = len([i for i in self.train_user if i == user])
            user_distributions.append(number_user)

        plt.bar(x=list(range(1, len(user_distributions)+1)),
                height=user_distributions)

        if test:
            user_distributions = []
            for user in np.arange(self.num_user):
                plt.subplot(r, c, 2)
                plt.title('Test user')
                number_user = len([i for i in self.test_user if i == user])
                user_distributions.append(number_user)

            plt.bar(x=list(range(1, len(user_distributions)+1)),
                    height=user_distributions)

        ### distribution activity ###

        act_distributions = []
        for act in np.arange(self.num_act):
            if test:
                plt.subplot(r, c, 3)
            else:
                plt.subplot(r, c, 2)
            plt.title('Train activity')
            number_act = len([i for i in self.train_act if i == act])
            act_distributions.append(number_act)
        plt.bar(x=list(range(1, len(act_distributions)+1)),
                height=act_distributions)

        if test:
            act_distributions = []
            for act in np.arange(self.num_act):
                plt.subplot(r, c, 4)
                plt.title('Test activity')
                number_act = len([i for i in self.test_act if i == act])
                act_distributions.append(number_act)
            plt.bar(x=list(range(1, len(act_distributions)+1)),
                    height=act_distributions)

        ### distribution activity for user for train ###
        distribution = []  # list of user and activity for user
        for user in np.arange(self.num_user):
            distribution.append([])
            for act in np.arange(self.num_act):
                samples = len([i for i, (u, a) in enumerate(
                    zip(self.train_user, self.train_act)) if a == act and u == user])
                distribution[user].append(samples)

        plt.figure()
        plt.title('Distribution act for user in train set')
        plt.xlabel('User id')
        plt.ylabel('Act id')
        _ = sn.heatmap(np.transpose(distribution),
                       linewidths=0.3, cmap='YlGnBu', annot=True, fmt="d")
        # plt.tight_layout()
        plt.show()

        if test:
            ### distribution activity for user for test ###
            distribution = []  # list of user and activity for user
            for user in np.arange(self.num_user):
                distribution.append([])
                for act in np.arange(self.num_act):
                    samples = len([i for i, (u, a) in enumerate(
                        zip(self.test_user, self.test_act)) if a == act and u == user])
                    distribution[user].append(samples)

            plt.figure()
            plt.title('Distribution act for user in test set')
            plt.xlabel('User id')
            plt.ylabel('Act id')
            _ = sn.heatmap(np.transpose(distribution),
                           linewidths=0.3, cmap='YlGnBu', annot=True, fmt="d")
            # plt.tight_layout()
            plt.show()

    def update_pred_based_on_act(self, predictions_user, label_user, label_activity):

        if predictions_user.numpy()[0] == label_user.numpy()[0]:
            self.final_pred_right_act[label_activity.numpy()[0]] += 1
        else:
            self.final_pred_wrong_act[label_activity.numpy()[0]] += 1

    def update_pred_based_on_user(self, predictions_user, label_user):

        if predictions_user.numpy()[0] == label_user.numpy()[0]:
            self.final_pred_right_user[label_user.numpy()[0]] += 1
        else:
            self.final_pred_wrong_user[label_user.numpy()[0]] += 1

    def total_sample_for_act(self):
        total_for_act = [0 for _ in np.arange(0, self.num_act)]
        for act in np.arange(0, self.num_act):
            total_for_act[act] += np.unique(self.test_act, return_counts=True)[1][act]
        return total_for_act

    def plot_pred_based_act(self):
        plt.figure()
        plt.title('Plot correct and wrong prediction based on activity')
        plt.xlabel('Activity')
        plt.ylabel('%')
        step = np.arange(0, self.num_act)
        total_for_act = self.total_sample_for_act()
        # pred_right = np.asarray(self.final_pred_right)/np.sum(self.final_pred_right)
        # pred_wrong = np.asarray(self.final_pred_wrong)/np.sum(self.final_pred_wrong)
        pred_right = np.asarray(self.final_pred_right_act) / \
            np.asarray(total_for_act)
        pred_wrong = np.asarray(self.final_pred_wrong_act) / \
            np.asarray(total_for_act)
        plt.plot(step, pred_right, 'g', label='Correct pred')
        plt.plot(step, pred_wrong, 'r', label='Wrong pred')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    def total_sample_for_user(self):
        total_for_user = [0 for _ in np.arange(0, self.num_user)]
        for user in np.arange(0, self.num_user):
            total_for_user[user] += len(
                self.test[np.where(self.test_user == user)])
        return total_for_user

    def plot_pred_based_user(self):
        plt.figure()
        plt.title('Plot correct and wrong prediction based on user')
        plt.xlabel('User')
        plt.ylabel('%')
        step = np.arange(0, self.num_user)
        total_for_user = self.total_sample_for_user()
        # pred_right = np.asarray(self.final_pred_right)/np.sum(self.final_pred_right)
        # pred_wrong = np.asarray(self.final_pred_wrong)/np.sum(self.final_pred_wrong)
        pred_right = np.asarray(self.final_pred_right_user) / \
            np.asarray(total_for_user)
        pred_wrong = np.asarray(self.final_pred_wrong_user) / \
            np.asarray(total_for_user)
        plt.plot(step, pred_right, 'g', label='Correct pred')
        plt.plot(step, pred_wrong, 'r', label='Wrong pred')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    def unify_act(self, mapping):
        num_class_return, act_train, act_test = self.dataset.unify_act_class(self.train_act, self.test_act, mapping)

        self.train_act = act_train
        self.test_act = act_test
        self.num_act = num_class_return
        self.final_pred_right_act = [0 for _ in np.arange(0, self.num_act)]
        self.final_pred_wrong_act = [0 for _ in np.arange(0, self.num_act)]
