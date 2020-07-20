from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from sklearn import utils as skutils
import math
import datetime
import json
import matplotlib.pyplot as plt
import sys
import pprint

from model.resNet182D.resnet18_2D import resnet18 as resnet2D
from model.resnet18_multibranch.resnet_18_multibranch import resnet18MultiBranch
from model.resNet18LSTM_parallel.resnet_18_lstm import resnet18_lstm as parallel
from model.resNet18LSTM_consecutive.resnet_18_lstm import resnet18_lstm as consecutive
from model.resNet181D.resnet18_1D import resnet18 as resnet1D
from util.dataset import Dataset
from util.tf_metrics import custom_metrics
from util.data_augmentation import random_transformation
from util.data_augmentation import random_guided_warp_multivariate
from util.utils import samples_to_down
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd


class Model():
    def __init__(self, dataset_name, configuration_file, multi_task, lr, model_type, fold=0, save_dir='log', outer_dir='OuterPartition/', overlap=5.0, magnitude=False, log=False):
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
        self.model_type = model_type
        self.sensor_dict = configuration_file.config[dataset_name]['SENSOR_DICT']
        self.fold = fold
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
                                                                                          self.fold[0],
                                                                                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.val_log_dir = "{}/{}/{}/{}/batch_{}/lr_{}/over_{}/fold_{}/{}/val".format(save_dir,
                                                                                      self.model_type,
                                                                                      self.dataset_name,
                                                                                      'multi_task' if self.multi_task else 'single_task',
                                                                                      self.batch_size,
                                                                                      self.lr,
                                                                                      str(overlap),
                                                                                      self.fold[0],
                                                                                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.train_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)        
        self.final_pred_right = [0 for _ in np.arange(0, self.num_act)]
        self.final_pred_wrong = [0 for _ in np.arange(0, self.num_act)]

    def create_dataset(self):
        if self.magnitude:
            channel = self.configuration.config[self.dataset_name]['WINDOW_AXES'] + len(
                list(self.configuration.config[self.dataset_name]['SENSOR_DICT'].keys()))
        else:
            channel = self.configuration.config[self.dataset_name]['WINDOW_AXES']
        if self.dataset_name == 'unimib':
            self.dataset = Dataset(path='data/datasets/UNIMIBDataset/',
                                   name=self.dataset_name,
                                   channel=channel,
                                   winlen=self.configuration.config[self.dataset_name]['WINDOW_SAMPLES'],
                                   user_num=30,
                                   act_num=9,
                                   outer_dir=self.outer_dir)
        elif self.dataset_name == 'sbhar':
            self.dataset = Dataset(path='data/datasets/SBHAR_processed/',
                                   name=self.dataset_name,
                                   channel=channel,
                                   winlen=100,
                                   user_num=30,
                                   act_num=12,
                                   outer_dir=self.outer_dir)
        elif self.dataset_name == 'realdisp':
            self.dataset = Dataset(path='data/datasets/REALDISP_processed/',
                                   name=self.dataset_name,
                                   channel=channel,
                                   winlen=100,
                                   user_num=17,
                                   act_num=33,
                                   outer_dir=self.outer_dir)
        elif self.dataset_name == 'unimib_sbhar':
            self.dataset = Dataset(path='data/datasets/merged_unimib_sbhar/',
                                   name=self.dataset_name,
                                   channel=channel,
                                   winlen=100,
                                   user_num=60,
                                   act_num=-1,
                                   outer_dir=self.outer_dir)

    def load_data(self, only_acc=False, normalize=True, delete=True):
        # gat data [examples, window_samples, axes, channel]
        TrainData, TrainLA, TrainLU, TestData, TestLA, TestLU = self.dataset.load_data(
            step=self.fold, overlapping=self.overlap, normalize=normalize, delete=delete)

        train_shape = TrainData.shape

        self.train = TrainData
        self.train_user = TrainLU
        self.train_act = TrainLA
        self.test = TestData
        self.test_user = TestLU
        self.test_act = TestLA

        if only_acc:
            TrainData = TrainData[:, :, [0, 1, 2, 3]]
            TestData = TestData[:, :, [0, 1, 2, 3]]
            self.axes = 4

        print('shape train data: {}'.format(train_shape))
        print('shape test data: {}'.format(TestData.shape))

        TrainData = tf.data.Dataset.from_tensor_slices(TrainData)
        TrainLA = tf.data.Dataset.from_tensor_slices(TrainLA)
        TrainLU = tf.data.Dataset.from_tensor_slices(TrainLU)

        TestData = tf.data.Dataset.from_tensor_slices(TestData)
        TestLA = tf.data.Dataset.from_tensor_slices(TestLA)
        TestLU = tf.data.Dataset.from_tensor_slices(TestLU)

        train_data = tf.data.Dataset.zip((TrainData, TrainLA, TrainLU))
        train_data = train_data.batch(self.batch_size, drop_remainder=True)
        self.train_data = train_data.shuffle(
            buffer_size=train_shape[0], reshuffle_each_iteration=True)

        test_data = tf.data.Dataset.zip((TestData, TestLA, TestLU))
        self.test_data = test_data.batch(1, drop_remainder=False)

    def augment_data(self, augmented_par=[], plot_augmented=False):

        shape_original = self.train.shape[0]

        train_augmented, label_user_augmented, label_act_augmented = self.dataset.augment_data(
            self.train, self.train_user, self.train_act, self.magnitude, augmented_par, plot_augmented)

        train, test = self.dataset.normalize_data(train_augmented, self.test)

        self.train = train_augmented
        self.train_user = label_user_augmented
        self.train_act = label_act_augmented

        TrainData = tf.data.Dataset.from_tensor_slices(train)
        TrainLA = tf.data.Dataset.from_tensor_slices(label_act_augmented)
        TrainLU = tf.data.Dataset.from_tensor_slices(label_user_augmented)

        TestData = tf.data.Dataset.from_tensor_slices(test)
        TestLA = tf.data.Dataset.from_tensor_slices(self.test_act)
        TestLU = tf.data.Dataset.from_tensor_slices(self.test_user)

        train_data = tf.data.Dataset.zip((TrainData, TrainLA, TrainLU))
        train_data = train_data.batch(self.batch_size, drop_remainder=True)
        self.train_data = train_data.shuffle(
            buffer_size=train.shape[0], reshuffle_each_iteration=True)

        test_data = tf.data.Dataset.zip((TestData, TestLA, TestLU))
        self.test_data = test_data.batch(1, drop_remainder=False)

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
                # penality = sum(tf.nn.l2_loss(tf_var)
                #               for tf_var in self.model.trainable_variables)
                loss_global = loss_a + loss_u
            else:
                predictions_user = self.model(batch, training=True)
                loss_u = self.loss_user(
                    y_true=label_user, y_pred=predictions_user)
                # penality = sum(tf.nn.l2_loss(tf_var)
                #               for tf_var in self.model.trainable_variables)
                loss_global = loss_u

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

    def train_model(self):
        if self.multi_task:
            self.train_multi_task()
        else:
            self.train_single_task()

    def train_single_task(self):
        for epoch in range(1, self.epochs + 1):
            cm = tf.zeros(shape=(self.dataset._user_num,
                                 self.dataset._user_num), dtype=tf.int32)

            for batch, _, label_user in self.train_data:
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
            '''
            if epoch == 50:
                df_cm = pd.DataFrame(cm.numpy(), index = [str(i) for i in range(0,self.dataset._user_num) ],
                                columns = [str(i) for i in range(0,self.dataset._user_num)])
                plt.figure(figsize = (30,21))
                sn.heatmap(df_cm, annot=True)
                plt.show()
            '''

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
            # plot confusion matrix
            '''
            if epoch == 50:
                print(cm.numpy())
                df_cm = pd.DataFrame(cm.numpy(), index = [str(i) for i in range(0,self.dataset._user_num) ],
                                columns = [str(i) for i in range(0,self.dataset._user_num)])
                plt.figure(figsize = (30,21))
                sn.heatmap(df_cm, annot=True)
                plt.show()
            '''

    def decay_lr(self, init_lr=0.001, drop_factor=0.25, drops_epoch=20, epoch=0):

        exp = np.floor((1 + epoch) / drops_epoch)
        alpha = init_lr * (drop_factor ** exp)
        return float(alpha)

    def plot_distribution_data(self, title=''):

        plt.figure(figsize=(12, 3))
        plt.style.use('seaborn-darkgrid')
        plt.title(f'Distribuzione dati in train e test {title}')

        ### distribution user ###

        user_distributions = []
        for user in np.unique(self.train_user):
            plt.subplot(3, 2, 1)
            plt.title('Train user')
            number_user = len([i for i in self.train_user if i == user])
            user_distributions.append(number_user)

        plt.bar(x=list(range(1, len(user_distributions)+1)),
                height=user_distributions)

        user_distributions = []
        for user in np.unique(self.test_user):
            plt.subplot(3, 2, 2)
            plt.title('Test user')
            number_user = len([i for i in self.test_user if i == user])
            user_distributions.append(number_user)

        plt.bar(x=list(range(1, len(user_distributions)+1)),
                height=user_distributions)

        ### distribution activity ###

        act_distributions = []
        for act in np.unique(self.train_act):
            plt.subplot(3, 2, 3)
            plt.title('Train activity')
            number_act = len([i for i in self.train_act if i == act])
            act_distributions.append(number_act)

        plt.bar(x=list(range(1, len(act_distributions)+1)),
                height=act_distributions)

        act_distributions = []
        for act in np.unique(self.test_act):
            plt.subplot(3, 2, 4)
            plt.title('Test activity')
            number_act = len([i for i in self.test_act if i == act])
            act_distributions.append(number_act)
        plt.bar(x=list(range(1, len(act_distributions)+1)),
                height=act_distributions)

        ### distribution activity for user for train ###
        distribution = []  # list of user and activity for user
        for user in set(self.train_user):
            distribution.append([])
            for act in set(self.train_act):
                samples = len([i for i, (u, a) in enumerate(
                    zip(self.train_user, self.train_act)) if a == act and u == user])
                distribution[user].append(samples)

        plt.figure()
        plt.title('Distribution act for user in train set')
        plt.xlabel('User id')
        plt.ylabel('Act id')
        _ = sn.heatmap(np.transpose(distribution),
                       linewidths=0.3, cmap='YlGnBu', annot=True)
        # plt.tight_layout()
        plt.show()

        ### distribution activity for user for test ###
        distribution = []  # list of user and activity for user
        for user in set(self.test_user):
            distribution.append([])
            for act in set(self.test_act):
                samples = len([i for i, (u, a) in enumerate(
                    zip(self.test_user, self.test_act)) if a == act and u == user])
                distribution[user].append(samples)

        plt.figure()
        plt.title('Distribution act for user in test set')
        plt.xlabel('User id')
        plt.ylabel('Act id')
        _ = sn.heatmap(np.transpose(distribution),
                       linewidths=0.3, cmap='YlGnBu', annot=True)
        # plt.tight_layout()
        plt.show()

    def update_pred_based_on_act(self, predictions_user, label_user, label_activity):

        if predictions_user.numpy()[0] == label_user.numpy()[0]:
            self.final_pred_right[label_activity.numpy()[0]] += 1
        else:
            self.final_pred_wrong[label_activity.numpy()[0]] += 1

    def total_sample_for_act(self):
        total_for_act = [0 for _ in np.arange(0, self.num_act)]
        for act in np.arange(0, self.num_act):
            total_for_act[act] += len(self.test[np.where(self.test_act == act)])
        return total_for_act

    def plot_pred_based_act(self):
        plt.figure()
        plt.title('Plot correct and wrong prediction based on activity')
        plt.xlabel('Activity')
        plt.ylabel('Accuracy')
        step = np.arange(0, self.num_act)
        total_for_act = self.total_sample_for_act()
        #pred_right = np.asarray(self.final_pred_right)/np.sum(self.final_pred_right)
        #pred_wrong = np.asarray(self.final_pred_wrong)/np.sum(self.final_pred_wrong)
        pred_right = np.asarray(self.final_pred_right) / \
            np.asarray(total_for_act)
        pred_wrong = np.asarray(self.final_pred_wrong) / \
            np.asarray(total_for_act)
        plt.plot(step, pred_right, 'g', label='Correct pred')
        plt.plot(step, pred_wrong, 'r', label='Wrong pred')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
