from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from sklearn import utils as skutils
import math
import datetime

from model.resNet18.resnet_18 import resnet18
from model.resnet18_multibranch.resnet_18_multibranch import resnet18MultiBranch
#from model.multi_branch.model_multi_input import model_multi_branch
#from model.multi_branch_lstm.model_multi_input_lstm import model_multi_branch_lstm
from util.data_loader import Dataset


class Model():
    def __init__(self, dataset_name, configuration_file, multi_task, lr, model_type, fold=0):
        self.dataset_name   = dataset_name
        self.configuration  = configuration_file
        self.multi_task     = multi_task
        self.lr             = lr
        self.model_type     = model_type
        self.epochs         = configuration_file.EPOCHS
        self.num_act        = configuration_file.config[dataset_name]['NUM_CLASSES_ACTIVITY']
        self.num_user       = configuration_file.config[dataset_name]['NUM_CLASSES_USER']
        self.batch_size     = configuration_file.BATCH_SIZE
        self.model_type     = model_type
        self.sensor_dict    = configuration_file.config[dataset_name]['SENSOR_DICT']
        self.fold           = fold
        self.train_log_dir  = "log/{}/{}/{}/batch_{}/lr_{}/fold_{}/{}/train".format(self.model_type,
                                                                        self.dataset_name,
                                                                        'mutli_task' if self.multi_task else 'single_task',
                                                                        self.batch_size,
                                                                        self.lr,
                                                                        self.fold,
                                                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.val_log_dir    = "log/{}/{}/{}/batch_{}/lr_{}/fold_{}/{}/val".format(self.model_type,
                                                                      self.dataset_name,
                                                                      'mutli_task' if self.multi_task else 'single_task',
                                                                      self.batch_size,
                                                                      self.lr,
                                                                      self.fold,
                                                                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.train_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)


    def create_dataset(self):
        if self.dataset_name == 'unimib':
            self.dataset = Dataset(path='data/datasets/UNIMIBDataset/',
                                name='unimib',
                                channel=3,
                                winlen=100,
                                user_num=30,
                                act_num=9)
        elif self.dataset_name == 'sbhar':
            self.dataset = Dataset(path='data/datasets/SBHAR_processed/',
                                name='sbhar',
                                channel=6,
                                winlen=100,
                                user_num=30,
                                act_num=12)
        elif self.dataset_name == 'realdisp':
            self.dataset = Dataset(path='data/datasets/REALDISP_processed/',
                                name='realdisp',
                                channel=9,
                                winlen=100,
                                user_num=17,
                                act_num=33,
                                save_dir='acc_gyro_magn/')      

    def load_data(self):
        # gat data [examples, window_samples, axes, channel]
        TrainData, TrainLA, TrainLU, TestData, TestLA, TestLU = self.dataset.load_data(step=self.fold)

        train_shape = TrainData.shape

        print('shape train data: {}'.format(train_shape))
        print('shape test data: {}'.format(TestData.shape))

        # reshape [examples, axes, window_samples, channel]
        TrainData = np.transpose(TrainData, (0, 2, 1, 3))
        TestData = np.transpose(TestData, (0, 2, 1, 3))

        TrainData = tf.data.Dataset.from_tensor_slices(TrainData)
        TrainLA = tf.data.Dataset.from_tensor_slices(TrainLA)
        TrainLU = tf.data.Dataset.from_tensor_slices(TrainLU)

        TestData = tf.data.Dataset.from_tensor_slices(TestData)
        TestLA = tf.data.Dataset.from_tensor_slices(TestLA)
        TestLU = tf.data.Dataset.from_tensor_slices(TestLU)

        train_data = tf.data.Dataset.zip((TrainData, TrainLA, TrainLU))
        train_data = train_data.batch(self.batch_size, drop_remainder=True)
        self.train_data = train_data.shuffle(buffer_size=train_shape[0], reshuffle_each_iteration=True)

        test_data = tf.data.Dataset.zip((TestData, TestLA, TestLU))
        self.test_data = test_data.batch(self.batch_size, drop_remainder=True)        


    def build_model(self):
        # create model
        if self.model_type == 'resnet18':
            self.model = resnet18(self.multi_task, self.num_act, self.num_user)
        if self.model_type == 'resnet18_multi_branch':
            self.model = resnet18MultiBranch(self.sensor_dict, self.multi_task, self.num_act, self.num_user)   

    def print_model_summary(self):
        axes = self.configuration.config[self.dataset_name]['WINDOW_AXES']
        samples = self.configuration.config[self.dataset_name]['WINDOW_SAMPLES']
        channels = self.configuration.config[self.dataset_name]['CHANNELS']
        self.model.build(input_shape=(None, axes, samples, channels))
        self.model.summary()

    def loss_opt_metric(self):
        # define loss and optimizer
        self.loss_act = tf.keras.losses.SparseCategoricalCrossentropy()
        self.loss_user = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # performance on train
        self.train_loss_activity = tf.keras.metrics.Mean(name='train_loss_activity')
        self.train_loss_user = tf.keras.metrics.Mean(name='train_loss_user')
        self.train_accuracy_activity = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy_activity')
        self.train_accuracy_user = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy_user')

        # performance on val
        self.valid_loss_activity = tf.keras.metrics.Mean(name='valid_loss_activity')
        self.valid_loss_user = tf.keras.metrics.Mean(name='valid_loss_user')
        self.valid_accuracy_activity = tf.keras.metrics.SparseCategoricalAccuracy(
            name='valid_accuracy_activity')
        self.valid_accuracy_user = tf.keras.metrics.SparseCategoricalAccuracy(
            name='valid_accuracy_user')

    @tf.function
    def train_step(self, batch, label_activity, label_user):
        with tf.GradientTape() as tape:
            if self.multi_task:
                predictions_act, predictions_user = self.model(batch, training=True)
                loss_a = self.loss_act(
                    y_true=label_activity,
                    y_pred=predictions_act)
                loss_u = self.loss_user(
                    y_true=label_user,
                    y_pred=predictions_user)
                penality = sum( tf.nn.l2_loss(tf_var) for tf_var in self.model.trainable_variables)
                loss_global = loss_a + loss_u + 0.003*penality
            else:
                predictions_user = self.model(batch, training=True)
                loss_u = self.loss_user(y_true=label_user, y_pred=predictions_user)
                penality = sum( tf.nn.l2_loss(tf_var) for tf_var in self.model.trainable_variables)
                loss_global = loss_u + 0.003*penality

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

    @tf.function
    def valid_step(self, batch, label_activity, label_user):
        if self.multi_task:
            predictions_act, predictions_user = self.model(batch, training=False)
            loss_a = self.loss_act(y_true=label_activity, y_pred=predictions_act)
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

    def train_single_task(self):
        for epoch in range(1, self.epochs + 1):
                for batch, _, label_user in self.train_data:
                    self.train_step(batch, None, label_user)

                print("TRAIN: epoch: {}/{}, loss_user: {:.5f}, acc_user: {:.5f}".format(epoch,
                                                                                        self.epochs,
                                                                                        self.train_loss_user.result().numpy(),
                                                                                        self.train_accuracy_user.result().numpy()))
                with self.train_writer.as_default():
                    tf.summary.scalar(
                        'loss_user', self.train_loss_user.result(), step=epoch)
                    tf.summary.scalar(
                        'accuracy_user', self.train_accuracy_user.result(), step=epoch)
                self.train_loss_user.reset_states()
                self.train_accuracy_user.reset_states()

                for batch, _, label_user in self.test_data:
                    self.valid_step(batch, None, label_user)

                print(
                    "VALIDATION: epoch: {}/{}, loss_user: {:.5f}, acc_user: {:.5f}".format(
                        epoch,
                        self.epochs,
                        self.valid_loss_user.result().numpy(),
                        self.valid_accuracy_user.result().numpy()))
                with self.val_writer.as_default():
                    tf.summary.scalar(
                        'loss_user', self.valid_loss_user.result(), step=epoch)
                    tf.summary.scalar(
                        'accuracy_user', self.valid_accuracy_user.result(), step=epoch)
                self.valid_loss_user.reset_states()
                self.valid_accuracy_activity.reset_states()

                if self.lr == 'dynamic':
                    new_lr = self.decay_lr(epoch=epoch)
                    self.optimizer.learning_rate.assign(new_lr)
                    with self.train_writer.as_default():
                        tf.summary.scalar("learning_rate", new_lr, step=epoch)
                
                print("####################################################################################")


    def train_multi_task(self):
        for epoch in range(1, self.epochs + 1):
            if self.multi_task:
                for batch, label_act, label_user in self.train_data:
                    self.train_step(batch, label_act, label_user)
                print(
                    "TRAIN: epoch: {}/{}, loss_act: {:.5f}, loss_user: {:.5f}, "
                    "acc_act: {:.5f}, acc_user: {:.5f}".format(
                        epoch,
                        self.epochs,
                        self.train_loss_activity.result().numpy(),
                        self.train_loss_user.result().numpy(),
                        self.train_accuracy_activity.result().numpy(),
                        self.train_accuracy_user.result().numpy()))
                with self.train_writer.as_default():
                    tf.summary.scalar(
                        'loss_activity', self.train_loss_activity.result(), step=epoch)
                    tf.summary.scalar('accuracy_activity',
                                    self.train_accuracy_activity.result(), step=epoch)
                    tf.summary.scalar(
                        'loss_user', self.train_loss_user.result(), step=epoch)
                    tf.summary.scalar(
                        'accuracy_user', self.train_accuracy_user.result(), step=epoch)
                self.train_loss_activity.reset_states()
                self.train_loss_user.reset_states()
                self.train_accuracy_activity.reset_states()
                self.train_accuracy_user.reset_states()

                for batch, label_act, label_user in self.test_data:
                    self.valid_step(batch, label_act, label_user)
                with self.val_writer.as_default():
                    tf.summary.scalar(
                        'loss_activity', self.valid_loss_activity.result(), step=epoch)
                    tf.summary.scalar('accuracy_activity',
                                    self.valid_accuracy_activity.result(), step=epoch)
                    tf.summary.scalar(
                        'loss_user', self.valid_loss_user.result(), step=epoch)
                    tf.summary.scalar(
                        'accuracy_user', self.valid_accuracy_user.result(), step=epoch)
                print(
                    "VALIDATION: epoch: {}/{}, loss_act: {:.5f}, loss_user: {:.5f}, "
                    "acc_act: {:.5f}, acc_user: {:.5f}".format(
                        epoch,
                        self.epochs,
                        self.valid_loss_activity.result().numpy(),
                        self.valid_loss_user.result().numpy(),
                        self.valid_accuracy_activity.result().numpy(),
                        self.valid_accuracy_user.result().numpy()))
                self.valid_loss_activity.reset_states()
                self.valid_loss_user.reset_states()
                self.valid_accuracy_activity.reset_states()
                self.valid_accuracy_user.reset_states()

                if self.lr == 'dynamic':
                    new_lr = self.decay_lr(epoch=epoch)
                    self.optimizer.learning_rate.assign(new_lr)
                    with self.train_writer.as_default():
                        tf.summary.scalar("learning_rate", new_lr, step=epoch)

    def decay_lr(self, initAlpha=0.001, factor=0.25, dropEvery=15, epoch=0):
        exp = np.floor((1 + epoch) / dropEvery)
        alpha = initAlpha * (factor ** exp)
        return float(alpha)