from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from sklearn import utils as skutils
import math
import datetime

from model.resNet18.resnet_18 import resnet18
from model.resnet18_multibranch.resnet_18_multibranch import resnet18MultiBranch
from model.resNet18LSTM_parallel.resnet_18_lstm import resnet18_lstm as parallel
from model.resNet18LSTM_consecutive.resnet_18_lstm import resnet18_lstm as consecutive
from model.resNet18monoKernel.resNet18_mono_kernel import resnet18MonoKernel
from util.data_loader import Dataset
from util.tf_metrics import custom_metrics
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
        self.log  = log
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
                                                                                  'mutli_task' if self.multi_task else 'single_task',
                                                                                  self.batch_size,
                                                                                  self.lr,
                                                                                  str(overlap),
                                                                                  self.fold,
                                                                                  datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.val_log_dir = "{}/{}/{}/{}/batch_{}/lr_{}/over_{}/fold_{}/{}/val".format(save_dir,
                                                                              self.model_type,
                                                                              self.dataset_name,
                                                                              'mutli_task' if self.multi_task else 'single_task',
                                                                              self.batch_size,
                                                                              self.lr,
                                                                              str(overlap),
                                                                              self.fold,
                                                                              datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.train_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)

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
                                   winlen=100,
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
                                   outer_dir=self.outer_dir,
                                   save_dir='acc_gyro_magn/')
        elif self.dataset_name == 'unimib_sbhar':
             self.dataset = Dataset(path='data/datasets/merged_unimib_sbhar/',
                                   name=self.dataset_name,
                                   channel=channel,
                                   winlen=100,
                                   user_num=60,
                                   act_num=-1,
                                   outer_dir=self.outer_dir)       

    def load_data(self, augmented=False):
        # gat data [examples, window_samples, axes, channel]
        TrainData, TrainLA, TrainLU, TestData, TestLA, TestLU = self.dataset.load_data(
            step=self.fold, overlapping=self.overlap, augmented=augmented)

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
        self.train_data = train_data.shuffle(
            buffer_size=train_shape[0], reshuffle_each_iteration=True)

        test_data = tf.data.Dataset.zip((TestData, TestLA, TestLU))
        self.test_data = test_data.batch(self.batch_size, drop_remainder=True)

    def load_data_merged(self, augmented=False):
        # gat data [examples, window_samples, axes, channel]
        TrainData, TrainLU, TestData, TestLU = self.dataset.load_data(
            step=self.fold, overlapping=self.overlap, augmented=augmented)

        train_shape = TrainData.shape

        print('shape train data: {}'.format(train_shape))
        print('shape test data: {}'.format(TestData.shape))

        # reshape [examples, axes, window_samples, channel]
        TrainData = np.transpose(TrainData, (0, 2, 1, 3))
        TestData = np.transpose(TestData, (0, 2, 1, 3))

        TrainData = tf.data.Dataset.from_tensor_slices(TrainData)
        TrainLU = tf.data.Dataset.from_tensor_slices(TrainLU)

        TestData = tf.data.Dataset.from_tensor_slices(TestData)
        TestLU = tf.data.Dataset.from_tensor_slices(TestLU)

        train_data = tf.data.Dataset.zip((TrainData, TrainLU))
        train_data = train_data.batch(self.batch_size, drop_remainder=True)
        self.train_data = train_data.shuffle(
            buffer_size=train_shape[0], reshuffle_each_iteration=True)

        test_data = tf.data.Dataset.zip((TestData, TestLU))
        self.test_data = test_data.batch(self.batch_size, drop_remainder=True)

    def build_model(self):
        # create model
        if self.model_type == 'resnet18':
            self.model = resnet18(
                self.multi_task, self.num_act, self.num_user, self.axes)
        if self.model_type == 'resnet18_multi_branch':
            self.model = resnet18MultiBranch(
                self.sensor_dict, self.multi_task, self.num_act, self.num_user)
        if self.model_type == 'resnet18_lstm_parallel':
            self.model = parallel(
                self.multi_task, self.num_act, self.num_user, self.axes)
        if self.model_type == 'resnet18_lstm_consecutive':
            self.model = consecutive(
                self.multi_task, self.num_act, self.num_user, self.axes)
        if self.model_type == 'resnet18MonoKernel':
            self.model = resnet18MonoKernel(
                self.multi_task, self.num_act, self.num_user)
        axes = self.axes
        samples = self.configuration.config[self.dataset_name]['WINDOW_SAMPLES']
        channels = self.configuration.config[self.dataset_name]['CHANNELS']
        self.model.build(input_shape=(None, axes, samples, channels))

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
                loss_global = loss_a + loss_u + 0.003*penality
            else:
                predictions_user = self.model(batch, training=True)
                loss_u = self.loss_user(
                    y_true=label_user, y_pred=predictions_user)
                penality = sum(tf.nn.l2_loss(tf_var)
                               for tf_var in self.model.trainable_variables)
                loss_global = loss_u + 0.003*penality
                #loss_global = loss_u

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
        cm = tf.math.confusion_matrix(label_user, tf.math.argmax(predictions_user, axis=1), num_classes=num_user)

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
        cm = tf.math.confusion_matrix(label_user, tf.math.argmax(predictions_user, axis=1), num_classes=num_user)

        return cm
        
    def train(self):
        if self.multi_task:
            self.train_multi_task()
        else:
            self.train_single_task()

    def train_single_task(self):
        for epoch in range(1, self.epochs + 1):
            cm = tf.zeros(shape=(self.dataset._user_num, self.dataset._user_num), dtype=tf.int32)
            if self.dataset_name != 'unimib_sbhar':
                for batch, _, label_user in self.train_data:
                    cm_batch = self.train_step(batch, None, label_user, self.dataset._user_num)
                    cm = cm + cm_batch
            else:
                 for batch, label_user in self.train_data:
                    cm_batch = self.train_step(batch, None, label_user, self.dataset._user_num)
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
            self.train_loss_user.reset_states()
            self.train_accuracy_user.reset_states()
            
            cm = tf.zeros(shape=(self.dataset._user_num, self.dataset._user_num), dtype=tf.int32)

            if self.dataset_name != 'unimib_sbhar':
                for batch, _, label_user in self.test_data:
                    cm_batch = self.valid_step(batch, None, label_user, self.dataset._user_num)
                    cm = cm + cm_batch
            else:
                for batch, label_user in self.test_data:
                    cm_batch = self.valid_step(batch, None, label_user, self.dataset._user_num)
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
            self.valid_loss_user.reset_states()
            self.valid_accuracy_user.reset_states()

            if self.lr == 'dynamic':
                new_lr = self.decay_lr(epoch=epoch)
                self.optimizer.learning_rate.assign(new_lr)
                with self.train_writer.as_default():
                    tf.summary.scalar("learning_rate", new_lr, step=epoch)
            
            if epoch == 50:
                df_cm = pd.DataFrame(cm.numpy(), index = [str(i) for i in range(0,self.dataset._user_num) ],
                                columns = [str(i) for i in range(0,self.dataset._user_num)])
                plt.figure(figsize = (30,21))
                sn.heatmap(df_cm, annot=True)
                plt.show()
            
    def train_multi_task(self):
        for epoch in range(1, self.epochs + 1):
            cm = tf.zeros(shape=(self.dataset._user_num, self.dataset._user_num), dtype=tf.int32)
            if self.multi_task:

                for batch, label_act, label_user in self.train_data:
                    cm_batch = self.train_step(batch, label_act, label_user, self.dataset._user_num)
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
                cm = tf.zeros(shape=(self.dataset._user_num, self.dataset._user_num), dtype=tf.int32)

                for batch, label_act, label_user in self.test_data:
                    cm_batch = self.valid_step(batch, label_act, label_user, self.dataset._user_num)
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
            #plot confusion matrix
            '''
            if epoch == 50:
                print(cm.numpy())
                df_cm = pd.DataFrame(cm.numpy(), index = [str(i) for i in range(0,self.dataset._user_num) ],
                                columns = [str(i) for i in range(0,self.dataset._user_num)])
                plt.figure(figsize = (30,21))
                sn.heatmap(df_cm, annot=True)
                plt.show()
            '''

    def decay_lr(self, initAlpha=0.001, factor=0.25, dropEvery=15, epoch=0):
        exp = np.floor((1 + epoch) / dropEvery)
        alpha = initAlpha * (factor ** exp)
        return float(alpha)
