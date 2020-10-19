import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from model.resNet182D.resnet18_2D import resnet18
from model.model_paper.model import ModelPaper
from util.utils import split_data_train_val_test_gait, normalize_data
from util.tf_metrics import custom_metrics


class ModelGait():
    def __init__(self, config, colab_path):
        if colab_path != '':
            self.path_data = colab_path + ''.join(config['ouisir']['PATH_DATA'].split('.')[1:])
        else:
            self.path_data = colab_path + config['ouisir']['PATH_DATA']
        self.num_user = config['ouisir']['NUM_CLASSES_USER']
        self.window_sample = config['ouisir']['WINDOW_SAMPLES']
        self.axis = config['ouisir']['AXIS']
        self.best_model = None

    def load_data(self, filter_num_user=None):
        self.data = np.load(self.path_data + 'data.npy')
        self.label = np.load(self.path_data + 'user_label.npy')
        self.sequences = np.load(self.path_data + 'sequences_label.npy')

        if filter_num_user != None:
            idx = np.isin(self.label, np.arange(filter_num_user))
            self.data = self.data[idx]
            self.label = self.label[idx]
            self.num_user = np.unique(self.label).shape[0]
            print(f'found {np.unique(self.label).shape[0]} user')

    def split_train_test(self, train_gait=8, val_test=0.5):
        self.train, self.val, self.test, self.train_label, self.val_label, self.test_label = split_data_train_val_test_gait(
            self.data, self.label, self.sequences, train_gait, val_test)
        print(f'{self.train.shape[0]} gait cycles for train')
        print(f'{self.val.shape[0]} gait cycles for val')
        print(f'{self.test.shape[0]} gait cycles for test')

    def normalize_data(self):
        self.train, self.val, self.test = normalize_data(
            self.train, self.val, self.test, self.axis)

    def create_tf_dataset(self, batch_size=128):
        train_tf = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(
                self.train), tf.data.Dataset.from_tensor_slices(
                self.train_label)))
        self.train_tf = train_tf.shuffle(
            buffer_size=self.train.shape[0],
            reshuffle_each_iteration=True).batch(
            batch_size,
            drop_remainder=True)
        self.val_tf = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(
                self.val), tf.data.Dataset.from_tensor_slices(
                self.val_label))).batch(1)
        self.test_tf = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(
                self.test), tf.data.Dataset.from_tensor_slices(
                self.test_label))).batch(1)

    def build_model(self, stride, fc, flatten, summary=False, name='our'):
        if name == 'our':
            self.model = resnet18(False, 1, self.num_user, stride, fc, flatten)
        else:
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

        return cm,

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
                    f"TRAIN epoch_ {epoch}/{epochs}, loss: {mean_loss}, acc: {accuracy}, prec: {metrics['macro_precision']}, rec :{metrics['macro_recall']}, f1: {metrics['macro_f1']}")

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
                    f"VALIDATION epoch_ {epoch}/{epochs}, loss: {mean_loss}, acc: {accuracy}, prec: {metrics['macro_precision']}, rec :{metrics['macro_recall']}, f1: {metrics['macro_f1']}")

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
            "\nTEST FINAL: loss: {:.5f}, acc: {:.5f}, macro_precision: {:.5f}, macro_recall: {:.5f}, macro_f1: {:.5f}".format(
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
