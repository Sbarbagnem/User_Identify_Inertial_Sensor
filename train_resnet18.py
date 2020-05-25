from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from sklearn import utils as skutils
import math
import datetime

from model.resNet18 import configuration
from model.resNet18.resnet_18 import resnet_18
from util.data_loader import Dataset


def print_model_summary(network, dataset, tran):
    axes = configuration.config[dataset]['WINDOW_AXES']
    samples = configuration.config[dataset]['WINDOW_SAMPLES']
    channels = configuration.config[dataset]['CHANNELS']
    if tran:
        network.build(input_shape=(None, axes, samples, channels))
    else:
        network.build(input_shape=(None, samples, axes, channels))
    network.summary()

def decay_lr(initAlpha=0.01, factor=0.25, dropEvery=10, epoch=0):
    exp = np.floor((1 + epoch) / dropEvery)
    alpha = initAlpha * (factor ** exp)
    return float(alpha)


if __name__ == '__main__':

    DATASET = 'unimib'
    NUM_ACT = configuration.config[DATASET]['NUM_CLASSES_ACTIVITY']
    NUM_USER = configuration.config[DATASET]['NUM_CLASSES_USER']
    BATCH_SIZE = configuration.BATCH_SIZE
    MULTI_TASK = True
    LR = 'static'
    tran = False

    if DATASET == 'unimib':
        dataset = Dataset(path='data/datasets/UNIMIBDataset/',
                          name='unimib',
                          channel=3,
                          winlen=100,
                          user_num=30,
                          act_num=9)
    elif DATASET == 'sbhar':
        dataset = Dataset(path='data/datasets/SBHAR_processed/',
                          name='sbhar',
                          channel=6,
                          winlen=100,
                          user_num=30,
                          act_num=12)
    elif DATASET == 'realdisp':
        dataset = Dataset(path='data/datasets/REALDISP_processed/',
                          name='realdisp',
                          channel=9,
                          winlen=100,
                          user_num=17,
                          act_num=33,
                          save_dir='acc_gyro_magn/')

    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # gat data [examples, window_samples, axes, channel]
    TrainData, TrainLA, TrainLU, TestData, TestLA, TestLU = dataset.load_data(
        step=0)

    train_shape = TrainData.shape
    test_shape = TestData.shape

    # reshape [examples, axes, window_samples, channel]
    if tran:
        TrainData = np.transpose(TrainData, (0, 2, 1, 3))
        TestData = np.transpose(TestData, (0, 2, 1, 3))

    TrainData = tf.data.Dataset.from_tensor_slices(
        TrainData).batch(BATCH_SIZE, drop_remainder=True)
    TrainLA = tf.data.Dataset.from_tensor_slices(
        TrainLA).batch(BATCH_SIZE, drop_remainder=True)
    TrainLU = tf.data.Dataset.from_tensor_slices(
        TrainLU).batch(BATCH_SIZE, drop_remainder=True)

    TestData = tf.data.Dataset.from_tensor_slices(
        TestData).batch(BATCH_SIZE, drop_remainder=True)
    TestLA = tf.data.Dataset.from_tensor_slices(
        TestLA).batch(BATCH_SIZE, drop_remainder=True)
    TestLU = tf.data.Dataset.from_tensor_slices(
        TestLU).batch(BATCH_SIZE, drop_remainder=True)

    train_data = tf.data.Dataset.zip(
        (TrainData, TrainLA, TrainLU)).shuffle(
        buffer_size=train_shape[0], reshuffle_each_iteration=True)
    test_data = tf.data.Dataset.zip((TestData, TestLA, TestLU)).shuffle(
        buffer_size=test_shape[0], reshuffle_each_iteration=False)

    # create model
    model = resnet_18(DATASET, MULTI_TASK)
    print_model_summary(network=model, dataset=DATASET, tran=True)

    # define loss and optimizer
    loss_act = tf.keras.losses.SparseCategoricalCrossentropy()
    loss_user = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # performance on train
    train_loss_activity = tf.keras.metrics.Mean(name='train_loss_activity')
    train_loss_user = tf.keras.metrics.Mean(name='train_loss_user')
    train_accuracy_activity = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy_activity')
    train_accuracy_user = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy_user')

    # performance on val
    valid_loss_activity = tf.keras.metrics.Mean(name='valid_loss_activity')
    valid_loss_user = tf.keras.metrics.Mean(name='valid_loss_user')
    valid_accuracy_activity = tf.keras.metrics.SparseCategoricalAccuracy(
        name='valid_accuracy_activity')
    valid_accuracy_user = tf.keras.metrics.SparseCategoricalAccuracy(
        name='valid_accuracy_user')

    @tf.function
    def train_step(batch, label_activity, label_user, multitask):
        with tf.GradientTape() as tape:
            if multitask:
                predictions_act, predictions_user = model(batch, training=True)
                loss_a = loss_act(
                    y_true=label_activity,
                    y_pred=predictions_act)
                loss_u = loss_user(
                    y_true=label_user, 
                    y_pred=predictions_user)
                loss_global = loss_a + loss_u
            else:
                predictions_user = model(batch, training=True)
                loss_u = loss_user(y_true=label_user, y_pred=predictions_user)
                loss_global = loss_u

        gradients = tape.gradient(loss_global, model.trainable_variables)
        optimizer.apply_gradients(
            grads_and_vars=zip(
                gradients,
                model.trainable_variables))
        if multitask:
            train_loss_activity.update_state(values=loss_a)
            train_accuracy_activity.update_state(
                y_true=label_activity, y_pred=predictions_act)
        train_loss_user.update_state(values=loss_u)
        train_accuracy_user.update_state(
            y_true=label_user, y_pred=predictions_user)

    @tf.function
    def valid_step(batch, label_activity, label_user, multitask):
        if multitask:
            predictions_act, predictions_user = model(batch, training=False)
            loss_a = loss_act(y_true=label_activity, y_pred=predictions_act)
        else:
            predictions_user = model(batch, training=False)
        loss_u = loss_user(y_true=label_user, y_pred=predictions_user)
        if multitask:
            valid_loss_activity.update_state(values=loss_a)
            valid_accuracy_activity.update_state(
                y_true=label_activity, y_pred=predictions_act)
        valid_loss_user.update_state(values=loss_u)
        valid_accuracy_user.update_state(
            y_true=label_user, y_pred=predictions_user)

    experiment = {
        'task': 'multitask' if MULTI_TASK else 'single_task',
        'batch_size': BATCH_SIZE,
        'lr': 'static' if LR == 'static' else 'dynamic',
        'tran': 'tran' if tran == True else 'no_tran',
        'filter': 'original'
    }

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # tensorboard config
    train_writer = tf.summary.create_file_writer(
        "resnet_18_board/{}/{}/batch_{}/lr_{}/{}/{}/{}/train".format(
            DATASET,
            experiment['task'],
            experiment['batch_size'],
            experiment['lr'],
            experiment['filter'],
            experiment['tran'],
            current_time))
    val_writer = tf.summary.create_file_writer(
        "resnet_18_board/{}/{}/batch_{}/lr_{}/{}/{}/{}/val".format(
            DATASET,
            experiment['task'],
            experiment['batch_size'],
            experiment['lr'],
            experiment['filter'],
            experiment['tran'],
            current_time))
    
    # start training
    for epoch in range(1, configuration.EPOCHS + 1):

        if MULTI_TASK:

            for batch, label_act, label_user in train_data:
                train_step(batch, label_act, label_user, MULTI_TASK)

            print(
                "TRAIN: epoch: {}/{}, loss_act: {:.5f}, loss_user: {:.5f}, "
                "acc_act: {:.5f}, acc_user: {:.5f}".format(
                    epoch,
                    configuration.EPOCHS,
                    train_loss_activity.result().numpy(),
                    train_loss_user.result().numpy(),
                    train_accuracy_activity.result().numpy(),
                    train_accuracy_user.result().numpy()))
            with train_writer.as_default():
                tf.summary.scalar('loss_activity', train_loss_activity.result(), step=epoch)
                tf.summary.scalar('accuracy_activity', train_accuracy_activity.result(), step=epoch)
                tf.summary.scalar('loss_user', train_loss_user.result(), step=epoch)
                tf.summary.scalar('accuracy_user', train_accuracy_user.result(), step=epoch)
            train_loss_activity.reset_states()
            train_loss_user.reset_states()
            train_accuracy_activity.reset_states()
            train_accuracy_user.reset_states()

            for batch, label_act, label_user in test_data:
                valid_step(batch, label_act, label_user, MULTI_TASK)
            with val_writer.as_default():
                tf.summary.scalar('loss_activity', valid_loss_activity.result(), step=epoch)
                tf.summary.scalar('accuracy_activity', valid_accuracy_activity.result(), step=epoch)
                tf.summary.scalar('loss_user', valid_loss_user.result(), step=epoch)
                tf.summary.scalar('accuracy_user', valid_accuracy_user.result(), step=epoch)
            print(
                "VALIDATION: epoch: {}/{}, loss_act: {:.5f}, loss_user: {:.5f}, "
                "acc_act: {:.5f}, acc_user: {:.5f}".format(
                    epoch,
                    configuration.EPOCHS,
                    valid_loss_activity.result().numpy(),
                    valid_loss_user.result().numpy(),
                    valid_accuracy_activity.result().numpy(),
                    valid_accuracy_user.result().numpy()))
            valid_loss_activity.reset_states()
            valid_loss_user.reset_states()
            valid_accuracy_activity.reset_states()
            valid_accuracy_user.reset_states()

            new_lr = decay_lr(epoch=epoch)
            optimizer.learning_rate.assign(new_lr)
            with train_writer.as_default():
                tf.summary.scalar("learning_rate", new_lr, step=epoch)

        else:
            for batch, _, label_user in train_data:
                train_step(batch, None, label_user, MULTI_TASK)

            if epoch == 3:
                optimizer.learning_rate.assign(0.00000001)
                print(optimizer.learning_rate)

            print("TRAIN: epoch: {}/{}, loss_user: {:.5f}, acc_user: {:.5f}".format(epoch,
                                                                                    configuration.EPOCHS,
                                                                                    train_loss_user.result().numpy(),
                                                                                    train_accuracy_user.result().numpy()))
            with train_writer.as_default():
                tf.summary.scalar('loss_user', train_loss_user.result(), step=epoch)
                tf.summary.scalar('accuracy_user', train_accuracy_user.result(), step=epoch)
            train_loss_user.reset_states()
            train_accuracy_user.reset_states()

            for batch, _, label_user in test_data:
                valid_step(batch, None, label_user, MULTI_TASK)

            print(
                "VALIDATION: epoch: {}/{}, loss_user: {:.5f}, acc_user: {:.5f}".format(
                    epoch,
                    configuration.EPOCHS,
                    valid_loss_user.result().numpy(),
                    valid_accuracy_user.result().numpy()))
            with val_writer.as_default():
                tf.summary.scalar('loss_user', valid_loss_user.result(), step=epoch)
                tf.summary.scalar('accuracy_user', valid_accuracy_user.result(), step=epoch)
            valid_loss_user.reset_states()
            valid_accuracy_activity.reset_states()

            new_lr = decay_lr(epoch=epoch)
            optimizer.learning_rate.assign(new_lr)
            with train_writer.as_default():
                tf.summary.scalar("learning_rate", new_lr, step=epoch)  

        print("####################################################################################")
    
    '''
        if epoch % save_every_n_epoch == 0:
            model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')

    # save weights
    model.save_weights(filepath=save_model_dir+"model", save_format='tf')

    # save the whole model
    # tf.saved_model.save(model, save_model_dir)
    '''
