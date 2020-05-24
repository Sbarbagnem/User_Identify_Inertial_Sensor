from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from sklearn import utils as skutils
import math

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

def one_hot(y, n_values):

    return np.eye(n_values)[np.array(y, dtype=np.int32)]

def next_batch(train_data, train_la, train_lu, data_pos, batch_size, num_act, num_user):

    train_size = train_data.shape[0]
    scale = data_pos + batch_size

    if scale > train_size:
        a = scale - train_size

        data1 = train_data[data_pos:]
        la1 = train_la[data_pos:]
        lu1 = train_lu[data_pos:]

        # shuffle after one cycle
        train_data, train_la, train_lu = skutils.shuffle(train_data, train_la, train_lu)

        data2 = train_data[: a]

        la2 = train_la[: a]
        lu2 = train_lu[: a]

        data = np.concatenate((data1, data2), axis=0)
        la = np.concatenate((la1, la2), axis=0)
        lu = np.concatenate((lu1, lu2), axis=0)

        data_pos = a

    else:
        data = train_data[data_pos: scale]
        la = train_la[data_pos: scale]
        lu = train_lu[data_pos: scale]
        data_pos = scale
        
    return data, one_hot(la, num_act), one_hot(lu, num_user), data_pos


if __name__ == '__main__':

    DATASET = 'sbhar'
    NUM_ACT = configuration.config[DATASET]['NUM_CLASSES_ACTIVITY']
    NUM_USER = configuration.config[DATASET]['NUM_CLASSES_USER']

    tran = False

    if DATASET == 'unimib':
        dataset = Dataset(  path='data/datasets/UNIMIBDataset/',
                            name='unimib',
                            channel=3,
                            winlen=100,
                            user_num=30,
                            act_num=9)
    elif DATASET == 'sbhar':
        dataset = Dataset(  path='data/datasets/SBHAR_processed/',
                            name='sbhar',
                            channel=6,
                            winlen=100,
                            user_num=30,
                            act_num=12)        

    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # gat data [examples, window_samples, axes, channel]
    TrainData, TrainLA, TrainLU, TestData, TestLA, TestLU = dataset.load_data(step=0)

    train_shape = TrainData.shape
    test_shape = TestData.shape

    # reshape [examples, axes, window_samples, channel]
    if tran:
        TrainData = np.transpose(TrainData, (0,2,1,3))
        TestData = np.transpose(TestData, (0,2,1,3))

    TrainData = tf.data.Dataset.from_tensor_slices(TrainData).batch(configuration.BATCH_SIZE, drop_remainder=True)
    TrainLA = tf.data.Dataset.from_tensor_slices(TrainLA).batch(configuration.BATCH_SIZE, drop_remainder=True)
    TrainLU = tf.data.Dataset.from_tensor_slices(TrainLU).batch(configuration.BATCH_SIZE, drop_remainder=True)

    TestData = tf.data.Dataset.from_tensor_slices(TestData).batch(configuration.BATCH_SIZE, drop_remainder=True) 
    TestLA = tf.data.Dataset.from_tensor_slices(TestLA).batch(configuration.BATCH_SIZE, drop_remainder=True)
    TestLU = tf.data.Dataset.from_tensor_slices(TestLU).batch(configuration.BATCH_SIZE, drop_remainder=True)

    train_data = tf.data.Dataset.zip((TrainData, TrainLA, TrainLU)).shuffle(buffer_size=train_shape[0], reshuffle_each_iteration=True)
    test_data = tf.data.Dataset.zip((TestData, TestLA, TestLU)).shuffle(buffer_size=test_shape[0], reshuffle_each_iteration=False)

    
    # create model
    model = resnet_18(dataset=DATASET)
    #print_model_summary(network=model, dataset=DATASET, tran=True)
    
    # define loss and optimizer
    loss_act   = tf.keras.losses.SparseCategoricalCrossentropy()
    loss_user  = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # performance on train
    train_loss_activity = tf.keras.metrics.Mean(name='train_loss_activity')
    train_loss_user     = tf.keras.metrics.Mean(name='train_loss_user')
    train_accuracy_activity = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_activity')
    train_accuracy_user     = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_user')

    # performance on val
    valid_loss_activity = tf.keras.metrics.Mean(name='valid_loss_activity')
    valid_loss_user     = tf.keras.metrics.Mean(name='valid_loss_user')
    valid_accuracy_activity = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy_activity')
    valid_accuracy_user     = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy_user')

    
    @tf.function
    def train_step(batch, label_activity, label_user):
        with tf.GradientTape() as tape:
            predictions_act, predictions_user = model(batch, training=True)
            loss_a = loss_act(y_true=label_activity, y_pred=predictions_act)
            loss_u = loss_user(y_true=label_user, y_pred=predictions_user)
            loss_global = loss_a + loss_u
                            
        gradients = tape.gradient(loss_global, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        train_loss_activity.update_state(values=loss_a)
        train_loss_user.update_state(values=loss_u)
        train_accuracy_activity.update_state(y_true=label_activity, y_pred=predictions_act)
        train_accuracy_user.update_state(y_true=label_user, y_pred=predictions_user)


    @tf.function
    def valid_step(batch, label_activity, label_user):
        predictions_act, predictions_user = model(batch, training=False)
        loss_a = loss_act(y_true=label_activity, y_pred=predictions_act)
        loss_u = loss_user(y_true=label_user, y_pred=predictions_user)

        valid_loss_activity.update_state(values=loss_a)
        valid_loss_user.update_state(values=loss_u)
        valid_accuracy_activity.update_state(y_true=label_activity, y_pred=predictions_act)
        valid_accuracy_user.update_state(y_true=label_user, y_pred=predictions_user)



    batch_size = configuration.BATCH_SIZE
    #train_total_step = math.ceil(TrainData.shape[0] / configuration.BATCH_SIZE)
    #test_total_step = math.ceil(TestData.shape[0] / configuration.BATCH_SIZE)
    
    # start training
    for epoch in range(1,configuration.EPOCHS+1):
        for batch, label_act, label_user in train_data:
            train_step(batch, label_act, label_user)
            
        #print(optimizer._decayed_lr(tf.float32))
        
        print("TRAIN: epoch: {}/{}, loss_act: {:.5f}, loss_user: {:.5f}, "
                "acc_act: {:.5f}, acc_user: {:.5f}".format(epoch,
                                                            configuration.EPOCHS,
                                                            train_loss_activity.result().numpy(),
                                                            train_loss_user.result().numpy(),
                                                            train_accuracy_activity.result().numpy(),
                                                            train_accuracy_user.result().numpy()))
        train_loss_activity.reset_states()
        train_loss_user.reset_states()
        train_accuracy_activity.reset_states()
        train_accuracy_user.reset_states()
    
        for batch, label_act, label_user in test_data:
            valid_step(batch, label_act, label_user)

        print("VALIDATION: epoch: {}/{}, loss_act: {:.5f}, loss_user: {:.5f}, "
                "acc_act: {:.5f}, acc_user: {:.5f}".format(epoch,
                                                            configuration.EPOCHS,
                                                            valid_loss_activity.result().numpy(),
                                                            valid_loss_user.result().numpy(),
                                                            valid_accuracy_activity.result().numpy(),
                                                            valid_accuracy_user.result().numpy()))  
        valid_loss_activity.reset_states()
        valid_loss_user.reset_states()
        valid_accuracy_activity.reset_states()
        valid_accuracy_user.reset_states()

        print("####################################################################################") 
    
    '''      
        if epoch % save_every_n_epoch == 0:
            model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')

    # save weights
    model.save_weights(filepath=save_model_dir+"model", save_format='tf')

    # save the whole model
    # tf.saved_model.save(model, save_model_dir)
    '''