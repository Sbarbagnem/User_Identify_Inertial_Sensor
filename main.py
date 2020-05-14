import argparse
import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2
from scipy.fftpack import fft
from sklearn import utils as skutils
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from model import model
from util.data_loader import Dataset

#import tensorflow.contrib.slim as slim
#from sklearn.neighbors import KNeighborsClassifier

class my_model(object):

    def __init__(self, version, gpu, fold, save_dir, dataset, framework, iter_steps=50000):

        self._dataset   = dataset
        self._gpu       = gpu
        self._log_path  = dataset._path+'log/'+"{}/".format(version)+"/fold{}/".format(fold)
        #self._fold = fold % 10
        self._fold      = fold
        self._save_path = dataset._path+'record/'+save_dir
        self._result_path = self._save_path + "/"
        self._framework = framework
        self._save_dir  = save_dir

        self._iter_steps        = iter_steps
        self._print_interval    = 100
        self._batch_size        = 100
        #self._epoch         = epochs
        self._min_lr            = 0.0005
        #self._max_lr            = 0.0015
        self._max_lr            = 0.003
        #self._decay_speed       = 10000
        self._decay_speed       = 2000
        self._data_pos          = 0

        if not os.path.exists(dataset._path+'record/'):
            os.mkdir(dataset._path+'record/')

        if not os.path.exists(dataset._path+'log/'):
            os.mkdir(dataset._path+'log/')          

        if not os.path.exists(self._log_path):
            os.makedirs(self._log_path, exist_ok=True)

        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path, exist_ok=True)

        if not os.path.exists(self._result_path):
            os.mkdir(self._result_path)

        # folder to save parameters for all datasets
        if not os.path.exists("./data/parameters/"):
            os.mkdir("./data/parameters/")
        # folder to save parameters of dataset
        if not os.path.exists("./data/parameters/{}".format(self._dataset._name)):
            os.mkdir("./data/parameters/{}".format(self._dataset._name))
        if not os.path.exists("./data/parameters/{}/{}".format(self._dataset._name,save_dir)):
            os.mkdir("./data/parameters/{}/{}".format(self._dataset._name,save_dir))

    def load_data(self):

        print("loading the data...")

        train_data, train_la, train_lu, test_data, test_la, test_lu = self._dataset.load_data(step=self._fold)
        self._train_data, self._train_la, self._train_lu = skutils.shuffle(train_data, train_la, train_lu)
        self._test_data = test_data
        self._test_la = test_la
        self._test_lu = test_lu

        print("finished data loading!")

        print("train data shape: {}\n".format(self._train_data.shape) +
              "train la shape: {}\n".format(self._train_la.shape) +
              "train lu shape: {}\n".format(self._train_lu.shape) +
              "test data shape: {}\n".format(self._test_data.shape) +
              "test la shape: {}\n".format(self._test_la.shape) +
              "test lu shape: {}\n".format(self._test_lu.shape))

    def one_hot(self, y, n_values):

        return np.eye(n_values)[np.array(y, dtype=np.int32)]

    def next_batch(self):

        train_size = self._train_data.shape[0]
        scale = self._data_pos+self._batch_size

        if scale > train_size:
            a = scale - train_size

            data1 = self._train_data[self._data_pos:]
            la1 = self._train_la[self._data_pos:]
            lu1 = self._train_lu[self._data_pos:]

            # shuffle after one cycle
            self._train_data, self._train_la, self._train_lu = skutils.shuffle(self._train_data, self._train_la, self._train_lu)

            data2 = self._train_data[: a]

            la2 = self._train_la[: a]
            lu2 = self._train_lu[: a]

            data = np.concatenate((data1, data2), axis=0)
            la = np.concatenate((la1, la2), axis=0)
            lu = np.concatenate((lu1, lu2), axis=0)

            self._data_pos = a
            return data, self.one_hot(la, self._dataset._train_act_num), self.one_hot(lu, self._dataset._train_user_num)
        else:
            data = self._train_data[self._data_pos: scale]
            la = self._train_la[self._data_pos: scale]
            lu = self._train_lu[self._data_pos: scale]
            self._data_pos = scale
            return data, self.one_hot(la, self._dataset._train_act_num), self.one_hot(lu, self._dataset._train_user_num)

    def build_model(self):

        self._is_training = tf.placeholder(dtype=tf.bool)
        self._learning_rate = tf.placeholder(dtype=tf.float32)
        self._X = tf.placeholder(dtype=tf.float32, shape=self._dataset._data_shape)
        self._YA = tf.placeholder(dtype=tf.int32, shape=[None, self._dataset._train_act_num])
        self._YU = tf.placeholder(dtype=tf.int32, shape=[None, self._dataset._train_user_num])

        self._a_loss_mean = tf.placeholder(dtype=tf.float32, shape=())
        self._u_loss_mean = tf.placeholder(dtype=tf.float32, shape=())

        if self._framework == 1:
            self._model = model.MTLMA_pretrain()
        elif self._framework == 2:
            self._model = model.MTLMA_train()
        else:
            print('model error!!!')
            exit(0)

        if self._framework == 1:
            a_preds, a_loss, u_preds, u_loss = self._model( self._X, self._YA, self._YU, self._dataset._train_act_num, self._dataset._train_user_num,
                                                            self._dataset._winlen, self._dataset._name, self._save_dir, self._fold, self._is_training)
            a_train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(a_loss, var_list=self._model.get_act_step_vars())
            u_train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(u_loss, var_list=self._model.get_user_step_vars())
        elif self._framework == 2:
            a_preds, a_loss, u_preds, u_loss, loss_global = self._model( self._X, self._YA, self._YU, self._dataset._train_act_num, self._dataset._train_user_num,
                                                            self._dataset._winlen, self._dataset._name, self._save_dir,self._fold, self._is_training)
            a_train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(loss_global, var_list=self._model.get_act_step_vars())
            u_train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(u_loss, var_list=self._model.get_user_step_vars())          

        # added train accuracy
        with tf.name_scope('train'):
            train_a_accuracy, train_a_accuracy_op = tf.metrics.accuracy(    labels=tf.argmax(self._YA,1), 
                                                                            predictions=tf.argmax(a_preds,1)
            )
            train_u_accuracy, train_u_accuracy_op = tf.metrics.accuracy(    labels=tf.argmax(self._YU,1), 
                                                                            predictions=tf.argmax(u_preds,1)
            )

        # added val accuracy
        with tf.name_scope('val'):
            val_a_accuracy, val_a_accuracy_op = tf.metrics.accuracy(    labels=tf.argmax(self._YA,1), 
                                                                        predictions=tf.argmax(a_preds,1)
            )
            val_u_accuracy, val_u_accuracy_op = tf.metrics.accuracy(    labels=tf.argmax(self._YU,1), 
                                                                        predictions=tf.argmax(u_preds,1)
            )


        learning_rate = tf.summary.scalar("learning rate", self._learning_rate)

        # aggiunto 
        with tf.name_scope('train'):
            train_activity_accuracy = summary_lib.scalar('a_accuracy', train_a_accuracy)
            train_user_accuracy = summary_lib.scalar('u_accuracy', train_u_accuracy)
            train_activity_loss = summary_lib.scalar('a_loss', a_loss)
            train_user_loss = summary_lib.scalar('u_loss', u_loss)

        with tf.name_scope('val'):
            val_activity_accuracy   = summary_lib.scalar('a_accuracy', val_a_accuracy)
            val_user_accuracy   = summary_lib.scalar('u_accuracy', val_u_accuracy)
            val_activity_loss    = summary_lib.scalar('a_loss', self._a_loss_mean)
            val_user_loss       = summary_lib.scalar('u_loss', self._u_loss_mean)

        #merged = tf.summary.merge_all()

        update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS)

        self._a_preds       = a_preds
        self._u_preds       = u_preds
        self._a_loss        = a_loss
        self._u_loss        = u_loss
        self._a_train_step  = a_train_step
        self._u_train_step  = u_train_step
        #self._merged        = merged
        self._update_ops    = update_ops
        self._lr            = learning_rate

        self._a_loss_train  = train_activity_loss
        self._u_loss_train  = train_user_loss
        self._a_loss_val    = val_activity_loss
        self._u_loss_val    = val_user_loss

        self._a_accuracy_op_train = train_a_accuracy_op
        self._u_accuracy_op_train = train_u_accuracy_op
        self._a_accuracy_op_val = val_a_accuracy_op
        self._u_accuracy_op_val = val_u_accuracy_op

        self._a_accuracy_train = train_activity_accuracy
        self._a_accuracy_val = val_activity_accuracy
        self._u_accuracy_train = train_user_accuracy
        self._u_accuracy_val = val_user_accuracy

    def predict(self, sess, lr):

        size = self._test_data.shape[0]
        batch_size = self._batch_size
        LAPreds = np.empty([0])
        LATruth = np.empty([0])
        LUPreds = np.empty([0])
        LUTruth = np.empty([0])
        a_loss_val = []
        u_loss_val = []

        for start, end in zip(range(0,           size,               batch_size),
                              range(batch_size,  size + batch_size,  batch_size)):

            end = end if end < size else size
            '''
            la_preds, lu_preds, = sess.run([self._a_preds, self._u_preds], feed_dict={
                self._X:            self._test_data[start: end],
                self._is_training:  False
            })

            '''
            la_preds, lu_preds, _, _, a_loss, u_loss = sess.run([self._a_preds, self._u_preds, self._a_accuracy_op_val, self._u_accuracy_op_val, self._a_loss, self._u_loss], feed_dict={
                self._X:            self._test_data[start: end],
                self._is_training:  False,
                self._YA:           self.one_hot(self._test_la[start : end], self._dataset._train_act_num),
                self._YU:           self.one_hot(self._test_lu[start : end], self._dataset._train_user_num),
                self._learning_rate: lr
            })     

            LAPreds = np.concatenate((LAPreds, np.argmax(la_preds, 1)))
            LATruth = np.concatenate((LATruth, self._test_la[start: end]))
            LUPreds = np.concatenate((LUPreds, np.argmax(lu_preds, 1)))
            LUTruth = np.concatenate((LUTruth, self._test_lu[start: end]))
            a_loss_val.append(a_loss)
            u_loss_val.append(u_loss)

        a_acc_val, u_acc_val = sess.run([self._a_accuracy_val, self._u_accuracy_val])
        a_loss, u_loss = sess.run([self._a_loss_val, self._u_loss_val], feed_dict={
            self._a_loss_mean: np.mean(a_loss_val),
            self._u_loss_mean: np.mean(u_loss_val)
        })

        return a_acc_val, u_acc_val, a_loss, u_loss, LATruth, LAPreds, LUTruth, LUPreds

    def save_paremeters(self, sess):

        print('Save paramter after pre-train')
        # import pdb; pdb.set_trace()
        for i in range(1, 4, 1):

            TensorA = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='act_network/a_conv{}'.format(i))
            TensorU = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='user_network/u_conv{}'.format(i))

            ParameterA, ParameterU = sess.run([TensorA, TensorU])

            np.save("./data/parameters/{}/{}/f{}a{}".format(self._dataset._name,self._save_dir,self._fold, i), ParameterA[0])
            np.save("./data/parameters/{}/{}/f{}u{}".format(self._dataset._name,self._save_dir,self._fold, i), ParameterU[0])

    def run_model(self):

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu selection
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 1  # 100% gpu
        sess_config.gpu_options.allow_growth = True      # dynamic growth

        #iter_steps = (self._train_data.shape[0] / self._batch_size) * self._epoch
        #epoch = 1
        tot = 0
        history = np.empty([0,5])


        with tf.Session(config=sess_config) as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            train_writer    = tf.summary.FileWriter( self._log_path + '/train', graph = sess.graph )
            test_write      = tf.summary.FileWriter( self._log_path + '/test',  graph = sess.graph )

            custom_layout= summary_lib.custom_scalar_pb(
                layout_pb2.Layout(category=[
                    layout_pb2.Category(
                        title='Accuracies',
                        chart=[
                            layout_pb2.Chart(
                                title='Activity',
                                multiline=layout_pb2.MultilineChartContent(tag=[
                                    r'train_1/a_accuracy', 
                                    r'val_1/a_accuracy'
                                ])),
                            layout_pb2.Chart(
                                title='User',
                                multiline=layout_pb2.MultilineChartContent(tag=[
                                    r'train_1/u_accuracy', 
                                    r'val_1/u_accuracy'
                                ])),
                        ]
                    ),
                    layout_pb2.Category(
                        title='Losses',
                        chart=[
                            layout_pb2.Chart(
                                title='Activity',
                                multiline=layout_pb2.MultilineChartContent(tag=[
                                    r'train_1/a_loss',
                                    r'val_1/a_loss'
                                ])),
                            layout_pb2.Chart(
                                title='User',
                                multiline=layout_pb2.MultilineChartContent(tag=[
                                    r'train_1/u_loss',
                                    r'val_1/u_loss'
                                ])),
                        ])
                ])
            )

            train_writer.add_summary(custom_layout)
                        
            # result_array    = np.empty( [0, 2, len( self._test_data )] )
            LARecord = np.empty([0, 2, self._test_data.shape[0]])
            LURecord = np.empty([0, 2, self._test_data.shape[0]])

            for i in range(self._iter_steps):

                data, la, lu = self.next_batch()
                lr = self._min_lr + (self._max_lr - self._min_lr) * math.exp(-i / self._decay_speed)
                tot += data.shape[0]
                '''
                if self._framework == 1:
                    summary, _, _, _, _, _ = sess.run([self._merged, self._update_ops, self._a_train_step, self._u_train_step, self._a_accuracy_op, self._u_accuracy_op], feed_dict={
                        self._X:                data,
                        self._YA:               la,
                        self._YU:               lu,
                        self._learning_rate:    lr,
                        self._is_training:      True})
                elif self._framework == 2:
                    summary, _, _, = sess.run([self._merged, self._update_ops, self._a_train_step], feed_dict={
                        self._X:                data,
                        self._YA:               la,
                        self._YU:               lu,
                        self._learning_rate:    lr,
                        self._is_training:      True})
                else:
                    print("model error")
                    exit()
                '''
                if self._framework == 1:
                    _, _, _, a_acc_train, u_acc_train, _, _, a_loss, u_loss, _lr = sess.run([    self._update_ops, self._a_train_step, self._u_train_step, self._a_accuracy_train, self._u_accuracy_train,
                                                                                            self._a_accuracy_op_train, self._u_accuracy_op_train, self._a_loss_train, self._u_loss_train, self._lr], 
                                                                                            feed_dict={
                                                                                                self._X:                data,
                                                                                                self._YA:               la,
                                                                                                self._YU:               lu,
                                                                                                self._learning_rate:    lr,
                                                                                                self._is_training:      True
                                                                                            }
                    )
                elif self._framework == 2:
                    _, _, a_acc_train, u_acc_train, _, _, a_loss, u_loss, _lr = sess.run([    self._update_ops, self._a_train_step, self._a_accuracy_train, self._u_accuracy_train,
                                                                                            self._a_accuracy_op_train, self._u_accuracy_op_train, self._a_loss_train, self._u_loss_train, self._lr], 
                                                                                            feed_dict={
                                                                                                self._X:                data,
                                                                                                self._YA:               la,
                                                                                                self._YU:               lu,
                                                                                                self._learning_rate:    lr,
                                                                                                self._is_training:      True
                                                                                            }
                    )
                else:
                    print("model error")
                    exit()

                train_writer.add_summary( a_acc_train, i )
                train_writer.add_summary( u_acc_train, i )
                train_writer.add_summary( a_loss, i)
                train_writer.add_summary( u_loss, i)
                train_writer.add_summary( _lr, i)

                if i % self._print_interval == 0:      

                    # added reset for validation metrics
                    stream_vars_valid = [v for v in tf.local_variables() if 'val/' in v.name]
                    sess.run(tf.variables_initializer(stream_vars_valid))             

                    a_acc_val, u_acc_val, a_loss, u_loss, LATruth, LAPreds, LUTruth, LUPreds = self.predict(sess, lr)

                    LARecord = np.append(LARecord, np.expand_dims(np.vstack((LATruth, LAPreds)), 0), axis=0)
                    LURecord = np.append(LURecord, np.expand_dims(np.vstack((LUTruth, LUPreds)), 0), axis=0)

                    AAccuracy = accuracy_score(LATruth, LAPreds, range(self._dataset._act_num))
                    Af1 = f1_score(LATruth, LAPreds, range(self._dataset._act_num), average='macro')

                    UAccuracy = accuracy_score(LUTruth, LUPreds, range(self._dataset._user_num))
                    Uf1 = f1_score(LUTruth, LUPreds, range(self._dataset._user_num), average='macro')

                    test_write.add_summary(a_acc_val, i)
                    test_write.add_summary(u_acc_val, i)
                    test_write.add_summary(a_loss, i)
                    test_write.add_summary(u_loss,i)
                    print("step: {},   AAccuracy: {},  Af1: {},  UAccuracy: {},  Uf1: {}".format(
                            i, AAccuracy, Af1, UAccuracy, Uf1))                   

                    history = np.concatenate((history, np.array([[i,AAccuracy,Af1,UAccuracy,Uf1]])), axis=0)

            if self._framework == 1:
                self.save_paremeters(sess)
                np.savetxt( self._result_path+'log_history_pre_train_{}.txt'.format(self._fold), 
                            history,
                            header='Step  AAaccuracy Af1 UAccuracy Uf1', 
                            fmt='%d %1.4f %1.4f %1.4f %1.4f',
                            delimiter='\t' )
                print('finish pretrain')
                    
            if self._framework == 2:

                # save log of train to file
                np.savetxt( self._result_path+'log_history_train_{}.txt'.format(self._fold), 
                            history,
                            header='Step  AAaccuracy Af1 UAccuracy Uf1', 
                            fmt='%d %1.4f %1.4f %1.4f %1.4f',
                            delimiter='\t' )

                LARecordFile = self._result_path + \
                    "AR_fold{}_".format(
                        self._fold) + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

                LURecordFile = self._result_path + \
                    "UR_fold{}_".format(
                        self._fold) + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

                np.save(LARecordFile, LARecord)
                np.save(LURecordFile, LURecord)
                print("finish train")

        tf.keras.backend.clear_session()


if __name__ == '__main__':

    parser  = argparse.ArgumentParser( description="deep MTL based activity and user recognition using wearable sensors" )

    # fold using for test in pretrain e train, repeat pretrain for every fold in cross-validation
    #parser.add_argument('-f', '--fold',         type=int,       default = 0     ) 
    #parser.add_argument('-d', '--dataset',      type=str,       default="unimib")
    parser.add_argument('-m', '--model',        type=int,       default = 2,        choices = [ 1, 2 ]  ) # 1: pretrain, 2: train

    args    = parser.parse_args()

    #for d in ['unimib', 'sbhar', 'realdisp']:
    for d in ['realdisp']:

        print('using {} dataset'.format(d))
    
        if d == "unimib":
            dataset = Dataset(  path='data/datasets/UNIMIBDataset/',
                                name='unimib',
                                channel=3,
                                winlen=100,
                                user_num=30,
                                act_num=9)
        elif d == "sbhar":
            dataset = Dataset(  path='data/datasets/SBHAR_processed/',
                                name='sbhar',
                                channel=6,
                                winlen=100,
                                user_num=30,
                                act_num=12) 
        elif d == "realdisp":
            dataset = Dataset(  path='data/datasets/REALDISP_processed/',
                                name='realdisp',
                                channel=9,
                                winlen=100,
                                user_num=17,
                                act_num=33,
                                save_dir='acc_gyro_magn/')


        for i in range(1):
            if args.model == 1:
                print('Pretrain with fold {} for test'.format(i))
                model_pretrain = my_model(  version="pre_train_acc_gyro_magn", gpu=0, fold=i, save_dir='acc_gyro_magn', 
                                            dataset=dataset, framework=1, iter_steps=10000)
                model_pretrain.load_data()
                model_pretrain.build_model()
                model_pretrain.run_model()
            elif args.model == 2:
                print('train with fold {} for test'.format(i))
                model_train = my_model(  version="train_acc_gyro_magn", gpu=0, fold=i, save_dir='acc_gyro_magn', 
                                            dataset=dataset, framework=2, iter_steps=50000)
                model_train.load_data()
                model_train.build_model()
                model_train.run_model()     
    
