import argparse
import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from scipy.fftpack import fft
from sklearn import utils as skutils
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from model import model
from util.data_loader import Dataset

#import tensorflow.contrib.slim as slim
#from sklearn.neighbors import KNeighborsClassifier

tf.compat.v1.disable_eager_execution()


class my_model(object):

    def __init__(self, version, gpu, fold, save_dir, dataset, framework, epochs):

        self._dataset = dataset
        self._gpu = gpu
        #self._log_path  = dataset._path+'log/'+version+'/'
        self._fold = fold % 10
        self._save_path = dataset._path+'record/'+save_dir
        self._framework = framework

        #self._iter_steps        = 50000
        #self._print_interval    = 100
        #self._batch_size        = 100
        self._epoch = epochs
        self._batch_size = 128
        self._min_lr = 0.0005
        self._max_lr = 0.0015
        self._decay_speed = 10000
        self._data_pos = 0

        if not os.path.exists(dataset._path+'record/'):
            os.mkdir(dataset._path+'record/')

        if not os.path.exists(self._save_path):
            os.mkdir(self._save_path)

        self._result_path = self._save_path + "/"
        if not os.path.exists(self._result_path):
            os.mkdir(self._result_path)

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

        self._is_training = tf.compat.v1.placeholder(dtype=tf.bool)
        self._learning_rate = tf.compat.v1.placeholder(dtype=tf.float32)
        self._X = tf.compat.v1.placeholder(dtype=tf.float32, shape=self._dataset._data_shape)
        self._YA = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, self._dataset._train_act_num])
        self._YU = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, self._dataset._train_user_num])

        if self._framework == 1:
            self._model = model.MTLMA_pretrain()
        elif self._framework == 2:
            self._model = model.MTLMA_train()
        else:
            print('model error!!!')
            exit(0)

        a_preds, a_loss, u_preds, u_loss = self._model(self._X, self._YA, self._YU, self._dataset._train_act_num, self._dataset._train_user_num,
                                                       self._dataset._winlen, self._dataset._name, self._fold, self._is_training)
        a_train_step = tf.compat.v1.train.AdamOptimizer(self._learning_rate).minimize(a_loss, var_list=self._model.get_act_step_vars())
        u_train_step = tf.compat.v1.train.AdamOptimizer(self._learning_rate).minimize(u_loss, var_list=self._model.get_user_step_vars())

        tf.compat.v1.summary.scalar("learning rate", self._learning_rate)

        merged = tf.compat.v1.summary.merge_all()

        update_ops = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS)

        self._a_preds = a_preds
        self._u_preds = u_preds
        self._a_train_step = a_train_step
        self._u_train_step = u_train_step
        self._merged = merged
        self._update_ops = update_ops

    def predict(self, sess):

        size = self._test_data.shape[0]
        batch_size = self._batch_size
        LAPreds = np.empty([0])
        LATruth = np.empty([0])
        LUPreds = np.empty([0])
        LUTruth = np.empty([0])

        for start, end in zip(range(0,           size,               batch_size),
                              range(batch_size,  size + batch_size,  batch_size)):

            end = end if end < size else size

            la_preds, lu_preds = sess.run([self._a_preds, self._u_preds], feed_dict={
                self._X:            self._test_data[start: end],
                self._is_training:  False
            })

            LAPreds = np.concatenate((LAPreds, np.argmax(la_preds, 1)))
            LATruth = np.concatenate((LATruth, self._test_la[start: end]))
            LUPreds = np.concatenate((LUPreds, np.argmax(lu_preds, 1)))
            LUTruth = np.concatenate((LUTruth, self._test_lu[start: end]))

        return LATruth, LAPreds, LUTruth, LUPreds

    def save_paremeters(self, sess):

        print('Save paramter after pre-train')
        # import pdb; pdb.set_trace()
        for i in range(1, 4, 1):

            TensorA = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='act_network/a_conv{}'.format(i))
            TensorU = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='user_network/u_conv{}'.format(i))

            ParameterA, ParameterU = sess.run([TensorA, TensorU])

            # folder to save parameters for all datasets
            if not os.path.exists("./data/parameters/"):
                os.mkdir("./data/parameters/")
            # folder to save parameters of dataset
            if not os.path.exists("./data/parameters/{}".format(self._dataset._name)):
                os.mkdir("./data/parameters/{}".format(self._dataset._name))

            np.save("./data/parameters/{}/f{}a{}".format(self._dataset._name,self._fold, i), ParameterA[0])
            np.save("./data/parameters/{}/f{}u{}".format(self._dataset._name,self._fold, i), ParameterU[0])

    def run_model(self):

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu selection
        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 1  # 100% gpu
        sess_config.gpu_options.allow_growth = True      # dynamic growth

        iter_steps = (self._train_data.shape[0] / self._batch_size) * self._epoch
        epoch = 1
        tot = 0
        history = np.empty([0,5])


        with tf.compat.v1.Session(config=sess_config) as sess:

            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            #train_writer    = tf.compat.v1.summary.FileWriter( self._log_path + '/train', graph = tf.compat.v1.get_default_graph() )
            # result_array    = np.empty( [0, 2, len( self._test_data )] )
            LARecord = np.empty([0, 2, self._test_data.shape[0]])
            LURecord = np.empty([0, 2, self._test_data.shape[0]])

            for i in range(int(iter_steps)):

                data, la, lu = self.next_batch()
                lr = self._min_lr + (self._max_lr - self._min_lr) * math.exp(-i / self._decay_speed)
                tot += data.shape[0]

                if self._framework == 1:
                    _, _, _, _ = sess.run([self._merged, self._update_ops, self._a_train_step, self._u_train_step], feed_dict={
                        self._X:                data,
                        self._YA:               la,
                        self._YU:               lu,
                        self._learning_rate:    lr,
                        self._is_training:      True})
                elif self._framework == 2:
                    _, _, _ = sess.run([self._merged, self._update_ops, self._a_train_step], feed_dict={
                        self._X:                data,
                        self._YA:               la,
                        self._YU:               lu,
                        self._learning_rate:    lr,
                        self._is_training:      True})
                else:
                    print("model error")
                    exit()

                #train_writer.add_summary( summary, i )

                if (tot/self._train_data.shape[0]) >= 1:

                    LATruth, LAPreds, LUTruth, LUPreds = self.predict(sess)

                    LARecord = np.append(LARecord, np.expand_dims(np.vstack((LATruth, LAPreds)), 0), axis=0)
                    LURecord = np.append(LURecord, np.expand_dims(np.vstack((LUTruth, LUPreds)), 0), axis=0)

                    AAccuracy = accuracy_score(LATruth, LAPreds, range(self._dataset._act_num))
                    Af1 = f1_score(LATruth, LAPreds, range(self._dataset._act_num), average='macro')

                    UAccuracy = accuracy_score(LUTruth, LUPreds, range(self._dataset._user_num))
                    Uf1 = f1_score(LUTruth, LUPreds, range(self._dataset._user_num), average='macro')

                    print("epoch: {}, step: {},   AAccuracy: {},  Af1: {},  UAccuracy: {},  Uf1: {}".format(
                        epoch, i, AAccuracy, Af1, UAccuracy, Uf1))

                    if self._framework == 2:
                        history = np.concatenate((history, np.array([[epoch,AAccuracy,Af1,UAccuracy,Uf1]])), axis=0)

                    epoch += 1
                    tot = 0

            if self._framework == 1:
                self.save_paremeters(sess)
                print('finish pretrain')
                    
            if self._framework == 2:

                # save log of train to file
                np.savetxt( self._result_path+'log_history_train.txt', 
                            history,
                            header='Epoch  AAaccuracy Af1 UAccuracy Uf1', 
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
    parser.add_argument('-m', '--model',        type=int,       default = 1,        choices = [ 1, 2 ]  ) # 1: pretrain, 2: train

    args    = parser.parse_args()

    for d in ['unimib', 'sbhar', 'realdisp']:

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
                                channel=6,
                                winlen=100,
                                user_num=17,
                                act_num=33)


        for i in range(1):
            if args.model == 1:
                print('Pretrain with fold {} for test'.format(i))
                model_pretrain = my_model(  version="", gpu=-1, fold=i, save_dir='', 
                                            dataset=dataset, framework=1, epochs=1)
                model_pretrain.load_data()
                model_pretrain.build_model()
                model_pretrain.run_model()
            elif args.model == 2:
                print('train with fold {} for test'.format(i))
                model_pretrain = my_model(  version="", gpu=-1, fold=args.fold, save_dir='', 
                                            dataset=dataset, framework=1, epochs=1)
                model_pretrain.load_data()
                model_pretrain.build_model()
                model_pretrain.run_model()     
    
