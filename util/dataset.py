import os
import shutil
import zipfile
import sys

import numpy as np
from absl import app, flags
from scipy import stats
from scipy.fftpack import fft
from scipy.io import loadmat
from sklearn import utils as skutils
from configuration import config

from util.data_augmentation import add_gaussian_noise, scaling_sequence, discriminative_guided_warp, random_guided_warp, random_transformation, random_guided_warp_multivariate

class Dataset(object):

    def __init__(self, path, name, channel, winlen, user_num, act_num, save_dir="", outer_dir='OuterPartition/'):
        self._path = path
        self._name = name
        self._channel = channel
        self._winlen = winlen
        self._user_num = user_num
        self._act_num = act_num
        self._train_user_num = user_num
        self._train_act_num = act_num
        self._data_shape = [None, self._winlen, self._channel, 1]
        self._save_dir = save_dir
        self.outer_dir = outer_dir
        self.augmented_fun = {
            'random_transformations': random_transformation,
            'random_warped': random_guided_warp_multivariate
        }

    def load_data(self, step=0, overlapping=0.5, normalize=True):

        TrainData = np.empty([0, self._winlen, self._channel], dtype=np.float)
        TrainLA = np.empty([0], dtype=np.int32)
        TrainLU = np.empty([0], dtype=np.int32)
        TrainID = np.empty([0], dtype=np.int32)

        TestData = np.empty([0, self._winlen, self._channel], dtype=np.float)
        TestLA = np.empty([0], dtype=np.int32)
        TestLU = np.empty([0], dtype=np.int32)
        TestID = np.empty([0], dtype=np.int32)

        for i in range(10):
            Data = np.load(self._path + self.outer_dir +
                           self._save_dir + 'fold{}/data.npy'.format(i))
            LA = np.load(self._path + self.outer_dir +
                        self._save_dir + 'fold{}/act_label.npy'.format(i))
            LU = np.load(self._path + self.outer_dir +
                         self._save_dir + 'fold{}/user_label.npy'.format(i))
            ID = np.load(self._path + self.outer_dir +
                         self._save_dir + 'fold{}/id.npy'.format(i))

            if i in step:
                TestData = np.concatenate((TestData, Data), axis=0)
                TestLA = np.concatenate((TestLA, LA), axis=0)
                TestLU = np.concatenate((TestLU, LU), axis=0)
                TestID = np.concatenate((TestID, ID), axis=0)
            else:
                TrainData = np.concatenate((TrainData, Data), axis=0)
                TrainLA = np.concatenate((TrainLA, LA), axis=0)
                TrainLU = np.concatenate((TrainLU, LU), axis=0)
                TrainID = np.concatenate((TrainID, ID), axis=0)

        print('before delete: ', TrainData.shape)

        # delete overlap samples form training_data, based on overlap percentage
        distances_to_delete = to_delete(overlapping)
        print('distance to delete: ', distances_to_delete)
        overlap_ID = np.empty([0], dtype=np.int32)
        for distance in distances_to_delete:
            overlap_ID = np.concatenate((overlap_ID, TestID+distance, TestID-distance))
        overlap_ID = np.unique(overlap_ID)
        invalid_idx = np.array([i for i in np.arange(
            len(TrainID)) if TrainID[i] in overlap_ID])

        TrainData = np.delete(TrainData, invalid_idx, axis=0)
        print('after delete: ', TrainData.shape)
        TrainLA = np.delete(TrainLA,   invalid_idx, axis=0)
        TrainLU = np.delete(TrainLU,   invalid_idx, axis=0)

        print('train data shape: {}'.format(TrainData.shape))

        TrainData, TrainLA, TrainLU = skutils.shuffle(TrainData, TrainLA, TrainLU)
        TrainData, TrainLA, TrainLU = skutils.shuffle(TrainData, TrainLA, TrainLU)
            
        # normalization
        if normalize:
            TrainData, TestData = self.normalize_data(TrainData, TestData)

        return TrainData, TrainLA, TrainLU, TestData, TestLA, TestLU

    def normalize_data(self, train, test):

        # normalization
        mean = np.mean(np.reshape(train, [-1, self._channel]), axis=0)
        std = np.std(np.reshape(train, [-1, self._channel]), axis=0)

        train = (train - mean)/std
        test = (test - mean)/std

        train = np.expand_dims(train, 3)
        test = np.expand_dims(test,  3)

        return train, test

    def augment_data(self, data, lu, la, magnitude, augmented_par, plot_augmented):

        if augmented_par != []:
            train_augmented = np.copy(data)
            label_user_augmented = np.copy(lu)
            label_act_augmented = np.copy(la)

            for fun in augmented_par:
                if fun == 'random_transformations':
                    train, lu, la = self.augmented_fun[fun](data, 
                                                            lu, 
                                                            la,
                                                            n_axis=self._channel,
                                                            n_sensor=len(config[self._name]['SENSOR_DICT']),
                                                            use_magnitude=magnitude,
                                                            log=plot_augmented) 

                if fun == 'random_warped':
                    train, lu, la = self.augmented_fun[fun](data,
                                                            lu,
                                                            la,
                                                            dtw_type='normal',
                                                            use_window=False,
                                                            magnitude=magnitude,
                                                            log=plot_augmented)

                train_augmented = np.concatenate((train_augmented, train), axis=0)
                label_user_augmented = np.concatenate((label_user_augmented, lu), axis=0)
                label_act_augmented = np.concatenate((label_act_augmented, la), axis=0)

                return train_augmented, label_user_augmented, label_act_augmented

def to_delete(overlapping):
    if overlapping == 5.0:
        return [1]
    if overlapping == 6.0:
        return [1,2]
    if overlapping == 7.0:
        return [1,2,3]
    if overlapping == 8.0:
        return [1,2,3,4]
    if overlapping == 9.0:
        return [1,2,3,4,5,6,7,8,9]
