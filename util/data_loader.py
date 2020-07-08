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

from util.data_augmentation import add_gaussian_noise, scaling_sequence, discriminative_guided_warp, wdba, random_guided_warp, random_transformation, random_guided_warp_multivariate

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

    def load_data(self, augmented=False, step=0, overlapping=0.5):

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

        print('train data before augmented: {}'.format(TrainData.shape))

        # adding augmentation to train data if set to true
        if augmented:

            #data_noisy = add_gaussian_noise(TrainData)

            random_data_transformed = []
            random_guided_warp_data = []

            random_guided_warp_data = random_guided_warp_multivariate(TrainData, labels_user=TrainLU, labels_activity=TrainLA, dtw_type='normal', use_window=False, magnitude=True)

            if random_guided_warp_data != []:
                TrainData = np.concatenate((TrainData, random_guided_warp_data), axis=0)
                TrainLA = np.tile(TrainLA, 2)
                TrainLU = np.tile(TrainLU, 2)

            random_data_transformed, lu, la = random_transformation(TrainData, TrainLU, TrainLA, use_magnitude=True, log=False)

            if random_data_transformed != []:
                TrainData = np.concatenate((TrainData, random_data_transformed), axis=0)
                TrainLA = np.append(TrainLA, la)
                TrainLU = np.append(TrainLU, lu)

        TrainData, TrainLA, TrainLU = skutils.shuffle(TrainData, TrainLA, TrainLU)
        TrainData, TrainLA, TrainLU = skutils.shuffle(TrainData, TrainLA, TrainLU)

        print('train data after augmented: {}'.format(TrainData.shape))
            
        # normalization
        mean = np.mean(np.reshape(TrainData, [-1, self._channel]), axis=0)
        std = np.std(np.reshape(TrainData, [-1, self._channel]), axis=0)

        TrainData = (TrainData - mean)/std
        TestData = (TestData - mean)/std

        TrainData = np.expand_dims(TrainData, 3)
        TestData = np.expand_dims(TestData,  3)

        if self._name != 'unimib_sbhar':
            return TrainData, TrainLA, TrainLU, TestData, TestLA, TestLU
        else:
            return TrainData, TrainLU, TestData, TestLU

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
