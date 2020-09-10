import os
import shutil
import zipfile
import sys
from tqdm import tqdm
import pprint

import numpy as np
from absl import app, flags
from scipy import stats
from scipy.fftpack import fft
from scipy.io import loadmat
from sklearn import utils as skutils

#from util.data_augmentation import add_gaussian_noise, scaling_sequence, discriminative_guided_warp, random_guided_warp, random_transformation, random_guided_warp_multivariate
from util.data_augmentation import random_transformation, random_guided_warp_multivariate, jitter, compute_sub_seq


class Dataset(object):

    def __init__(self, path, name, channel, winlen, user_num, act_num, config_file, outer_dir='OuterPartition/'):
        self._path = path
        self._name = name
        self._channel = channel
        self._winlen = winlen
        self._user_num = user_num
        self._act_num = act_num
        self._train_user_num = user_num
        self._train_act_num = act_num
        self._data_shape = [None, self._winlen, self._channel, 1]
        self.outer_dir = outer_dir
        self.augmented_fun = {
            'random_transformations': random_transformation,
            'random_warped': random_guided_warp_multivariate
        }
        self.config_file = config_file

    def load_data(self, fold_val, fold_test, overlapping):

        """
        Load train, validation and test data based on path and datataset name passed in Dataset object.
        
        Parameters
        ----------
        fold_val : list[int]
            List of fold used for validation.
        fold_test : list[int]
            List of fold used for test.
        overlapping : float -> 5.0, 6.0, ....
            Represent percentage of overlapping between sequences.

        """
        
        # train data
        TrainData = np.empty([0, self._winlen, self._channel], dtype=np.float)
        TrainLA = np.empty([0], dtype=np.int32)
        TrainLU = np.empty([0], dtype=np.int32)
        TrainID = np.empty([0], dtype=np.int32)

        # validation data
        ValidData = np.empty([0, self._winlen, self._channel], dtype=np.float)
        ValidLA = np.empty([0], dtype=np.int32)
        ValidLU = np.empty([0], dtype=np.int32)
        ValidID = np.empty([0], dtype=np.int32)

        # test data
        if fold_test != []:
            TestData = np.empty([0, self._winlen, self._channel], dtype=np.float)
            TestLA = np.empty([0], dtype=np.int32)
            TestLU = np.empty([0], dtype=np.int32)
            TestID = np.empty([0], dtype=np.int32)

        for i in range(10):
            Data = np.load(self._path + self.outer_dir +
                           'fold{}/data.npy'.format(i))
            LA = np.load(self._path + self.outer_dir +
                         'fold{}/act_label.npy'.format(i))
            LU = np.load(self._path + self.outer_dir +
                         'fold{}/user_label.npy'.format(i))
            ID = np.load(self._path + self.outer_dir +
                         'fold{}/id.npy'.format(i))

            if i in fold_test:
                TestData = np.concatenate((TestData, Data), axis=0)
                TestLA = np.concatenate((TestLA, LA), axis=0)
                TestLU = np.concatenate((TestLU, LU), axis=0)
                TestID = np.concatenate((TestID, ID), axis=0)
            elif i in fold_val:
                ValidData = np.concatenate((ValidData, Data), axis=0)
                ValidLA = np.concatenate((ValidLA, LA), axis=0)
                ValidLU = np.concatenate((ValidLU, LU), axis=0)
                ValidID = np.concatenate((ValidID, ID), axis=0)               
            else:
                TrainData = np.concatenate((TrainData, Data), axis=0)
                TrainLA = np.concatenate((TrainLA, LA), axis=0)
                TrainLU = np.concatenate((TrainLU, LU), axis=0)
                TrainID = np.concatenate((TrainID, ID), axis=0)

        # delete overlap samples form training_data, based on overlap percentage
        print('Shape train data before delete overlap sequence: ', TrainData.shape)
        distances_to_delete = to_delete(overlapping)
        overlap_ID = np.empty([0], dtype=np.int32)

        for distance in distances_to_delete:
            '''
            if fold_test != []:
                overlap_ID = np.concatenate(
                    (overlap_ID, TestID+distance, TestID-distance, ValidID+distance, ValidID-distance))
            else:
                overlap_ID = np.concatenate(
                    (overlap_ID, ValidID+distance, ValidID-distance))  
            '''
        overlap_ID = np.concatenate((overlap_ID, TestID+distance, TestID-distance))
        overlap_ID = np.unique(overlap_ID)
        invalid_idx = np.array([i for i in np.arange(
            len(TrainID)) if TrainID[i] in overlap_ID])

        TrainData = np.delete(TrainData, invalid_idx, axis=0)
        TrainLA = np.delete(TrainLA,   invalid_idx, axis=0)
        TrainLU = np.delete(TrainLU,   invalid_idx, axis=0)

        print('Shape train data after deleted overlap sequence: ', TrainData.shape)

        TrainData, TrainLA, TrainLU = skutils.shuffle(
            TrainData, TrainLA, TrainLU)
        TrainData, TrainLA, TrainLU = skutils.shuffle(
            TrainData, TrainLA, TrainLU)

        print('Shape val data: ', ValidData.shape)

        if fold_test != []:
            print('Shape test data: ', TestData.shape)
            return TrainData, TrainLA, TrainLU, ValidData, ValidLA, ValidLU, TestData, TestLA, TestLU
        else:
            return TrainData, TrainLA, TrainLU, ValidData, ValidLA, ValidLU, None, None, None

    def normalize_data(self, train, val, test=None):

        # normalization
        mean = np.mean(np.reshape(train, [-1, self._channel]), axis=0)
        std = np.std(np.reshape(train, [-1, self._channel]), axis=0)

        train = (train - mean)/std
        val = (val - mean)/std

        train = np.expand_dims(train, 3)
        val = np.expand_dims(val,  3)

        if test is not None:
            test = (test - mean)/std
            test = np.expand_dims(test, 3)

        if test is not None:
            return train, val, test
        else:
            return train, val, None

    def augment_data(self, data, lu, la, magnitude, augmented_par, function_to_apply, compose, only_compose, plot_augmented, ratio_random_transformations, n_func_to_apply):
        if augmented_par != []:
            for t,ratio in zip(augmented_par, ratio_random_transformations):
                if t == 'random_transformations': 
                    print('\n apply random transformations \n')
                    train_aug, lu_aug, la_aug = self.augmented_fun['random_transformations'](data,
                                                                                           lu,
                                                                                           la,
                                                                                           n_axis=self._channel,
                                                                                           n_sensor=len(
                                                                                               self.config_file.config[self._name]['SENSOR_DICT']),
                                                                                           use_magnitude=magnitude,
                                                                                           log=plot_augmented,
                                                                                           ratio=ratio,
                                                                                           compose=compose,
                                                                                           only_compose=only_compose,
                                                                                           function_to_apply=function_to_apply,
                                                                                           n_func_to_apply=n_func_to_apply)
                if t == 'random_warped':
                    print('\n apply random warped \n')
                    train_aug, lu_aug, la_aug = self.augmented_fun['random_warped'](data,
                                                                                  lu,
                                                                                  la,
                                                                                  dtw_type='normal', # normal or shape
                                                                                  use_window=False,
                                                                                  magnitude=magnitude,
                                                                                  ratio=ratio,
                                                                                  log=plot_augmented)
                data = np.concatenate(
                    (data, train_aug), axis=0)
                lu = np.concatenate(
                    (lu, lu_aug), axis=0)
                la = np.concatenate(
                    (la, la_aug), axis=0)

            return data, lu, la

    def unify_act_class(self, act_train, act_test, mapping):
        num_class_return = mapping['NUM_CLASSES_ACTIVITY']

        for act in mapping['mapping'].keys():
            for to_merge in mapping['mapping'][act]:
                act_train = np.where(act_train == to_merge, act, act_train)
                act_test = np.where(act_test == to_merge, act, act_test)


        return num_class_return, act_train, act_test

def to_delete(overlapping):

    """
    Return a list of distance to overlapping sequence.
    
    Parameters
    ----------
    overlapping : float
        Overlap percentage used.

    """

    if overlapping == 5.0:
        return [1]
    if overlapping == 6.0:
        return [1, 2]
    if overlapping == 7.0:
        return [1, 2, 3]
    if overlapping == 8.0:
        return [1, 2, 3, 4]
    if overlapping == 9.0:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9]
