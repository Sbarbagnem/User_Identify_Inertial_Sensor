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
from util.utils import split_balanced_data, delete_overlap, to_delete

#from util.data_augmentation import add_gaussian_noise, scaling_sequence, discriminative_guided_warp, random_guided_warp, random_transformation, random_guided_warp_multivariate
from util.data_augmentation import random_transformation, random_guided_warp_multivariate, jitter, compute_sub_seq


class Dataset(object):

    def __init__(self, path, channel, winlen, user_num, act_num, outer_dir='OuterPartition/'):
        self._path = path
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

    def load_data(self, fold_test, overlapping, realdisp=False):

        """
        Load train, validation and test data based on path and datataset name passed in Dataset object.
        
        Parameters
        ----------
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
        if realdisp:
            TrainDI = np.empty([0], dtype=np.int32)

        # test data
        TestData = np.empty([0, self._winlen, self._channel], dtype=np.float)
        TestLA = np.empty([0], dtype=np.int32)
        TestLU = np.empty([0], dtype=np.int32)
        TestID = np.empty([0], dtype=np.int32)
        if realdisp:
            TestDI = np.empty([0], dtype=np.int32)

        for i in range(10):
            Data = np.load(self._path + self.outer_dir +
                           'fold{}/data.npy'.format(i))
            LA = np.load(self._path + self.outer_dir +
                         'fold{}/act_label.npy'.format(i))
            LU = np.load(self._path + self.outer_dir +
                         'fold{}/user_label.npy'.format(i))
            ID = np.load(self._path + self.outer_dir +
                         'fold{}/id.npy'.format(i))
            if realdisp:
                DI = np.load(self._path + self.outer_dir +
                            'fold{}/di.npy'.format(i))

            if i == fold_test:
                TestData = np.concatenate((TestData, Data), axis=0)
                TestLA = np.concatenate((TestLA, LA), axis=0)
                TestLU = np.concatenate((TestLU, LU), axis=0)
                TestID = np.concatenate((TestID, ID), axis=0)    
                if realdisp:
                    TestDI = np.concatenate((TestDI, DI), axis=0)
            else:
                TrainData = np.concatenate((TrainData, Data), axis=0)
                TrainLA = np.concatenate((TrainLA, LA), axis=0)
                TrainLU = np.concatenate((TrainLU, LU), axis=0)
                TrainID = np.concatenate((TrainID, ID), axis=0)
                if realdisp:
                    TrainDI = np.concatenate((TrainDI, DI), axis=0)  

        print('Shape train data before delete overlap sequence: ', TrainData.shape)

        # delete overlap samples between train and test based on overlap percentage
        distances_to_delete = to_delete(overlapping)
        invalid_idx = delete_overlap(TrainID, TestID, distances_to_delete)
        TrainData = np.delete(TrainData, invalid_idx, axis=0)
        TrainLA = np.delete(TrainLA, invalid_idx, axis=0)
        TrainLU = np.delete(TrainLU, invalid_idx, axis=0)
        TrainID = np.delete(TrainID, invalid_idx, axis=0)
        if realdisp:
            TrainDI = np.delete(TrainDI, invalid_idx, axis=0)

        print('Shape train data after deleted overlap sequence from test set: ', TrainData.shape)

        # split train data in 80% train and 20% validation
        if realdisp:
            TrainData, TrainLU, TrainLA, TrainID, TrainDI = skutils.shuffle(TrainData, TrainLU, TrainLA, TrainID, TrainDI)
            indexes = split_balanced_data(TrainLU, TrainLA, folders=5, di=TrainDI, log=False)
        else:
            TrainData, TrainLU, TrainLA, TrainID = skutils.shuffle(TrainData, TrainLU, TrainLA, TrainID)
            indexes = split_balanced_data(TrainLU, TrainLA, folders=5, log=False)           

        idx_val = indexes[str(0)]
        idx_train = [indexes[str(i)] for i in [1,2,3,4]]
        idx_train = [y for x in idx_train for y in x]

        ValidData = TrainData[idx_val,:,:]
        ValidLA = TrainLA[idx_val]
        ValidLU = TrainLU[idx_val]
        if realdisp:
            ValidDI = TrainDI[idx_val]

        TrainData = TrainData[idx_train,:,:]
        TrainLA = TrainLA[idx_train]
        TrainLU = TrainLU[idx_train]
        if realdisp:
            TrainDI = TrainDI[idx_train]

        print('Shape train data after split train and val: ', TrainData.shape)
        print('Shape val data : ', ValidData.shape)
        print('Shape test data: ', TestData.shape)

        if len(np.intersect1d(TrainID, TestID))>0:
            sys.exit('there is overlap between train/val and test, this affect (improve) performance on test')

        if realdisp:
            return TrainData, TrainLA, TrainLU, TrainDI, ValidData, ValidLA, ValidLU, ValidDI, TestData, TestLA, TestLU, TestDI
        else:
            return TrainData, TrainLA, TrainLU, ValidData, ValidLA, ValidLU, TestData, TestLA, TestLU

    def normalize_data(self, train, val, test=None):

        # normalization
        mean = np.mean(np.reshape(train, [-1, self._channel]), axis=0)
        std = np.std(np.reshape(train, [-1, self._channel]), axis=0)

        train = (train - mean)/std
        val = (val - mean)/std
        test = (test - mean)/std

        train = np.expand_dims(train, 3)
        val = np.expand_dims(val,  3)     
        test = np.expand_dims(test, 3)
        return train, val, test

    def augment_data(self, data, lu, la, magnitude, augmented_par, function_to_apply, compose, only_compose, plot_augmented, ratio_random_transformations, n_func_to_apply, n_sensor):
        if augmented_par != []:
            for t,ratio in zip(augmented_par, ratio_random_transformations):
                if t == 'random_transformations': 
                    print('\n apply random transformations \n')
                    train_aug, lu_aug, la_aug = self.augmented_fun['random_transformations'](data,
                                                                                           lu,
                                                                                           la,
                                                                                           n_axis=self._channel,
                                                                                           n_sensor=n_sensor,
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
