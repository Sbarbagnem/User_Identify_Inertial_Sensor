import os
import shutil
import zipfile

import numpy as np
from absl import app, flags
from scipy import stats
from scipy.fftpack import fft
from scipy.io import loadmat
from sklearn import utils as skutils

class Dataset( object ):

    def __init__( self, path, name, channel, winlen, user_num, act_num, save_dir= ""):
        self._path              = path 
        self._name              = name 
        self._channel           = channel 
        self._winlen            = winlen 
        self._user_num          = user_num 
        self._act_num           = act_num
        self._train_user_num    = user_num 
        self._train_act_num     = act_num
        self._data_shape        = [ None, self._winlen, self._channel, 1 ]
        self._save_dir          = save_dir

    def load_data( self, step = 0 ):

        TrainData   = np.empty( [0, self._winlen, self._channel], dtype=np.float )
        TrainLA     = np.empty( [0], dtype=np.int32 )
        TrainLU     = np.empty( [0], dtype=np.int32 )
        TrainID     = np.empty( [0], dtype=np.int32 )

        TestData   = np.empty( [0, self._winlen, self._channel], dtype=np.float )
        TestLA     = np.empty( [0], dtype=np.int32 )
        TestLU     = np.empty( [0], dtype=np.int32 )
        TestID     = np.empty( [0], dtype=np.int32 )

        for i in range(10):
            Data    = np.load( self._path + 'OuterPartition/' + self._save_dir + 'fold{}/data.npy'.format(i) )
            LA      = np.load( self._path + 'OuterPartition/' + self._save_dir + 'fold{}/act_label.npy'.format(i) )
            LU      = np.load( self._path + 'OuterPartition/' + self._save_dir + 'fold{}/user_label.npy'.format(i) )
            ID      = np.load( self._path + 'OuterPartition/' + self._save_dir + 'fold{}/id.npy'.format(i) )

            if step == i:
                TestData   = np.concatenate( (TestData, Data), axis=0 )
                TestLA     = np.concatenate( (TestLA, LA), axis=0 )
                TestLU     = np.concatenate( (TestLU, LU), axis=0 )
                TestID     = np.concatenate( (TestID, ID), axis=0 )
            else:
                TrainData   = np.concatenate( (TrainData, Data), axis=0 )
                TrainLA     = np.concatenate( (TrainLA, LA), axis=0 )
                TrainLU     = np.concatenate( (TrainLU, LU), axis=0 )
                TrainID     = np.concatenate( (TrainID, ID), axis=0 )

        # delete overlap samples form training_data
        overlap_ID  = np.unique( np.concatenate( ( TestID+1, TestID-1) ) )
        invalid_idx = np.array( [ i for i in np.arange( len(TrainID) ) if TrainID[i] in overlap_ID ] )

        TrainData  = np.delete( TrainData, invalid_idx, axis=0 )
        TrainLA    = np.delete( TrainLA,   invalid_idx, axis=0 )
        TrainLU    = np.delete( TrainLU,   invalid_idx, axis=0 )
        # deleted

        # normalization
        mean    = np.mean( np.reshape(  TrainData, [-1, self._channel] ), axis=0 )
        std     = np.std( np.reshape(   TrainData, [-1, self._channel] ), axis=0 )

        TrainData  = (TrainData - mean)/std
        TestData   = (TestData - mean)/std

        TrainData  = np.expand_dims( TrainData, 3 )
        TestData   = np.expand_dims( TestData,  3 )

        return TrainData, TrainLA, TrainLU, TestData, TestLA, TestLU