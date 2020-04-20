import os
import shutil
import zipfile
import argparse

import numpy as np
from scipy import stats
from scipy.io import loadmat
from sklearn import utils as skutils

from sliding_window import sliding_window

def preprocessing(dataset, path, path_out):
    if dataset == 'unimib':
        unimib_process(path, path_out)
    else:
        pass
        # TODO add call to other datasets

def unimib_process(path, path_out):

    root_path       = path + 'unimib_dataset'
    raw_data_path   = root_path + '/raw/'
    processed_path  = path_out + 'OuterPartition'

    win_len         = 100   # 50 Hz * 2 seconds
    channel         = 3   #
    ID_generater    = 1     

    data    = np.empty( [0, win_len, channel], dtype=np.float )
    la      = np.empty( [0], dtype=np.int32 )
    lu      = np.empty( [0], dtype=np.int32 )
    ID      = np.empty( [0], dtype=np.int32 )

    signal  = loadmat( raw_data_path + 'full_data.mat' )['full_data']

    activity_table  = [ 'StandingUpFS', 'StandingUpFL', 'Walking', 'Running', 'GoingUpS', 'Jumping', 'GoingDownS', 'LyingDownFS', 'SittingDown',
                        'FallingForw', 'FallingRight', 'FallingBack', 'HittingObstacle', 'FallingWithPS', 'FallingBackSC', 'Syncope', 'FallingLeft' ]

    # id subject
    for sid in range( 30 ):
        # id activity
        for aid in range( 9 ):
            act = activity_table[ aid ]
            trials = ( signal[sid, 0][0, 0][act].shape )[0]        
            # id trial 
            for tid in range( trials ):

                # take only x,y,z axis
                _data = signal[sid, 0][0, 0][act][tid,0][0:3, :]
                _data = np.transpose( _data )

                _data_windows   = sliding_window( _data, ( win_len, channel ), ( int( win_len/2 ), 1 ) )
                invalid_idx     = np.where( np.any( np.isnan( np.reshape( _data_windows, [-1, win_len*channel] ) ), axis=1 ) )[0]
                _data_windows   = np.delete( _data_windows, invalid_idx, axis=0 )
                _id             = np.arange( ID_generater, ID_generater+len(_data_windows) ) # id for every window

                data            = np.concatenate( (data, _data_windows), axis=0 ) # concat verticaly every window
                ID              = np.concatenate( (ID,   _id), axis=0 )
                ID_generater    = ID_generater + len( _data_windows ) + 10

            _la     = np.full( len(data)-len(la) , aid, dtype=np.int32 ) # label activity for every window
            la      = np.concatenate( (la, _la), axis=0 )
        
        _lu     = np.full( len(data)-len(lu) , sid, dtype=np.int32 ) # label user for every window
        lu      = np.concatenate( (lu, _lu), axis=0 )

    # shuffle
    data, la, lu, ID    = skutils.shuffle( data, la, lu, ID )
    data, la, lu, ID    = skutils.shuffle( data, la, lu, ID )

    if not os.path.exists( processed_path + '/'):
        os.mkdir( processed_path + '/' )

    # partition
    for i in range( 10 ):

        # clear dir
        if os.path.exists( processed_path+'/fold{}'.format(i) ):
            shutil.rmtree( processed_path+'/fold{}'.format(i) )
        os.mkdir( processed_path+'/fold{}'.format(i) )

        idx    = np.arange( int( len(data)*0.1*i ), int( len(data)*0.1*(i+1) ), 1 )
        np.save( processed_path+'/fold{}/data'.format(i),       data[idx] )
        np.save( processed_path+'/fold{}/user_label'.format(i), lu[idx] )
        np.save( processed_path+'/fold{}/act_label'.format(i),  la[idx] )
        np.save( processed_path+'/fold{}/id'.format(i),         ID[idx] )
    
if __name__ == '__main__':

    parser  = argparse.ArgumentParser( description="preprocessing pipeline for signal" )

    parser.add_argument( '-d', '--dataset', type=str, choices=['unimib', 'sbhar', 'realdisp', 'ouisir'], default = "unimib", help='dataset to preprocessing' )
    parser.add_argument( '-p', '--path', type=str, default='../data/datasets/', help='path to dataset')
    parser.add_argument( '-o', '--out', type=str, default='../data/datasets/UNIMIBDataset/', help='path to store data preprocessed')

    args    = parser.parse_args()

    preprocessing(args.dataset, args.path, args.out)