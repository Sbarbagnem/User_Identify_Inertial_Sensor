import os
import shutil
import zipfile

import numpy as np
from scipy import stats
from scipy.io import loadmat
from sklearn import utils as skutils

from sliding_window import sliding_window

def unimib_process():

    root_path       = '/data/datasets/unimib_dataset'
    raw_data_path   = root_path + '/raw/'
    processed_path  = '/data/datasets/UNIMIBDataset/OuterPartition'

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

    for sid in range( 30 ):
        for aid in range( 9 ):

            act = activity_table[ aid ]
            trials = ( signal[sid, 0][0, 0][act].shape )[0]

            for tid in range( trials ):

                _data = signal[sid, 0][0, 0][act][tid,0][0:3, :]
                _data = np.transpose( _data )

                _data_windows   = sliding_window( _data, ( win_len, channel ), ( int( win_len/2 ), 1 ) )
                invalid_idx     = np.where( np.any( np.isnan( np.reshape( _data_windows, [-1, win_len*channel] ) ), axis=1 ) )[0]
                _data_windows   = np.delete( _data_windows, invalid_idx, axis=0 )
                _id             = np.arange( ID_generater, ID_generater+len(_data_windows) )

                data            = np.concatenate( (data, _data_windows), axis=0 )
                ID              = np.concatenate( (ID,   _id), axis=0 )
                ID_generater    = ID_generater + len( _data_windows ) + 10
            
            _la     = np.full( len(data)-len(la) , aid, dtype=np.int32 )
            la      = np.concatenate( (la, _la), axis=0 )

        _lu     = np.full( len(data)-len(lu) , sid, dtype=np.int32 )
        lu      = np.concatenate( (lu, _lu), axis=0 )

    # shuffle
    data, la, lu, ID    = skutils.shuffle( data, la, lu, ID )
    data, la, lu, ID    = skutils.shuffle( data, la, lu, ID )

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
    unimib_process()