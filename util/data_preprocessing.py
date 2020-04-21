import os
import shutil
import zipfile
import argparse

import numpy as np
from scipy import stats
from scipy.io import loadmat
from sklearn import utils as skutils

from sliding_window import sliding_window

def preprocessing(dataset, path, path_out, positions):
    
    if dataset == 'unimib':
        unimib_process(path, path_out)
    elif dataset == 'sbhar':
        sbhar_process(path, path_out)
    elif dataset == 'realdisp':
        realdisp_process(path, path_out, positions)

def realdisp_process(path, path_out, positions='all'):
    '''
        positions:  list of positions sensor to consider in preprocessing.
                    If "all" parameter is passed no filter to position will
                    be apply.
    '''

    '''
        structure of log file in dataset (120 columns):
            1       -> trial's number
            2       -> timestamp in microseconds
            3:119   -> data from sensors
                order positions -> RLA | RUA | BACK | LUA | LLA | RC | RT | LT | LC
                    for every postion   -> acc:[x,y,z] gyr:[x,y,z] mag:[x,y,z] quat:[1,2,3,4]
            120     -> id activity (0 if unknown)

    '''

    root_path       = path + 'REALDISP'
    raw_data_path   = root_path + '/'
    processed_path  = path_out + 'OuterPartition'

    win_len         = 100
    channel         = 3
    ID_generater    = 1
    number_sensor   = 2

    data    = np.empty( [0, win_len, channel*number_sensor], dtype=np.float )
    la      = np.empty( [0], dtype=np.int32 )
    lu      = np.empty( [0], dtype=np.int32 )
    ID      = np.empty( [0], dtype=np.int32 )

    if not os.path.exists( path_out ):
        os.mkdir( path_out ) 

    for fl in os.listdir(raw_data_path):
        if fl.startswith('subject'):
            print("Leggo log file ", fl)
            log_file    = np.loadtxt(fname=raw_data_path + fl)
            id_user     = int(fl.split('_')[0].split('subject')[1])
            print(id_user)
            
            # take only acc and gyro data
            acc_gyro    = log_file[:,2:119]
            activities  = log_file[:,-1].astype('int32')
            activities  = activities.reshape(acc_gyro.shape[0], 1)
            trials      = log_file[:,0].astype('int32')
            trials      = trials.reshape(acc_gyro.shape[0], 1)

            

            offset  = 6
            step    = 0

            # delete mag and quat columns
            for _ in range(9):
                acc_gyro = np.delete(acc_gyro, np.arange((offset*step) + offset ,(offset*step) + offset + 7), axis=1)
                step += 1

            step = 0
            
            # filter positions based on parameter positions
            if positions != 'all':
                if 'RLA' not in positions:
                    acc_gyro = acc_gyro[:,step*offset:]
                else:
                    step += 1
                if 'RUA' not in positions:
                    acc_gyro = acc_gyro[:,step*offset:]
                else:
                    step += 1
                if 'BACK' not in positions:
                    acc_gyro = acc_gyro[:,step*offset:]
                else:
                    step += 1
                if 'LUA' not in positions:
                    acc_gyro = acc_gyro[:,step*offset:]
                else:
                    step += 1
                if 'LLA' not in positions:
                    acc_gyro = acc_gyro[:,step*offset:]
                else:
                    step += 1
                if 'RC' not in positions:
                    acc_gyro = acc_gyro[:,step*offset:]
                else:
                    step += 1
                if 'RT' not in positions:
                    acc_gyro = acc_gyro[:,step*offset:]
                else:
                    step += 1
                if 'LT' not in positions:
                    acc_gyro = acc_gyro[:,step*offset:]
                else:
                    step += 1
                if 'LC' not in positions:
                    acc_gyro = acc_gyro[:,step*offset:]

            step = 0

            acc_gyro    = np.concatenate((trials, acc_gyro, activities), axis=1)

            # delete record with unknown activity label
            acc_gyro    = np.delete(acc_gyro, np.where(acc_gyro[:,-1]==0), axis=0)

            # cycle on activity
            for id_act in np.unique(acc_gyro[:,-1]):
                #print(int(id_act))
                step = 0

                temp = acc_gyro[np.where(acc_gyro[:,-1]==int(id_act))]

                # sliding window on every of 9 sensor
                for _ in range(9):       

                    acc     = temp[:,1+(offset*step):(step*offset)+4]
                    gyro    = temp[:,4+(offset*step):(step*offset)+7]

                    try:
                        _data_windows_acc   = sliding_window( acc, ( win_len, channel ), ( int( win_len/2 ), 1 ) )
                    except:
                        print("Not enough data for sliding window")
                    invalid_idx         = np.where( np.any( np.isnan( np.reshape( _data_windows_acc, [-1, win_len*channel] ) ), axis=1 ) )[0]
                    _data_windows_acc   = np.delete( _data_windows_acc, invalid_idx, axis=0 )

                    _data_windows_gyro  = sliding_window( gyro, ( win_len, channel ), ( int( win_len/2 ), 1 ) )
                    invalid_idx         = np.where( np.any( np.isnan( np.reshape( _data_windows_gyro, [-1, win_len*channel] ) ), axis=1 ) )[0]
                    _data_windows_gyro  = np.delete( _data_windows_gyro, invalid_idx, axis=0 )

                    try:
                        acc_gyro_concat    = np.concatenate((_data_windows_acc, _data_windows_gyro), axis=2)
                    except:
                        print("There is only one sliding window")
                        acc_gyro_concat    = np.concatenate((_data_windows_acc, _data_windows_gyro), axis=1)  
                        acc_gyro_concat    = acc_gyro_concat.reshape((1,acc_gyro_concat.shape[0],acc_gyro_concat.shape[1]))
                    
                    step += 1

                    _id             = np.arange( ID_generater, ID_generater+len(acc_gyro_concat)) # id for every window
                    data            = np.concatenate( (data, acc_gyro_concat), axis=0 ) # concat verticaly every window
                    ID              = np.concatenate( (ID,   _id), axis=0 )
                    ID_generater    = ID_generater + len( acc_gyro_concat ) + 10            

                # update la
                _la     = np.full( len(data)-len(la) , int(id_act) - 1, dtype=np.int32 ) # label activity for every window
                la      = np.concatenate( (la, _la), axis=0 )

            # update gloabl variabel lu
            _lu     = np.full( len(data)-len(lu) , int(id_user) - 1, dtype=np.int32 ) # label user for every window
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
            


def sbhar_process(path, path_out):
    print('Processing sbhar dataset')

    root_path       = path + 'SBHAR'
    raw_data_path   = root_path + '/RawData/'
    processed_path  = path_out + 'OuterPartition'

    win_len         = 100
    channel         = 3     
    ID_generater    = 1
    number_sensor   = 2

    data    = np.empty( [0, win_len, channel*number_sensor], dtype=np.float )
    la      = np.empty( [0], dtype=np.int32 )
    lu      = np.empty( [0], dtype=np.int32 )
    ID      = np.empty( [0], dtype=np.int32 )

    if not os.path.exists( path_out ):
        os.mkdir( path_out )    

    # read labels.txt with inf about data (id_exp, id_usr, id_activity, start_sample, stop_sample)
    info = np.loadtxt(fname=raw_data_path + 'labels.txt')

    # for subject
    for id_user in np.unique(info[:,1]):
        user_info = info[np.where(info[:,1]==int(id_user))]
        
        # for activity
        for id_act in np.unique(user_info[:,2]):
            activity_info = user_info[np.where(user_info[:,2]==int(id_act))]
            acc = np.empty( [0, channel], dtype=np.float )
            gyro = np.empty( [0, channel], dtype=np.float )

            for tid in np.unique(activity_info[:,0]):
                data_info = activity_info[np.where(activity_info[:,0]==int(tid))]
                
                # open associated file
                path_file_acc = "acc_exp"
                path_file_giro = "gyro_exp"
                if int(tid) < 10:
                    path_file_acc   += "0{}_user".format(int(tid))
                    path_file_giro  += "0{}_user".format(int(tid))
                else:
                    path_file_acc   += "{}_user".format(int(tid))
                    path_file_giro  += "{}_user".format(int(tid))
                if int(id_user) < 10:
                    path_file_acc   += "0{}.txt".format(int(id_user))
                    path_file_giro  += "0{}.txt".format(int(id_user))
                else:
                    path_file_acc   += "{}.txt".format(int(id_user))
                    path_file_giro  += "{}.txt".format(int(id_user))

                file_acc    = np.loadtxt(fname=raw_data_path + path_file_acc) 
                file_gyro   = np.loadtxt(fname=raw_data_path + path_file_giro)

                # concat data
                for start,stop in zip(data_info[:,-2], data_info[:,-1]):
                    #print(start, " ", stop) 
                    acc = np.concatenate((acc, file_acc[int(start):int(stop),:]), axis=0)
                    gyro = np.concatenate((gyro, file_gyro[int(start):int(stop),:]), axis=0)

            # sliding window
            try:
                _data_windows_acc   = sliding_window( acc, ( win_len, channel ), ( int( win_len/2 ), 1 ) )
            except:
                print("Not enough data for sliding window")
            invalid_idx         = np.where( np.any( np.isnan( np.reshape( _data_windows_acc, [-1, win_len*channel] ) ), axis=1 ) )[0]
            _data_windows_acc   = np.delete( _data_windows_acc, invalid_idx, axis=0 )

            _data_windows_gyro  = sliding_window( gyro, ( win_len, channel ), ( int( win_len/2 ), 1 ) )
            invalid_idx         = np.where( np.any( np.isnan( np.reshape( _data_windows_gyro, [-1, win_len*channel] ) ), axis=1 ) )[0]
            _data_windows_gyro  = np.delete( _data_windows_gyro, invalid_idx, axis=0 )

            try:
                acc_gyro    = np.concatenate((_data_windows_acc, _data_windows_gyro), axis=2)
            except:
                print("There is only one sliding window")
                acc_gyro    = np.concatenate((_data_windows_acc, _data_windows_gyro), axis=1)  
                acc_gyro    = acc_gyro.reshape((1,acc_gyro.shape[0],acc_gyro.shape[1]))

            _id             = np.arange( ID_generater, ID_generater+len(acc_gyro)) # id for every window
            data            = np.concatenate( (data, acc_gyro), axis=0 ) # concat verticaly every window
            ID              = np.concatenate( (ID,   _id), axis=0 )
            ID_generater    = ID_generater + len( acc_gyro ) + 10

            _la     = np.full( len(data)-len(la) , int(id_act) - 1, dtype=np.int32 ) # label activity for every window
            la      = np.concatenate( (la, _la), axis=0 )

        _lu     = np.full( len(data)-len(lu) , int(id_user) - 1, dtype=np.int32 ) # label user for every window
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

def unimib_process(path, path_out):
    print('Processing unimib dataset')

    root_path       = path + 'unimib_dataset'
    raw_data_path   = root_path + '/raw/'
    processed_path  = path_out + 'OuterPartition'

    win_len         = 100   # 50 Hz * 2 seconds
    channel         = 3     # only acc
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

    #preprocessing("unimib", "../data/datasets/", "../data/datasets/UNIMIBDataset/")
    #preprocessing("sbhar", "../data/datasets/", "../data/datasets/SBHAR_processed/")
    preprocessing('realdisp', "../data/datasets/", "../data/datasets/REALDISP_processed/", "all")