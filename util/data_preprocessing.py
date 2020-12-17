import os
import shutil
import zipfile
import argparse
import sys
from tqdm import tqdm
import pandas as pd
from pprint import pprint
import numpy as np
from scipy import stats
from scipy.io import loadmat
from sklearn import utils as skutils
import matplotlib.pyplot as plt
import pywt
import random
from itertools import islice

from sliding_window import sliding_window
from utils import str2bool, split_balanced_data, denoiseData, detectGaitCycle, segment2GaitCycle, remove_g_component, interpolated


def ou_isir_process_cycle_based(
        path_data,
        path_out,
        plot_denoise=False,
        plot_peak=False,
        plot_interpolated=False,
        plot_auto_corr_coeff=False,
        denoise=False,
        plot_split=False,
        gcLen=None,
        gyroscope=False):


    cycles_interpolated = []
    label_user = []
    lu = -1
    to_interp = 80 #120

    # read files
    print('Read csv file')
    for f in tqdm(os.listdir(path_data)):

        if 'seq0' in f:
            lu += 1

        ### READ ACCELETOMETER DATA ###
        acc = pd.read_csv(path_data + '/' + f, header=None,
                         skiprows=[0,1], skipfooter=1, usecols=[3, 4, 5], engine='python') # only acc
        acc = acc.values
        acc = remove_g_component(acc, sampling_rate=100, plot=False)
        if denoise == True:
            acc = denoiseData(acc, plot_denoise)
        peaks = detectGaitCycle(acc, plot_peak, plot_auto_corr_coeff, gcLen)
        cycles_acc = segment2GaitCycle(peaks, acc, plot_split=False)

        ### READ GYROSCOPE DATA ###
        gyro = pd.read_csv(path_data + '/' + f, header=None,
                        skiprows=[0,1], skipfooter=1, usecols=[0,1,2], engine='python')
        gyro = gyro.values
        if denoise == True:
            gyro = denoiseData(gyro, plot_denoise)
        cycles_gyro = segment2GaitCycle(peaks, gyro, plot_split=False)

        cycles = [np.hstack((acc,gyro)) for acc,gyro in zip(cycles_acc, cycles_gyro)]

        cycles = interpolated(cycles, to_interp, plot_interpolated)

        cycles_interpolated.append(cycles)
        label_user.extend([lu]*len(cycles))

    cycles_interpolated = np.concatenate(cycles_interpolated, axis=0)
    
    # compute magnitude
    magnitude_acc = np.concatenate((cycles_interpolated[:,:,[0,1,2]], np.sqrt(
        np.sum(np.power(cycles_interpolated[:,:,[0,1,2]], 2), 2, keepdims=True))), 2)
    magnitude_gyro = np.concatenate((cycles_interpolated[:,:,[3,4,5]], np.sqrt(
        np.sum(np.power(cycles_interpolated[:,:,[3,4,5]], 2), 2, keepdims=True))), 2)
    cycles_interpolated = np.concatenate((magnitude_acc,magnitude_gyro), axis=2)
    
    # plot 4 cycles extracted from 10 random user to see similarity
    '''
    random_sub = random.sample(range(0,100), 10)
    for sub in random_sub:
            plt.figure(figsize=(12, 3))
            plt.style.use('seaborn-darkgrid')
            idx = np.where(np.asarray(label_user) == sub)
            temp_cycles = cycles_interpolated[idx]
            for i,axis in zip(range(3), ['x','y','z']):
                plt.subplot(3, 1, i + 1)
                plt.gca().set_title(f'Axis {axis}')
                for cycle,color in zip(temp_cycles[:4,:,i], ['b-.', 'r-o', 'g-*', 'm-v']):
                    plt.plot(
                        np.arange(to_interp), cycle[:], color, markersize=5)
            plt.tight_layout()
            plt.show()       
    '''
    print(f'Found {cycles_interpolated.shape[0]} gait cycles')
    print(
        f'Min number of cycle for user is: {min(np.unique(label_user, return_counts=True)[1])}')
    print(
        f'Max number of cycle for user is: {max(np.unique(label_user, return_counts=True)[1])}')
    print(
        f'Mean number of cycle for user is: {int(np.mean(np.unique(label_user, return_counts=True)[1]))}')
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    np.save(path_out + '/data', cycles_interpolated)
    np.save(path_out + '/user_label', label_user)


def ou_isir_process_window_based(
        path_data,
        path_out,
        overlap,
        window_len):
    stride = int((1 - overlap)*window_len)

    # define variables
    data = []
    lu = []
    sessions = []
    lu_temp = 0

    # read files
    print('Read csv file')
    for f in tqdm(os.listdir(path_data)):

        if 'seq0' in f:
            sess_temp = 0
        else:
            sess_temp = 1

        ### READ ACCELETOMETER DATA ###
        acc = pd.read_csv(path_data + '/' + f, header=None,
                         skiprows=[0,1], skipfooter=1, usecols=[3, 4, 5], engine='python')
        acc = remove_g_component(acc.values, sampling_rate=100, plot=False)

        ### READ GYROSCOPE DATA ###
        gyro = pd.read_csv(path_data + '/' + f, header=None,
                         skiprows=[0,1], skipfooter=1, usecols=[0,1,2], engine='python')
        gyro = gyro.values
        acc_gyro = np.hstack((acc,gyro))
        data.append(acc_gyro)
        lu.append(lu_temp)
        sessions.append(sess_temp)

        if 'seq1' in f:
            lu_temp += 1

    # define list to save
    data_windows = []
    label_user = []
    ID = []
    sessions_window = []
    id_temp=  0

    print('Sliding window')
    for signal, user, session in zip(data, lu, sessions):
        windows = sliding_window(signal, (window_len, signal.shape[1]), (stride, 1))
        if windows.ndim == 2:
            windows = np.reshape(windows, (1, windows.shape[0], windows.shape[1]))
        data_windows.append(windows)
        label_user.extend([user]*len(windows))
        # to del overlap sequence between train and test
        ID.extend(np.arange(id_temp, id_temp + len(windows)))
        sessions_window.extend([session]*len(windows))
        id_temp = id_temp + len(windows) + 10

    data_windows = np.concatenate(data_windows, axis=0)
    
    print('Compute magnitude')
    magnitude_acc = np.concatenate((data_windows[:,:,[0,1,2]], np.sqrt(
        np.sum(np.power(data_windows[:,:,[0,1,2]], 2), 2, keepdims=True))), 2)
    magnitude_gyro = np.concatenate((data_windows[:,:,[3,4,5]], np.sqrt(
        np.sum(np.power(data_windows[:,:,[3,4,5]], 2), 2, keepdims=True))), 2)
    data_windows = np.concatenate((magnitude_acc,magnitude_gyro), axis=2)

    print(f'Found {data_windows.shape[0]} total samples')

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    np.save(path_out + '/data', data_windows)
    np.save(path_out + '/user_label', label_user)
    np.save(path_out + '/id.npy', ID)
    np.save(path_out + '/sessions.npy', sessions_window)


def realdisp_process(
        path_data,
        path_out,
        sensors_type=['acc', 'gyro', 'magn'],
        positions=['all'],
        sensors_displacement=['ideal', 'self'],
        magnitude=True,
        size_overlapping=0.5,
        win_len=100,
        authentication=False):
    """
    Create 10 folds for dataset REALDISP.

    Parameters
    ----------
    path_data : str
        Path to raw data.
    path_out : str
        Path to store folds processed.
    sensors_type : list[str]
        A list to choose sensors from acceleromenter, gyroscope and magnetometer.
    positons : list[str]
        A list of positions to maintain from total 9 positions.
    sensors_displacement : list[str]
        A list of sensor displacement to mantain.
    magnitude : bool
        User or not magnitude
    size_overlapping : float between 0 and 1
        Percentage of overlapping used in sliding windows
    win_len : int
        Length of window used in sliding window.
    """

    '''
    Data in log file subject{id}_{sensors_displacement}.log
    Structure of log file in dataset (120 columns):
        0       -> trial's number (second)
        1       -> timestamp in microseconds
        2:118   -> data from sensors
                    order positions     -> RLA | RUA | BACK | LUA | LLA | RC | RT | LT | LC
                    for every postion   -> acc:[x,y,z] gyr:[x,y,z] mag:[x,y,z] quat:[1,2,3,4]
        119     -> id activity (0 if unknown)
    '''

    places = ['RLA', 'RUA', 'BACK', 'LUA', 'LLA', 'RC', 'RT', 'LT', 'LC']
    sensors = ['acc', 'gyro', 'magn', 'quat']
    axis = [['x', 'y', 'z'], ['x', 'y', 'z'],
            ['x', 'y', 'z'], ['1', '2', '3', '4']]

    HEADER = ['sec', 'microsec']
    HEADER.extend(['_'.join([place, sensor, a])
                   for place in places for sensor, ax in zip(sensors, axis) for a in ax])
    HEADER.extend(['act'])

    print('Processing realdisp dataset')

    raw_data_path = path_data
    if magnitude:
        processed_path = path_out + \
            'OuterPartition_magnitude_{}_{}'.format(
                '_'.join(sensors_displacement), str(size_overlapping * 10))
    else:
        processed_path = path_out + \
            'OuterPartition_{}_{}'.format(
                '_'.join(sensors_displacement), str(size_overlapping * 10))

    if win_len != 100:
        processed_path = processed_path + f'_wl_{win_len}'
    if authentication:
        processed_path = f"../data/authentication/REALDISP_{positions[0]}"

    channel = 3
    ID_generater = 1

    if magnitude:
        channel = 4

    number_sensor = len(sensors_type)

    data = []  # list of data window
    la = []  # list of label activity for every user
    lu = []  # list of label user for every window
    di = []  # list of displacement type, to split balanced also based on this
    id_pos = []  # list of position for every window (LT,RT,....)
    ID = []

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    zero_act = 0
    total_data = 0

    for fl in tqdm(os.listdir(raw_data_path)):
        # filter based on displacement choosen
        if fl.startswith('subject') and any(displace in fl for displace in sensors_displacement):
            log_file = pd.DataFrame(pd.read_table(
                raw_data_path + fl).values, columns=HEADER)
            log_file.drop(['sec', 'microsec'], inplace=True, axis=1)
            log_file = log_file.astype({"act": int})

            id_user = int(fl.split('_')[0].split('subject')[1])

            if 'ideal' in fl.split('_')[1]:
                disp = 0
            if 'self' in fl.split('_')[1]:
                disp = 1
            if 'mutual' in fl.split('_')[1]:
                disp = 2

            zero_act += log_file[log_file['act'] == 0].shape[0]
            total_data += log_file.shape[0]

            # delete sample with act == 0 (unkwnown)
            log_file = log_file[log_file.act != 0]
            activities = np.unique(log_file.act)

            # filter based on position sensor choosen (BACK, RLA, RUA, ...) and sensor tyep (acc, gyro, ...)
            mask = [(col.split('_')[0] in positions and col.split('_')[
                     1] in sensors_type) for col in log_file.columns.tolist()[:-1]]
            mask.append(True)
            col_filter = log_file.columns[mask]
            log_file = log_file[col_filter]

            # cycle on activity
            for id_act in activities:

                temp = log_file[log_file.act == id_act]

                # sliding window on different position choose
                for pos in positions:

                    merge_sensor = []

                    p = positions.index(pos)

                    if 'acc' in sensors_type:
                        acc = temp[[col for col in temp.columns if (
                            'acc' in col and pos in col)]].to_numpy()
                        if magnitude:
                            acc = np.concatenate((acc, np.apply_along_axis(lambda x: np.sqrt(
                                np.sum(np.power(x, 2))), axis=1, arr=acc).reshape(-1, 1)), axis=1)
                        merge_sensor.append(acc)
                    if 'gyro' in sensors_type:
                        gyro = temp[[col for col in temp.columns if (
                            'gyro' in col and pos in col)]].to_numpy()
                        if magnitude:
                            gyro = np.concatenate((gyro, np.apply_along_axis(lambda x: np.sqrt(
                                np.sum(np.power(x, 2))), axis=1, arr=gyro).reshape(-1, 1)), axis=1)
                        merge_sensor.append(gyro)
                    if 'magn' in sensors_type:
                        magn = temp[[col for col in temp.columns if (
                            'magn' in col and pos in col)]].to_numpy()
                        if magnitude:
                            magn = np.concatenate((magn, np.apply_along_axis(lambda x: np.sqrt(
                                np.sum(np.power(x, 2))), axis=1, arr=magn).reshape(-1, 1)), axis=1)
                        merge_sensor.append(magn)

                    # concat sensor and sliding only one time
                    merge_sensor = np.hstack(merge_sensor)
                    _data_windows = sliding_window(
                        merge_sensor, (win_len, merge_sensor.shape[1]), (int(win_len * (1 - size_overlapping)), 1))

                    # id for every window
                    _id = np.arange(
                        ID_generater, ID_generater + len(_data_windows))
                    ID.extend(_id)
                    ID_generater = ID_generater + len(_data_windows) + 10

                    # concat verticaly every window
                    data.append(_data_windows)

                    # id position
                    id_pos.extend([p]*(len(_data_windows)))

                # update la
                _la = [int(id_act) - 1] * (len(ID) - len(la))
                la.extend(_la)

            # update gloabl variabel lu and di
            _lu = [int(id_user) - 1] * (len(ID) - len(lu))
            _di = [disp] * (len(ID) - len(di))
            lu.extend(_lu)
            di.extend(_di)

    print('punti con attività sconosciuta ', zero_act, ' su ', total_data)

    # define array dimension for data and labels
    data_array = np.zeros(
        [len(lu), win_len, channel * number_sensor], dtype=np.float)
    la_array = np.zeros([len(la)], dtype=np.int32)
    lu_array = np.zeros([len(lu)], dtype=np.int32)
    di_array = np.zeros([len(di)], dtype=np.int32)
    ID_array = np.zeros([len(ID)], dtype=np.int32)
    id_pos_array = np.zeros([len(id_pos)], dtype=np.int32)

    idx = 0
    for i, el in enumerate(data):
        n = el.shape[0]
        ID_array[idx:idx + n] = ID[idx:idx + n]
        lu_array[idx:idx + n] = lu[idx:idx + n]
        la_array[idx:idx + n] = la[idx:idx + n]
        di_array[idx:idx + n] = di[idx:idx + n]
        id_pos_array[idx:idx+n] = id_pos[idx:idx+n]
        data_array[idx:idx + n, :, :] = el
        idx += n

    # shuffle
    if not authentication:
        data, la, lu, di, id_pos, ID = skutils.shuffle(
            data, la, lu, di, id_pos, ID)

        print(f'shape data {data.shape}')

        indexes = split_balanced_data(lu, la, folders=10, di=di)

        #plt_user_distribution(indexes, lu)

        #plt_act_distribution(indexes, la)

        # partition
        for i in range(10):

            # clear dir
            if os.path.exists(processed_path + '/fold{}'.format(i)):
                shutil.rmtree(processed_path + '/fold{}'.format(i))
            os.mkdir(processed_path + '/fold{}'.format(i))

            #idx = np.arange(int(len(data)*0.1*i), int(len(data)*0.1*(i+1)), 1)
            idx = indexes[str(i)]
            np.save(processed_path + '/fold{}/data'.format(i), data[idx])
            np.save(processed_path + '/fold{}/user_label'.format(i), lu[idx])
            np.save(processed_path + '/fold{}/act_label'.format(i), la[idx])
            np.save(processed_path + '/fold{}/id'.format(i), ID[idx])
            np.save(processed_path + '/fold{}/di'.format(i), di[idx])
            np.save(processed_path + '/fold{}/pos'.format(i), id_pos[idx])
    else:
        np.save(processed_path + '/data', data_array)
        np.save(processed_path + '/user_label', lu_array)
        np.save(processed_path + '/act_label', la_array)
        np.save(processed_path + '/id', ID_array)    

def sbhar_process(
        path_data,
        path_out,
        magnitude,
        size_overlapping,
        win_len,
        six_adl=False,
        authentication=False):
    print('Processing sbhar dataset')

    raw_data_path = path_data + 'RawData/'

    if not authentication:
        if magnitude:
            if six_adl:
                processed_path = path_out + \
                    'OuterPartition_magnitude_sbhar_six_adl_{}'.format(
                        str(size_overlapping * 10))
            else:
                processed_path = path_out + \
                    'OuterPartition_magnitude_{}'.format(
                        str(size_overlapping * 10))
        else:
            processed_path = path_out + \
                'OuterPartition_{}'.format(str(size_overlapping * 10))
        if win_len != 100:
            processed_path = processed_path + f'_wl_{win_len}'
    else:
        if six_adl:
            processed_path = "../data/authentication/SBHAR"
        else:
            sys.exit('Only six adl for sbhar dataset')

    #win_len = 100
    if magnitude:
        channel = 4
    else:
        channel = 3
    ID_generater = 1
    number_sensor = 2

    data = []
    la = []
    lu = []
    ID = []
    sessions = []  # sessions for authentication experiment

    '''
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    '''

    # read labels.txt with inf about data (id_exp, id_usr, id_activity,
    # start_sample, stop_sample)
    info = np.loadtxt(fname=raw_data_path + 'labels.txt')

    # for subject
    for id_user in tqdm(np.unique(info[:, 1])):
        user_info = info[np.where(info[:, 1] == int(id_user))]

        # for activity
        for id_act in np.unique(user_info[:, 2]):
            activity_info = user_info[np.where(user_info[:, 2] == int(id_act))]

            # for sessions
            for i, tid in enumerate(np.unique(activity_info[:, 0])):
                acc = np.empty([0, 3], dtype=np.float)
                gyro = np.empty([0, 3], dtype=np.float)
                data_info = activity_info[np.where(
                    activity_info[:, 0] == int(tid))]

                # open associated file
                path_file_acc = "acc_exp"
                path_file_giro = "gyro_exp"
                if int(tid) < 10:
                    path_file_acc += "0{}_user".format(int(tid))
                    path_file_giro += "0{}_user".format(int(tid))
                else:
                    path_file_acc += "{}_user".format(int(tid))
                    path_file_giro += "{}_user".format(int(tid))
                if int(id_user) < 10:
                    path_file_acc += "0{}.txt".format(int(id_user))
                    path_file_giro += "0{}.txt".format(int(id_user))
                else:
                    path_file_acc += "{}.txt".format(int(id_user))
                    path_file_giro += "{}.txt".format(int(id_user))

                file_acc = np.loadtxt(fname=raw_data_path + path_file_acc)
                file_gyro = np.loadtxt(fname=raw_data_path + path_file_giro)

                # concat data
                for start, stop in zip(data_info[:, -2], data_info[:, -1]):
                    acc = np.concatenate(
                        (acc, file_acc[int(start):int(stop), :]), axis=0)
                    gyro = np.concatenate(
                        (gyro, file_gyro[int(start):int(stop), :]), axis=0)

                if magnitude:
                    acc = np.concatenate((acc, np.apply_along_axis(lambda x: np.sqrt(
                        np.sum(np.power(x, 2))), axis=1, arr=acc).reshape(-1, 1)), axis=1)
                    gyro = np.concatenate((gyro, np.apply_along_axis(lambda x: np.sqrt(
                        np.sum(np.power(x, 2))), axis=1, arr=gyro).reshape(-1, 1)), axis=1)

                # sliding window
                try:
                    _data_windows_acc = sliding_window(
                        acc, (win_len, channel), (int(win_len * (1 - size_overlapping)), 1))
                    _data_windows_gyro = sliding_window(
                        gyro, (win_len, channel), (int(win_len * (1 - size_overlapping)), 1))
                except BaseException:
                    print("Not enough data for sliding window")
                    print(id_user, id_act)
                try:
                    acc_gyro = np.concatenate(
                        (_data_windows_acc, _data_windows_gyro), axis=2)
                except BaseException:
                    print("There is only one sliding window")
                    acc_gyro = np.concatenate(
                        (_data_windows_acc, _data_windows_gyro), axis=1)
                    acc_gyro = acc_gyro.reshape(
                        (1, acc_gyro.shape[0], acc_gyro.shape[1]))

                _id = np.arange(ID_generater, ID_generater +
                                len(acc_gyro))  # id for every window
                # concat verticaly every window
                data.append(acc_gyro)
                ID.extend(_id)
                ID_generater = ID_generater + len(acc_gyro) + 10

                # label activity for every window
                _la = [int(id_act) - 1] * (len(ID) - len(la))
                la.extend(_la)

                # trials for authentication test
                if authentication:
                    session = [i] * len(_id)
                    sessions.extend(session)

        # update gloabl variabel lu
        _lu = [int(id_user) - 1] * (len(ID) - len(lu))
        lu.extend(_lu)

    # define array dimension for data and labels
    data_array = np.zeros(
        [len(lu), win_len, channel * number_sensor], dtype=np.float)
    la_array = np.zeros([len(lu)], dtype=np.int32)
    lu_array = np.zeros([len(lu)], dtype=np.int32)
    ID_array = np.zeros([len(lu)], dtype=np.int32)
    if authentication:
        sessions_array = np.zeros([len(lu)], dtype=np.int32)

    idx = 0
    for i, el in enumerate(data):
        n = el.shape[0]
        ID_array[idx:idx + n] = ID[idx:idx + n]
        lu_array[idx:idx + n] = lu[idx:idx + n]
        la_array[idx:idx + n] = la[idx:idx + n]
        if authentication:
            sessions_array[idx:idx + n] = sessions[idx:idx + n]
        data_array[idx:idx + n, :, :] = el
        idx += n

    # delete data of transition action for six_adl dataset
    if six_adl:
        print('make six adl dataset')
        idx_to_del = np.where(la_array > 5)
        data_array = np.delete(data_array, idx_to_del, axis=0)
        la_array = np.delete(la_array, idx_to_del, axis=0)
        lu_array = np.delete(lu_array, idx_to_del, axis=0)
        ID_array = np.delete(ID_array, idx_to_del, axis=0)
        if authentication:
            sessions_array = np.delete(sessions_array, idx_to_del, axis=0)

    if not os.path.exists(processed_path + '/'):
        os.mkdir(processed_path + '/')

    if not authentication:
        data_array, lu_array, la_array, ID_array, sessions_array = skutils.shuffle(
            data_array, lu_array, la_array, ID_array, sessions_array)
        indexes = split_balanced_data(lu_array, la_array, folders=10)

        #plt_user_distribution(indexes, lu_array)

        #plt_act_distribution(indexes, la_array)

        # partition
        for i in range(10):

            # clear dir
            if os.path.exists(processed_path + '/fold{}'.format(i)):
                shutil.rmtree(processed_path + '/fold{}'.format(i))
            os.mkdir(processed_path + '/fold{}'.format(i))

            idx = indexes[str(i)]
            np.save(processed_path + '/fold{}/data'.format(i), data_array[idx])
            np.save(processed_path +
                    '/fold{}/user_label'.format(i), lu_array[idx])
            np.save(processed_path +
                    '/fold{}/act_label'.format(i), la_array[idx])
            np.save(processed_path + '/fold{}/id'.format(i), ID_array[idx])
    else:
        np.save(processed_path + '/data', data_array)
        np.save(processed_path + '/user_label', lu_array)
        np.save(processed_path + '/act_label', la_array)
        np.save(processed_path + '/id', ID_array)
        np.save(processed_path + '/sessions', sessions_array)


def unimib_process(path_data, path_out, magnitude, size_overlapping, win_len, authentication=False):
    print('Processing unimib dataset')

    raw_data_path = path_data + 'data/'
    if magnitude:
        processed_path = path_out + \
            'OuterPartition_magnitude_{}'.format(
                str(size_overlapping * 10))
    else:
        processed_path = path_out + \
            'OuterPartition_{}'.format(str(size_overlapping * 10))
    if win_len != 100:
        processed_path = processed_path + f'_wl_{win_len}'

    if authentication:
        processed_path = "../data/authentication/SHAR"

    # win_len = 100 # 50 Hz * 2 seconds
    if magnitude:
        channel = 4
    else:
        channel = 3
    ID_generater = 1

    data = np.empty([0, win_len, channel], dtype=np.float)
    la = np.empty([0], dtype=np.int32)
    lu = np.empty([0], dtype=np.int32)
    ID = np.empty([0], dtype=np.int32)
    gender_map = {'M': 0, 'F': 1}
    # for split equal gender authentication test
    gender = np.empty([0], dtype=np.int32)
    sessions = np.empty([0], dtype=np.int32)

    signal = loadmat(raw_data_path + 'full_data.mat')['full_data']

    activity_table = [
        'StandingUpFS',
        'StandingUpFL',
        'Walking',
        'Running',
        'GoingUpS',
        'Jumping',
        'GoingDownS',
        'LyingDownFS',
        'SittingDown']

    # id subject
    for sid in range(30):
        gender_user = gender_map[signal[sid, 1][0].strip()]
        # id activity
        for aid in range(9):
            act = activity_table[aid]
            trials = (signal[sid, 0][0, 0][act].shape)[0]
            # id trial
            for tid in range(trials):
                if magnitude:
                    _data = signal[sid, 0][0, 0][act][tid, 0][(0, 1, 2, 5), :]
                else:
                    _data = signal[sid, 0][0, 0][act][tid, 0][0:3, :]
                _data = np.transpose(_data)
                _data_windows = sliding_window(
                    _data, (win_len, channel), (int(win_len * (1 - size_overlapping)), 1))
                invalid_idx = np.where(np.any(np.isnan(np.reshape(
                    _data_windows, [-1, win_len * channel])), axis=1))[0]  # delete window with NaN sample
                _data_windows = np.delete(_data_windows, invalid_idx, axis=0)
                _id = np.arange(ID_generater, ID_generater +
                                len(_data_windows))  # id for every window

                # concat verticaly every window
                data = np.concatenate((data, _data_windows), axis=0)
                ID = np.concatenate((ID, _id), axis=0)
                ID_generater = ID_generater + len(_data_windows) + 10

                # trials for authentication test
                session = np.repeat(tid, len(_data_windows))
                sessions = np.concatenate((sessions, session), axis=0)

            # label activity for every window
            _la = np.full(len(data) - len(la), aid, dtype=np.int32)
            la = np.concatenate((la, _la), axis=0)

        # label user for every window
        _lu = np.full(len(data) - len(lu), sid, dtype=np.int32)
        _gender = np.full(len(data) - len(gender), gender_user, dtype=np.int32)
        lu = np.concatenate((lu, _lu), axis=0)
        gender = np.concatenate((gender, _gender), axis=0)

    if not os.path.exists(processed_path + '/'):
        os.makedirs(processed_path + '/')

    # split balanced data, return array (10, indexes_folder)
    if not authentication:
        data, lu, la, ID = skutils.shuffle(
            data, lu, la, ID)
        indexes = split_balanced_data(lu, la, folders=10)

        #plt_user_distribution(indexes, lu)
        #plt_act_distribution(indexes, la)

        # create dir partition
        for i in range(10):

            # clear dir
            if os.path.exists(processed_path + '/fold{}'.format(i)):
                shutil.rmtree(processed_path + '/fold{}'.format(i))
            os.mkdir(processed_path + '/fold{}'.format(i))

            #idx = np.arange(int(len(data)*0.1*i), int(len(data)*0.1*(i+1)), 1)
            idx = indexes[str(i)]
            np.save(processed_path + '/fold{}/data'.format(i), data[idx])
            np.save(processed_path + '/fold{}/user_label'.format(i), lu[idx])
            np.save(processed_path + '/fold{}/act_label'.format(i), la[idx])
            np.save(processed_path + '/fold{}/id'.format(i), ID[idx])
    else:
        np.save(processed_path + '/data', data)
        np.save(processed_path + '/user_label', lu)
        np.save(processed_path + '/act_label', la)
        np.save(processed_path + '/id', ID)
        np.save(processed_path + '/gender', gender)
        np.save(processed_path + '/sessions', sessions)


def plt_user_distribution(dict_indexes, lu):

    plt.figure(figsize=(12, 3))
    plt.style.use('seaborn-darkgrid')

    for folder in np.arange(len(dict_indexes)):
        plt.subplot(2, 5, folder + 1)
        plt.title('folder {}'.format(folder + 1))
        folder_index = dict_indexes[str(folder)]
        user_distributions = []
        for user in np.unique(lu):
            number_user = len([i for index, i in enumerate(
                lu) if i == user and index in folder_index])
            user_distributions.append(number_user)

        plt.bar(x=np.arange(len(user_distributions)),
                height=user_distributions)
    plt.tight_layout()
    plt.show()


def plt_act_distribution(dict_indexes, la):

    plt.figure(figsize=(12, 3))
    plt.style.use('seaborn-darkgrid')

    for folder in np.arange(len(dict_indexes)):
        plt.subplot(2, 5, folder + 1)
        plt.title('folder {}'.format(folder + 1))
        folder_index = dict_indexes[str(folder)]
        act_distributions = []
        for act in np.unique(la):
            number_act = len([i for index, i in enumerate(
                la) if i == act and index in folder_index])
            act_distributions.append(number_act)

        plt.bar(x=np.arange(len(act_distributions)), height=act_distributions)
    plt.tight_layout()
    plt.show()


def plot_signal(data):
    plt.figure(figsize=(12, 3))
    plt.style.use('seaborn-darkgrid')
    plt.plot(np.arange(data.shape[0]), data, 'b-')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="preprocessing pipeline for signal")

    # for all dataset
    parser.add_argument(
        '-dataset',
        '--dataset',
        type=str,
        choices=[
            'unimib',
            'sbhar',
            'sbhar_six_adl',
            'realdisp',
            'ouisir'],
        help='dataset to preprocessing',
        required=True)
    parser.add_argument('-p', '--path', type=str,
                        default='../data/datasets/', help='path to dataset')
    parser.add_argument('-win_len', '--win_len', type=int,
                        default=100, help='windows slice len')
    parser.add_argument(
        '-overlap',
        '--overlap',
        type=float,
        nargs='+',
        default=[0.5],
        help='overlap in sliding window')
    parser.add_argument(
        '-magnitude',
        '--magnitude',
        type=str2bool,
        default=[True],
        nargs='+',
        help='bool use or not magnitude')
    parser.add_argument(
        '-authentication',
        '--authentication',
        type=str2bool,
        help='read data for authentication, it doesn\'t split data in 10 fold',
        default=False
    )

    # for realdisp dataset
    parser.add_argument(
        '-sensor_place',
        '--sensor_place',
        type=str,
        nargs='+',
        default=[],
        choices=['LC', 'RC', 'LT', 'RT', 'LLA', 'RLA', 'LUA', 'RUA', 'BACK'],
        help='choose sensor\'s place to used',
        required=False)
    parser.add_argument(
        '-sensor_displacement',
        '--sensor_displacement',
        type=str,
        nargs='+',
        default=['ideal'],
        choices=['ideal', 'self', 'mutual'],
        help='for only realdisp dataset, choose types of sensor displacemente using'
    )

    # for ouisir dataset
    parser.add_argument(
        '-method',
        '--method',
        type=str,
        choices=['cycle_based', 'window_based'],
        help='type of method for signal segmentation'
    )
    parser.add_argument(
        '-plot_denoise',
        '--plot_denoise',
        type=str2bool,
        default=False,
        help='Plot original and denoised signal'
    )
    parser.add_argument(
        '-plot_peak',
        '--plot_peak',
        type=str2bool,
        default=False,
        help='Plot detected peak of gait cycles in signal'
    )
    parser.add_argument(
        '-plot_interpolated',
        '--plot_interpolated',
        type=str2bool,
        default=False,
        help='Plot original and interpolated signal to fixed length'
    )
    parser.add_argument(
        '-plot_auto_corr_coeff',
        '--plot_auto_corr_coeff',
        type=str2bool,
        default=False,
        help='Plot autocorrelation and 2nd peak take for estimated cycle length'
    )
    parser.add_argument(
        '-denoise',
        '--denoise',
        type=str2bool,
        default=False
    )
    parser.add_argument(
        '-window_len',
        '--window_len',
        type=int
    )
    parser.add_argument(
        '-gcLen',
        '--gcLen',
        type=int
    )

    args = parser.parse_args()

    if args.dataset == 'realdisp' and args.sensor_place == []:
        parser.error("Realdisp required sensor placed to use")

    for magnitude in [*args.magnitude]:
        for overlap in [*args.overlap]:
            if args.dataset == 'unimib':
                unimib_process(
                    path_data="../data/datasets/unimib_dataset/",
                    path_out="../data/datasets/UNIMIB_processed/",
                    magnitude=magnitude,
                    size_overlapping=overlap,
                    win_len=args.win_len,
                    authentication=args.authentication)
            elif args.dataset == 'sbhar':
                sbhar_process(
                    path_data="../data/datasets/SBHAR/",
                    path_out="../data/datasets/SBHAR_processed/",
                    magnitude=magnitude,
                    size_overlapping=overlap,
                    win_len=args.win_len,
                    six_adl=False,
                    authentication=args.authentication)
            elif args.dataset == 'sbhar_six_adl':
                sbhar_process(
                    path_data="../data/datasets/SBHAR/",
                    path_out="../data/datasets/SBHAR_processed/",
                    magnitude=magnitude,
                    size_overlapping=overlap,
                    win_len=args.win_len,
                    six_adl=True,
                    authentication=args.authentication)
            elif args.dataset == 'realdisp':
                realdisp_process(
                    path_data="../data/datasets/REALDISP/",
                    path_out=f"../data/datasets/REALDISP_processed_{'_'.join(args.sensor_place)}/",
                    positions=args.sensor_place,
                    sensors_displacement=args.sensor_displacement,
                    magnitude=magnitude,
                    size_overlapping=overlap,
                    win_len=args.win_len,
                    authentication=args.authentication)
            elif args.dataset == 'ouisir':
                if args.method == 'cycle_based':
                    denoise = "denoise" if args.denoise else "no_denoise"
                    autocorr = 'autocorr' if args.gcLen == None else "no_autocorr"
                    path_out = '../data/datasets/OUISIR_processed/cycle_based/{}/{}'.format(
                        denoise, autocorr)
                    ou_isir_process_cycle_based(
                        path_data='../data/datasets/OU-ISIR-gait/AutomaticExtractionData_IMUZCenter',
                        path_out=path_out,
                        plot_denoise=args.plot_denoise,
                        plot_peak=args.plot_peak,
                        plot_interpolated=args.plot_interpolated,
                        plot_auto_corr_coeff=args.plot_auto_corr_coeff,
                        denoise=args.denoise,
                        gcLen=args.gcLen
                    )
                elif args.method == 'window_based':
                    if args.authentication:
                        path_out=f'../data/authentication/OUISIR_window_based/{args.window_len}/{int(args.overlap[0]*100)}/'
                    else:
                        path_out=f'../data/datasets/OUISIR_processed/window_based/{args.window_len}/{int(args.overlap[0]*100)}/'
                    ou_isir_process_window_based(
                        path_data='../data/datasets/OU-ISIR-gait/AutomaticExtractionData_IMUZCenter',
                        path_out=f'../data/datasets/OUISIR_processed/window_based/{args.window_len}/{int(args.overlap[0]*100)}/',
                        window_len=args.window_len,
                        overlap=args.overlap[0]
                    )
