import os
import shutil
import zipfile
import argparse
import sys

import numpy as np
from scipy import stats
from scipy.io import loadmat
from sklearn import utils as skutils
import matplotlib.pyplot as plt

from sliding_window import sliding_window


def preprocessing(dataset, path, path_out, save_dir="", sensors_type='all', positions='all', magnitude=False, size_overlapping=0.5):

    if dataset == 'unimib':
        unimib_process(path, path_out, magnitude, size_overlapping)
    elif dataset == 'sbhar':
        sbhar_process(path, path_out, magnitude, size_overlapping)
    elif dataset == 'realdisp':
        realdisp_process(path, path_out, save_dir, sensors_type,
                         positions, magnitude, size_overlapping)


def realdisp_process(path, path_out, save_dir, sensors_type='all', positions='all', magnitude=True, size_overlapping=0.5):
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

    print('Processing realdisp dataset')

    root_path = path + 'REALDISP'
    raw_data_path = root_path + '/'
    if magnitude:
        processed_path = path_out + \
            'OuterPartition_magnitude_prova_balance_{}'.format(
                str(size_overlapping*10))
    else:
        processed_path = path_out + \
            'OuterPartition_{}'.format(str(size_overlapping*10))

    win_len = 100
    channel = 3
    ID_generater = 1
    if sensors_type == 'acc_gyro_magn':
        number_sensor = 3
    else:
        number_sensor = 2

    data = np.empty([0, win_len, channel*number_sensor], dtype=np.float)
    la = np.empty([0], dtype=np.int32)
    lu = np.empty([0], dtype=np.int32)
    ID = np.empty([0], dtype=np.int32)

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    for fl in os.listdir(raw_data_path):
        if fl.startswith('subject'):
            print("Leggo log file ", fl)
            log_file = np.loadtxt(fname=raw_data_path + fl)
            id_user = int(fl.split('_')[0].split('subject')[1])
            # print(id_user)

            # take only acc and gyro data
            sensors = log_file[:, 2:119]
            activities = log_file[:, -1].astype('int32')
            activities = activities.reshape(sensors.shape[0], 1)
            trials = log_file[:, 0].astype('int32')
            trials = trials.reshape(sensors.shape[0], 1)

            step = 0

            # take accelerometer, gyroscope
            if sensors_type == 'acc_gyro':
                offset = 6
                to_del = 7
            elif sensors_type == 'acc_gyro_magn':
                offset = 9
                to_del = 4

            for _ in range(9):
                sensors = np.delete(sensors, np.arange(
                    (offset*step) + offset, (offset*step) + offset + to_del), axis=1)
                step += 1

            step = 0

            # filter positions based on parameter positions
            if positions != 'all':
                if 'RLA' not in positions:
                    acc_gyro = acc_gyro[:, step*offset:]
                else:
                    step += 1
                if 'RUA' not in positions:
                    acc_gyro = acc_gyro[:, step*offset:]
                else:
                    step += 1
                if 'BACK' not in positions:
                    acc_gyro = acc_gyro[:, step*offset:]
                else:
                    step += 1
                if 'LUA' not in positions:
                    acc_gyro = acc_gyro[:, step*offset:]
                else:
                    step += 1
                if 'LLA' not in positions:
                    acc_gyro = acc_gyro[:, step*offset:]
                else:
                    step += 1
                if 'RC' not in positions:
                    acc_gyro = acc_gyro[:, step*offset:]
                else:
                    step += 1
                if 'RT' not in positions:
                    acc_gyro = acc_gyro[:, step*offset:]
                else:
                    step += 1
                if 'LT' not in positions:
                    acc_gyro = acc_gyro[:, step*offset:]
                else:
                    step += 1
                if 'LC' not in positions:
                    acc_gyro = acc_gyro[:, step*offset:]

            sensors = np.concatenate((trials, sensors, activities), axis=1)

            # delete record with unknown activity label
            sensors = np.delete(sensors, np.where(sensors[:, -1] == 0), axis=0)

            # cycle on activity
            for id_act in np.unique(sensors[:, -1]):
                # print(int(id_act))
                step = 0

                temp = sensors[np.where(sensors[:, -1] == int(id_act))]

                # sliding window on every of 9 sensor
                for _ in range(9):

                    acc = temp[:, 1+(offset*step):(step*offset)+4]
                    gyro = temp[:, 4+(offset*step):(step*offset)+7]

                    if magnitude:
                        acc = np.concatenate((acc, np.apply_along_axis(lambda x: np.sqrt(
                            np.sum(np.power(x, 2))), axis=1, arr=acc).reshape(-1, 1)), axis=1)
                        gyro = np.concatenate((gyro, np.apply_along_axis(lambda x: np.sqrt(
                            np.sum(np.power(x, 2))), axis=1, arr=gyro).reshape(-1, 1)), axis=1)

                    if sensors_type == 'acc_gyro_magn':
                        magn = temp[:, 7+(offset*step):(step*offset)+10]
                        if magnitude:
                            magn = np.concatenate((magn, np.apply_along_axis(lambda x: np.sqrt(
                                np.sum(np.power(x, 2))), axis=1, arr=magn).reshape(-1, 1)), axis=1)

                    try:
                        _data_windows_acc = sliding_window(
                            acc, (win_len, channel), (int(win_len/2), 1))
                    except:
                        print("Not enough data for sliding window")
                    invalid_idx = np.where(np.any(
                        np.isnan(np.reshape(_data_windows_acc, [-1, win_len*channel])), axis=1))[0]
                    _data_windows_acc = np.delete(
                        _data_windows_acc, invalid_idx, axis=0)

                    _data_windows_gyro = sliding_window(
                        gyro, (win_len, channel), (int(win_len/2), 1))
                    invalid_idx = np.where(np.any(
                        np.isnan(np.reshape(_data_windows_gyro, [-1, win_len*channel])), axis=1))[0]
                    _data_windows_gyro = np.delete(
                        _data_windows_gyro, invalid_idx, axis=0)

                    if sensors_type == 'acc_gyro_magn':
                        _data_windows_magn = sliding_window(
                            magn, (win_len, channel), (int(win_len/2), 1))
                        invalid_idx = np.where(np.any(
                            np.isnan(np.reshape(_data_windows_magn, [-1, win_len*channel])), axis=1))[0]
                        _data_windows_magn = np.delete(
                            _data_windows_magn, invalid_idx, axis=0)

                    try:
                        if sensors_type == 'all':
                            temp_sensor = np.concatenate(
                                (_data_windows_acc, _data_windows_gyro), axis=2)
                        temp_sensor = np.concatenate(
                            (_data_windows_acc, _data_windows_gyro, _data_windows_magn), axis=2)
                    except:
                        print("There is only one sliding window")
                        if sensors_type == 'all':
                            temp_sensor = np.concatenate(
                                (_data_windows_acc, _data_windows_gyro), axis=1)
                        else:
                            temp_sensor = np.concatenate(
                                (_data_windows_acc, _data_windows_gyro, _data_windows_magn), axis=1)
                        temp_sensor = temp_sensor.reshape(
                            (1, temp_sensor.shape[0], temp_sensor.shape[1]))

                    step += 1

                    # id for every window
                    _id = np.arange(
                        ID_generater, ID_generater+len(temp_sensor))
                    # concat verticaly every window
                    data = np.concatenate((data, temp_sensor), axis=0)
                    ID = np.concatenate((ID,   _id), axis=0)
                    ID_generater = ID_generater + len(temp_sensor) + 10

                #ID_generater = ID_generater + len(temp_sensor) + 10

                # update la
                # label activity for every window
                _la = np.full(len(data)-len(la),
                              int(id_act) - 1, dtype=np.int32)
                la = np.concatenate((la, _la), axis=0)

            # update gloabl variabel lu
            _lu = np.full(len(data)-len(lu), int(id_user) - 1,
                          dtype=np.int32)  # label user for every window
            lu = np.concatenate((lu, _lu), axis=0)

    # shuffle
    data, la, lu, ID = skutils.shuffle(data, la, lu, ID)
    data, la, lu, ID = skutils.shuffle(data, la, lu, ID)

    indexes = split_balanced_data(lu, la, folders=10)

    plt_user_distribution(indexes, lu)

    plt_act_distribution(indexes, la)

    # partition
    for i in range(10):

        # clear dir
        if os.path.exists(processed_path+'/fold{}'.format(i)):
            shutil.rmtree(processed_path+'/fold{}'.format(i))
        os.mkdir(processed_path+'/fold{}'.format(i))

        #idx = np.arange(int(len(data)*0.1*i), int(len(data)*0.1*(i+1)), 1)
        idx = indexes[str(i)]
        np.save(processed_path+'/fold{}/data'.format(i),       data[idx])
        np.save(processed_path+'/fold{}/user_label'.format(i), lu[idx])
        np.save(processed_path+'/fold{}/act_label'.format(i),  la[idx])
        np.save(processed_path+'/fold{}/id'.format(i),         ID[idx])


def sbhar_process(path, path_out, magnitude, size_overlapping):
    print('Processing sbhar dataset')

    root_path = path + 'SBHAR'
    raw_data_path = root_path + '/RawData/'
    if magnitude:
        processed_path = path_out + \
            'OuterPartition_magnitude_{}'.format(
                str(size_overlapping*10))
    else:
        processed_path = path_out + \
            'OuterPartition_{}'.format(str(size_overlapping*10))

    win_len = 100
    if magnitude:
        channel = 4
    else:
        channel = 3
    ID_generater = 1
    number_sensor = 2

    data = np.empty([0, win_len, channel*number_sensor], dtype=np.float)
    la = np.empty([0], dtype=np.int32)
    lu = np.empty([0], dtype=np.int32)
    ID = np.empty([0], dtype=np.int32)

    if not os.path.exists(path_out):
        os.mkdir(path_out)

    # read labels.txt with inf about data (id_exp, id_usr, id_activity, start_sample, stop_sample)
    info = np.loadtxt(fname=raw_data_path + 'labels.txt')

    # for subject
    for id_user in np.unique(info[:, 1]):
        user_info = info[np.where(info[:, 1] == int(id_user))]

        # for activity
        for id_act in np.unique(user_info[:, 2]):
            activity_info = user_info[np.where(user_info[:, 2] == int(id_act))]
            acc = np.empty([0, 3], dtype=np.float)
            gyro = np.empty([0, 3], dtype=np.float)

            for tid in np.unique(activity_info[:, 0]):
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
                    #print(start, " ", stop)
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
                    acc, (win_len, channel), (int(win_len*(1-size_overlapping)), 1))
            except:
                print("Not enough data for sliding window")
            invalid_idx = np.where(np.any(
                np.isnan(np.reshape(_data_windows_acc, [-1, win_len*channel])), axis=1))[0]
            _data_windows_acc = np.delete(
                _data_windows_acc, invalid_idx, axis=0)

            _data_windows_gyro = sliding_window(
                gyro, (win_len, channel), (int(win_len*(1-size_overlapping)), 1))
            invalid_idx = np.where(np.any(
                np.isnan(np.reshape(_data_windows_gyro, [-1, win_len*channel])), axis=1))[0]
            _data_windows_gyro = np.delete(
                _data_windows_gyro, invalid_idx, axis=0)

            try:
                acc_gyro = np.concatenate(
                    (_data_windows_acc, _data_windows_gyro), axis=2)
            except:
                print("There is only one sliding window")
                acc_gyro = np.concatenate(
                    (_data_windows_acc, _data_windows_gyro), axis=1)
                acc_gyro = acc_gyro.reshape(
                    (1, acc_gyro.shape[0], acc_gyro.shape[1]))

            _id = np.arange(ID_generater, ID_generater +
                            len(acc_gyro))  # id for every window
            # concat verticaly every window
            data = np.concatenate((data, acc_gyro), axis=0)
            ID = np.concatenate((ID,   _id), axis=0)
            ID_generater = ID_generater + len(acc_gyro) + 10

            # label activity for every window
            _la = np.full(len(data)-len(la), int(id_act) - 1, dtype=np.int32)
            la = np.concatenate((la, _la), axis=0)

        _lu = np.full(len(data)-len(lu), int(id_user) - 1,
                      dtype=np.int32)  # label user for every window
        lu = np.concatenate((lu, _lu), axis=0)

    # shuffle
    data, la, lu, ID = skutils.shuffle(data, la, lu, ID)
    data, la, lu, ID = skutils.shuffle(data, la, lu, ID)

    if not os.path.exists(processed_path + '/'):
        os.mkdir(processed_path + '/')

    indexes = split_balanced_data(lu, la, folders=10)

    #plt_user_distribution(indexes, lu)

    #plt_act_distribution(indexes, la)

    # partition
    for i in range(10):

        # clear dir
        if os.path.exists(processed_path+'/fold{}'.format(i)):
            shutil.rmtree(processed_path+'/fold{}'.format(i))
        os.mkdir(processed_path+'/fold{}'.format(i))

        #idx = np.arange(int(len(data)*0.1*i), int(len(data)*0.1*(i+1)), 1)
        idx = indexes[str(i)]
        np.save(processed_path+'/fold{}/data'.format(i),       data[idx])
        np.save(processed_path+'/fold{}/user_label'.format(i), lu[idx])
        np.save(processed_path+'/fold{}/act_label'.format(i),  la[idx])
        np.save(processed_path+'/fold{}/id'.format(i),         ID[idx])


def unimib_process(path, path_out, magnitude, size_overlapping):
    print('Processing unimib dataset')

    root_path = path + 'unimib_dataset'
    raw_data_path = root_path + '/data/'
    if magnitude:
        processed_path = path_out + \
            'OuterPartition_magnitude_{}'.format(
                str(size_overlapping*10))
    else:
        processed_path = path_out + \
            'OuterPartition_{}'.format(str(size_overlapping*10))
    win_len = 100   # 50 Hz * 2 seconds
    if magnitude:
        channel = 4
    else:
        channel = 3
    ID_generater = 1

    data = np.empty([0, win_len, channel], dtype=np.float)
    la = np.empty([0], dtype=np.int32)
    lu = np.empty([0], dtype=np.int32)
    ID = np.empty([0], dtype=np.int32)

    signal = loadmat(raw_data_path + 'full_data.mat')['full_data']

    activity_table = ['StandingUpFS', 'StandingUpFL', 'Walking', 'Running', 'GoingUpS', 'Jumping', 'GoingDownS', 'LyingDownFS', 'SittingDown',
                      'FallingForw', 'FallingRight', 'FallingBack', 'HittingObstacle', 'FallingWithPS', 'FallingBackSC', 'Syncope', 'FallingLeft']

    # id subject
    for sid in range(30):
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
                    _data, (win_len, channel), (int(win_len*(1-size_overlapping)), 1))
                invalid_idx = np.where(
                    np.any(np.isnan(np.reshape(_data_windows, [-1, win_len*channel])), axis=1))[0]
                _data_windows = np.delete(_data_windows, invalid_idx, axis=0)
                _id = np.arange(ID_generater, ID_generater +
                                len(_data_windows))  # id for every window

                # concat verticaly every window
                data = np.concatenate((data, _data_windows), axis=0)
                ID = np.concatenate((ID,   _id), axis=0)
                ID_generater = ID_generater + len(_data_windows) + 10

            # label activity for every window
            _la = np.full(len(data)-len(la), aid, dtype=np.int32)
            la = np.concatenate((la, _la), axis=0)

        # label user for every window
        _lu = np.full(len(data)-len(lu), sid, dtype=np.int32)
        lu = np.concatenate((lu, _lu), axis=0)

    # shuffle
    data, la, lu, ID = skutils.shuffle(data, la, lu, ID)
    data, la, lu, ID = skutils.shuffle(data, la, lu, ID)

    if not os.path.exists(processed_path + '/'):
        os.mkdir(processed_path + '/')

    # split balanced data, return array (10, indexes_folder)
    indexes = split_balanced_data(lu, la, folders=10)

    #plt_user_distribution(indexes, lu)

    #plt_act_distribution(indexes, la)

    # create dir partition
    for i in range(10):

        # clear dir
        if os.path.exists(processed_path+'/fold{}'.format(i)):
            shutil.rmtree(processed_path+'/fold{}'.format(i))
        os.mkdir(processed_path+'/fold{}'.format(i))

        #idx = np.arange(int(len(data)*0.1*i), int(len(data)*0.1*(i+1)), 1)
        idx = indexes[str(i)]
        np.save(processed_path+'/fold{}/data'.format(i),       data[idx])
        np.save(processed_path+'/fold{}/user_label'.format(i), lu[idx])
        np.save(processed_path+'/fold{}/act_label'.format(i),  la[idx])
        np.save(processed_path+'/fold{}/id'.format(i),         ID[idx])


def split_balanced_data(lu, la, folders=10):

    print('Numero totale di esempi: {}'.format(len(lu)))

    # dict to save indexes' example for every folder
    indexes = {}
    for i in np.arange(folders):
        indexes[str(i)] = []

    #distribution_act = [0 for _ in range(0,np.unique(la).shape[0])]
    # print(len(distribution_act))

    last_folder = 0

    # balance split label user-activity in every folders
    for user in np.unique(lu):  # specific user
        temp_index_label_user = [index for index, x in enumerate(
            lu) if x == user]  # index of specific user

        for act in np.unique(la):  # specific activity
            temp_index_label_act = [index for index, x in enumerate(
                la) if x == act and index in temp_index_label_user]  # index of specific activity of user
            #distribution_act[act] += len(temp_index_label_act)

            while(len(temp_index_label_act) > 0):
                for folder in range(last_folder, folders):
                    if len(temp_index_label_act) > 0:
                        indexes[str(folder)].append(temp_index_label_act[0])
                        del temp_index_label_act[0]
                        if folder == folders-1:
                            last_folder = 0
                        else:
                            last_folder = folder
                    else:
                        continue

    for key in indexes.keys():
        print(f'Numero campioni nel folder {key}: {len(indexes[key])}')

    for folder1, l1 in zip(indexes.keys(), indexes.values()):
        for folder2, l2 in zip(indexes.keys(), indexes.values()):
            if folder1 != folder2:
                a = [el1 for el1 in l1 if any(el2 == el1 for el2 in l2)]
                if a != []:
                    print('duplicates elements in different folder')
                    exit()
    return indexes


def plt_user_distribution(dict_indexes, lu):

    plt.figure(figsize=(12, 3))
    plt.style.use('seaborn-darkgrid')

    for folder in np.arange(len(dict_indexes)):
        plt.subplot(2, 5, folder+1)
        plt.title('folder {}'.format(folder+1))
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
        plt.subplot(2, 5, folder+1)
        plt.title('folder {}'.format(folder+1))
        folder_index = dict_indexes[str(folder)]
        act_distributions = []
        for act in np.unique(la):
            number_act = len([i for index, i in enumerate(
                la) if i == act and index in folder_index])
            act_distributions.append(number_act)

        plt.bar(x=np.arange(len(act_distributions)), height=act_distributions)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="preprocessing pipeline for signal")

    parser.add_argument('-d', '--dataset', type=str, choices=[
                        'unimib', 'sbhar', 'realdisp', 'ouisir'], default="unimib", help='dataset to preprocessing')
    parser.add_argument('-p', '--path', type=str,
                        default='../data/datasets/', help='path to dataset')
    parser.add_argument('-o', '--out', type=str, default='../data/datasets/UNIMIBDataset/',
                        help='path to store data preprocessed')

    args = parser.parse_args()
    for magnitude in [False, True]:
        for overlap in [0.5, 0.6, 0.7, 0.8, 0.9]:
            preprocessing("unimib", "../data/datasets/", "../data/datasets/UNIMIBDataset/", magnitude=magnitude, size_overlapping=overlap)
            preprocessing("sbhar", "../data/datasets/", "../data/datasets/SBHAR_processed/", magnitude=magnitude, size_overlapping=overlap)
            #preprocessing('realdisp', "../data/datasets/", "../data/datasets/REALDISP_processed/", sensors_type="acc_gyro_magn",
            #              save_dir='', positions="all", magnitude=magnitude, size_overlapping=overlap)
