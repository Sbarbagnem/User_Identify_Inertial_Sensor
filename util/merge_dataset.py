import numpy as np
import os
import shutil
from sklearn import utils as skutils
import matplotlib.pyplot as plt

from configuration import config


def merge_data(dataset_a, dataset_b):
    
    '''
        merge data
    '''

    sensors_to_merge = min(dataset_a.shape[-1], dataset_b.shape[-1])
    dataset_a = dataset_a[:,:,:sensors_to_merge]
    dataset_b = dataset_b[:,:,:sensors_to_merge]

    merged_dataset = np.concatenate((dataset_a, dataset_b))

    return merged_dataset

def read_outer_partition(dataset, path_outer, magnitude=True, overlap=5.0):
    if magnitude:
        path = path_outer + 'OuterPartition_magnitude_' + str(overlap) + '/'
        channel = config[dataset]['WINDOW_AXES'] + len(list(config[dataset]['SENSOR_DICT'].keys()))
    else:
        path = path_outer + 'OuterPartition_' + str(overlap) + '/'
        channel = config[dataset]['WINDOW_AXES'] 
    
    data = np.empty([0, 100, channel], dtype=np.float)
    lu = np.empty([0], dtype=np.int32)
    idx = np.empty([0], dtype=np.int32)

    for fold in os.listdir(path):
        data_temp = np.load(os.path.join(path, fold, 'data.npy'))
        lu_temp = np.load(os.path.join(path, fold, 'user_label.npy'))
        idx_temp = np.load(os.path.join(path, fold, 'id.npy'))
        data = np.concatenate((data, data_temp))
        lu = np.concatenate((lu, lu_temp))
        idx = np.concatenate((idx, idx_temp))
        

    return data, lu, idx

def uniform_user_index(list_user, list_index):

    for i in range(len(list_user))[1:]:
        max_temp = np.max(list_user[i-1])
        list_user[i] = list(list_user[i] + max_temp + 1)
        max_temp = np.max(list_index[i-1])
        list_index[i] = list(list_index[i] + max_temp)

    # flat list
    list_user = [item for sublist in list_user for item in sublist]
    list_index = [item for sublist in list_index for item in sublist]
    return list_user, list_index

def save_mergede_dataset(path_to_save, data, lu, idx):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    data, lu, idx = skutils.shuffle(data, lu, idx)
    data, lu, idx = skutils.shuffle(data, lu, idx)

    lu = np.asarray(lu)
    idx = np.asarray(idx)

    # create dir partition
    for i in range(10):

        # clear dir
        if os.path.exists(path_to_save+'/fold{}'.format(i)):
            shutil.rmtree(path_to_save+'/fold{}'.format(i))
        os.mkdir(path_to_save+'/fold{}'.format(i))

        idx_temp = np.arange(int(len(data)*0.1*i), int(len(data)*0.1*(i+1)), 1)
        #idx = indexes[str(i)]
        np.save(path_to_save+'/fold{}/data'.format(i),       data[idx_temp])
        np.save(path_to_save+'/fold{}/user_label'.format(i), lu[idx_temp])
        np.save(path_to_save+'/fold{}/id'.format(i),         idx[idx_temp])

def balance_dataset(data, label, idx):

    label = np.asarray(label)
    idx = np.asarray(idx)

    data_balanced = np.empty([0, data.shape[1], data.shape[2]], dtype=np.float)
    label_balanced = np.empty([0], dtype=np.int32)
    idx_balanced = np.empty([0], dtype=np.int32)

    labels, freq = np.unique(label, return_counts=True)
    min_freq = min(freq)
    max_freq = max(freq)

    max_freq_balanc = int((max_freq - min_freq)/3 + min_freq)

    for user in labels:
        idx_user = np.where(label==user)[0] # indici dati user
        if len(idx_user) > max_freq_balanc:
            idx_user = idx_user[:max_freq_balanc]
        data_balanced = np.concatenate((data_balanced, data[idx_user,:,:]))
        label_balanced = np.concatenate((label_balanced, label[idx_user]))
        idx_balanced = np.concatenate((idx_balanced, idx[idx_user]))

    return data_balanced, label_balanced, idx_balanced

if __name__ == "__main__":

    dataset_a = 'unimib'
    dataset_b = 'sbhar'
    path_a = config[dataset_a]['PATH_OUTER_PARTITION']
    path_b = config[dataset_b]['PATH_OUTER_PARTITION']
    magnitude = True
    overlap = 5.0

    data_A, lu_A, idx_A = read_outer_partition(dataset_a, path_a, magnitude=magnitude, overlap=overlap)
    data_B, lu_B, idx_B = read_outer_partition(dataset_b, path_b, magnitude=magnitude, overlap=overlap)

    lu, idx = uniform_user_index(list_user = [lu_A,lu_B], list_index=[idx_A,idx_B]) 
    data = merge_data(data_A, data_B)
    '''
    label, freq = np.unique(lu, return_counts=True)
    l = list(range(1, len(label)+1))
    plt.barh(l, width=freq, height=0.5)
    plt.yticks(l, label, rotation='horizontal')
    plt.show()
    
    # balance dataset (downsample majority classes)
    data, lu, idx = balance_dataset(data, lu, idx)
    
    label, freq = np.unique(lu, return_counts=True)
    l = list(range(1, len(label)+1))
    plt.barh(l, width=freq, height=0.5)
    plt.yticks(l, label, rotation='horizontal')
    plt.show()
    '''
    path_to_save = f'../data/datasets/merged_{dataset_a}_{dataset_b}/OuterPartition_'
    if magnitude:
        path_to_save = path_to_save + 'magnitude_'
    path_to_save = path_to_save + str(overlap) + '/'

    save_mergede_dataset(path_to_save, data, lu, idx)