import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pywt
from pprint import pprint
from sklearn import utils as skutils
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import scale as scale_sklearn
from scipy import signal, fftpack
from scipy.signal import find_peaks as find_peaks_scipy
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d  


def plot_performance(ActivityAccuracy, UserAccuracy, fold, path_to_save, save=False):

    plt.plot(ActivityAccuracy)
    plt.plot(UserAccuracy)
    plt.title('Fold {} for test'.format(fold))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Activity_accuracy', 'User_accuracy'], loc='lower right')

    if save:
        plt.savefig(path_to_save + 'plot_{}.png'.format(fold))

    plt.show()


def mean_cross_performance(history):

    mean_activity_accuracy = np.mean(history[:, 1], axis=0)
    mean_user_accuracy = np.mean(history[:, 3], axis=0)

    return mean_activity_accuracy, mean_user_accuracy


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def split_balanced_data(lu, la, folders, di=None, log=True):

    if log:
        print('Numero totale di esempi: {}'.format(len(lu)))

    # dict to save indexes' example for every folder
    indexes = {}
    for i in np.arange(folders):
        indexes[str(i)] = []

    last_folder = 0

    if di is not None:
        for displace in np.unique(di):
            temp_index_label_displace = [index for index, x in enumerate(
                di) if x == displace]  # index of specific displace
            for user in np.unique(lu):
                temp_index_label_user = [index for index, x in enumerate(
                    lu) if x == user and index in temp_index_label_displace]  # index of specific user
                for act in np.unique(la):
                    temp_index_label_act = [index for index, x in enumerate(
                        la) if x == act and index in temp_index_label_user]  # index of specific activity of user
                    # same percentage data in every folder
                    while(len(temp_index_label_act) > 0):
                        for folder in range(last_folder, folders):
                            if len(temp_index_label_act) > 0:
                                indexes[str(folder)].append(
                                    temp_index_label_act[0])
                                del temp_index_label_act[0]
                                if folder == folders - 1:
                                    last_folder = 0
                                else:
                                    last_folder = folder
                            else:
                                continue
    else:
        for user in np.unique(lu):
            temp_index_label_user = [index for index, x in enumerate(
                lu) if x == user]  # index of specific user
            for act in np.unique(la):
                temp_index_label_act = [index for index, x in enumerate(
                    la) if x == act and index in temp_index_label_user]  # index of specific activity of user
                # same percentage data in every folder
                while(len(temp_index_label_act) > 0):
                    for folder in range(last_folder, folders):
                        if len(temp_index_label_act) > 0:
                            indexes[str(folder)].append(
                                temp_index_label_act[0])
                            del temp_index_label_act[0]
                            if folder == folders - 1:
                                last_folder = 0
                            else:
                                last_folder = folder
                        else:
                            continue
    if log:
        for key in indexes.keys():
            print(f'Numero campioni nel folder {key}: {len(indexes[key])}')

    return indexes


def delete_overlap(train_id, val_id, distances_to_delete):
    overlap_ID = np.empty([0], dtype=np.int32)

    for distance in distances_to_delete:
        overlap_ID = np.concatenate(
            (overlap_ID, val_id+distance, val_id-distance))

    overlap_ID = np.unique(overlap_ID)
    invalid_idx = np.array([i for i in np.arange(
        len(train_id)) if train_id[i] in overlap_ID])

    return invalid_idx


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


def mapping_act_label(dataset_name):
    if 'unimib' in dataset_name or 'shar' in dataset_name:
        return ['StandingUpFS', 'StandingUpFL', 'Walking', 'Running', 'GoingUpS',
                'Jumping', 'GoingDownS', 'LyingDownFS', 'SittingDown']
    if 'sbhar' in dataset_name:
        return ['Walking', 'Walking upstairs', 'Walking downstairs', 'Sitting', 'Standing',
                'Laying', 'Stand to sit', 'Sit to stand', 'Sit to lie', 'Lie to sit',
                'Stand to lie', 'Lie to stan']
    if 'realdisp' in dataset_name:
        return ['Walking', 'Jogging', 'Running', 'Jump up', 'Jump front & back', 'Jump sideways', 'Jump leg/arms open/closed',
                'Jump rope', 'Trunk twist (arms outstretched)', 'Trunk twist (elbows bended)', 'Waist bends forward', 'Waist rotation',
                'Waist bends opposite hands', 'Reach heels backwards', 'Lateral bend', 'Lateral bend arm up', 'Repetitive forward stretching',
                'Upper trunk and lower body', 'Arms lateral elevation', 'Arms frontal elevation', 'Frontal hand claps', 'Arms frontal crossing',
                'Shoulders high rotation', 'Shoulders low rotation', 'Arms inner rotation', 'Knees to breast', 'Heels to backside', 'Nkees bending',
                'Knees bend forward', 'Rotation on knees', 'Rowing', 'Elliptic bike', 'Cycling']


def plot_pred_based_act(correct_predictions, label_act, folds=1, title='', dataset_name='', file_name='', colab_path=None, save_plot=False, save_txt=False, show_plot=False):

    if np.array(correct_predictions).ndim == 1:
        correct = correct_predictions
    else:
        correct = np.sum(correct_predictions, axis=0)/folds

    width = 0.35

    plt.bar(np.arange(0, len(label_act)), correct, width, color='g')
    plt.ylabel('% correct prediction')
    plt.xlabel('Activity')
    plt.title(title, pad=5)
    plt.xticks(np.arange(0, len(label_act)), label_act, rotation='vertical')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()

    if colab_path is not None:
        path_to_save = colab_path + f'/plot/{dataset_name}/'
    else:
        path_to_save = f'plot/{dataset_name}/'

    if save_plot:
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        plt.savefig(path_to_save + f'{file_name}.png')

    if save_txt:
        if os.path.isfile(path_to_save + 'performance_based_act.txt'):
            f = open(path_to_save + 'performance_based_act.txt', 'a+')
        else:
            f = open(path_to_save + 'performance_based_act.txt', 'w+')
        for l, p in zip(label_act, correct):
            f.write(f"{l}: {p}\r\n")
        f.close()
    if show_plot:
        plt.show()


def save_mean_performance_txt(performances, dataset_name, colab_path):

    if colab_path is not None:
        path_to_save = colab_path + f'/mean_performance/{dataset_name}/'
    else:
        path_to_save = f'mean_performance/{dataset_name}/'

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    if os.path.isfile(path_to_save + 'mean_performance.txt'):
        f = open(path_to_save + 'mean_performance.txt', 'a+')
    else:
        f = open(path_to_save + 'mean_performance.txt', 'w+')

    for key in list(performances.keys()):
        f.write(f"{key}: {performances[key]}\r\n")
    f.close()


def smooth(coef):
    window_len = 5
    s = np.r_[coef[window_len-1:0:-1], coef, coef[-1:-window_len:-1]]
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def detectGaitCycle(data, plot_peak=False, plot_auto_corr_coeff=False, gcLen=None):

    selected_data = data[:,2] # z axis
    samples = data.shape[0]

    autocorr = False if gcLen != None else True

    peaks = find_thresh_peak(selected_data)

    # compute gcLen based on autocorrelation of signal if not given by default
    gcLen, auto_corr_coeff, peak_auto_corr = find_gcLen(data)

    if plot_auto_corr_coeff:
        plt.figure(figsize=(12, 3))
        plt.style.use('seaborn-darkgrid')
        plt.plot(np.arange(len(auto_corr_coeff)), auto_corr_coeff, 'b-')
        plt.scatter(peak_auto_corr, auto_corr_coeff[peak_auto_corr], c='red')
        plt.tight_layout()
        plt.show()

    peaks, to_plot = find_peaks(peaks, selected_data, gcLen, autocorr)

    if plot_peak or False: 
        plt.figure(figsize=(12, 3))
        plt.style.use('seaborn-darkgrid')
        plt.subplot(3,1,1)
        plt.plot(np.arange(samples), data[:,0], 'b-', label='x')
        plt.vlines(peaks, ymin=min(data[:,0]), ymax=max(data[:,0]), color='black', ls='dotted')
        plt.legend(loc='upper right')
        plt.subplot(3,1,2)
        plt.plot(np.arange(samples), data[:,1], 'g-', label='y')
        plt.vlines(peaks, ymin=min(data[:,1]), ymax=max(data[:,1]), color='black', ls='dotted')
        plt.legend(loc='upper right')
        plt.subplot(3,1,3)
        plt.plot(np.arange(samples), data[:,2], 'r-', label='z')
        plt.vlines(peaks, ymin=min(data[:,2]), ymax=max(data[:,2]), color='black', ls='dotted')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    return peaks

def segment2GaitCycle(peaks, segment, plot_split):
    cycles = []
    for i in range(0, len(peaks)-1):
        cycle = segment[peaks[i]:peaks[i+1],:]
        cycles.append(cycle)
    if plot_split:
        for cycle in cycles:
            if segment2GaitCycle:
                plt.figure(figsize=(12, 3))
                plt.style.use('seaborn-darkgrid')
                plt.plot(np.arange(cycle.shape[0]), cycle[:,0], 'b-', label='x')
                plt.plot(np.arange(cycle.shape[0]), cycle[:,1], 'g-', label='y')
                plt.plot(np.arange(cycle.shape[0]), cycle[:,2], 'r-', label='z')
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.show()
    return cycles


def split_data_train_val_test_gait(data,
                                   label_user,
                                   id_window,
                                   sessions,
                                   method,
                                   overlap,
                                   split,
                                   plot_split):

    train_data = []
    val_data = []
    test_data = []
    train_label = []
    val_label = []
    test_label = []

    if method == 'cycle_based':
        for user in np.unique(label_user):

            # filter for user
            idx = np.where(label_user == user)
            data_temp = data[idx]
            user_temp = label_user[idx]

            # shuffle cycles to take random between first and second session
            data_temp, user_temp = skutils.shuffle(data_temp, user_temp)

            samples = data_temp.shape[0]

            # split cycles based on paper, 8 gait for train, 0.5 of remain for val and 0.5 for test
            if split == 'paper':
                # gait cycle for train
                train_gait = 8
                # to have at least one sample for every user in train, val and test
                if samples < 10:
                    train_gait = samples - 2
                    val_gait = 1
                else:
                    val_gait = round((samples - train_gait)/2)
            # split cycles in a standard way 70% train, 20% val and 10% test
            else:
                if samples <= 5:
                    train_gait = samples - 2
                    val_gait = 1
                else:
                    train_gait = round(samples*0.7)
                    val_gait = round(samples*0.2)                

            # train
            train_data.append(data_temp[:train_gait])
            train_label.extend(user_temp[:train_gait])

            # val
            val_data.append(data_temp[train_gait:val_gait+train_gait])
            val_label.extend(user_temp[train_gait:val_gait+train_gait])

            # test
            test_data.append(data_temp[val_gait+train_gait:])
            test_label.extend(user_temp[val_gait+train_gait:])

            # plot train val and test cycle for user
            if plot_split:
                plt.figure(figsize=(12, 3))
                plt.style.use('seaborn-darkgrid')
                for i,c in enumerate(train_data[-1][:5]):
                    plt.subplot(3, 5, i+1)
                    plt.plot(np.arange(c.shape[0]), c[:, 0], 'g-', label='x') 
                    plt.plot(np.arange(c.shape[0]), c[:, 1], 'r-', label='y') 
                    plt.plot(np.arange(c.shape[0]), c[:, 2], 'b-', label='z') 
                for i,c in enumerate(val_data[-1][:5]):
                    plt.subplot(3, 5, i+1+5)
                    plt.plot(np.arange(c.shape[0]), c[:, 0], 'g-', label='x') 
                    plt.plot(np.arange(c.shape[0]), c[:, 1], 'r-', label='y') 
                    plt.plot(np.arange(c.shape[0]), c[:, 2], 'b-', label='z') 
                for i,c in enumerate(test_data[-1][:5]):
                    plt.subplot(3, 5, i+1+10)
                    plt.plot(np.arange(c.shape[0]), c[:, 0], 'g-', label='x') 
                    plt.plot(np.arange(c.shape[0]), c[:, 1], 'r-', label='y') 
                    plt.plot(np.arange(c.shape[0]), c[:, 2], 'b-', label='z') 
                plt.tight_layout()
                plt.show()            

    elif method == 'window_based':

        if overlap == None:
            raise Exception('Overlap must not be empty for window base method')

        if overlap == 50:
            distances_to_delete = [1]
        elif overlap == 75:
            distances_to_delete = [1, 2, 3]

        # 70% train, 20% val, 10% test
        for user in np.unique(label_user):

            idx = np.where(label_user == user)
            data_temp = data[idx]
            user_temp = label_user[idx]
            id_temp = id_window[idx]

            # shuffle for random pick
            data_temp, user_temp, id_temp = skutils.shuffle(data_temp, user_temp, id_temp)

            # number of window for user and session, in train, val and test
            samples = data_temp.shape[0]
            train_val_percentage = round(samples*0.9)
            if train_val_percentage == samples:
                train_val_percentage -= 1

            # train_val
            train = data_temp[:train_val_percentage]
            user_train = user_temp[:train_val_percentage]
            id_train = id_temp[:train_val_percentage]

            # test
            test = data_temp[train_val_percentage:]
            user_test = user_temp[train_val_percentage:]
            id_test = id_temp[train_val_percentage:]

            # delete overlap sequence between train and test
            if overlap != 0:
                overlap_idx = delete_overlap(
                    id_train, id_test, distances_to_delete)
                train_temp = np.delete(train, overlap_idx, axis=0)
                user_train_temp = np.delete(user_train, overlap_idx, axis=0)
            else:
                train_temp = train
                user_train_temp = user_train
            
            # split train in train and val
            train_percentage = int(train_temp.shape[0] * 0.8)
            if train_percentage == train_temp.shape[0]:
                train_percentage -= 1
            train = train_temp[:train_percentage]
            user_train = user_train_temp[:train_percentage]
            val = train_temp[train_percentage:]
            user_val = user_train_temp[train_percentage:]

            # train
            train_data.append(train)
            train_label.extend(user_train)

            # val
            val_data.append(val)
            val_label.extend(user_val)

            # test
            test_data.append(test)
            test_label.extend(user_test)

            if plot_split:
                plt.figure(figsize=(12, 3))
                plt.style.use('seaborn-darkgrid')
                for i,c in enumerate(train_data[-1][:5]):
                    plt.subplot(3, 5, i+1)
                    plt.plot(np.arange(c.shape[0]), c[:, 0], 'g-', label='x') 
                    plt.plot(np.arange(c.shape[0]), c[:, 1], 'r-', label='y') 
                    plt.plot(np.arange(c.shape[0]), c[:, 2], 'b-', label='z') 
                for i,c in enumerate(val_data[-1][:5]):
                    plt.subplot(3, 5, i+1+5)
                    plt.plot(np.arange(c.shape[0]), c[:, 0], 'g-', label='x') 
                    plt.plot(np.arange(c.shape[0]), c[:, 1], 'r-', label='y') 
                    plt.plot(np.arange(c.shape[0]), c[:, 2], 'b-', label='z') 
                for i,c in enumerate(test_data[-1][:5]):
                    plt.subplot(3, 5, i+1+10)
                    plt.plot(np.arange(c.shape[0]), c[:, 0], 'g-', label='x') 
                    plt.plot(np.arange(c.shape[0]), c[:, 1], 'r-', label='y') 
                    plt.plot(np.arange(c.shape[0]), c[:, 2], 'b-', label='z') 
                plt.tight_layout()
                plt.show() 

    elif method == 'window_based_svm':

        # 70% train, 30% test
        for user in np.unique(label_user):

            idx = np.where(label_user == user)
            data_temp = data[idx]
            user_temp = label_user[idx]
            id_temp = id_window[idx]

            # number of window for user and session, in train, val and test
            samples = data_temp.shape[0]
            train_percentage = int(samples*0.7)
            if train_percentage == samples:
                train_percentage -= 1

            # shuffle for random pick
            data_temp, user_temp, id_temp = skutils.shuffle(data_temp, user_temp, id_temp)

            # train
            train = data_temp[:train_percentage]
            user_train = user_temp[:train_percentage]
            id_train = id_temp[:train_percentage]

            # test
            test = data_temp[train_percentage:]
            user_test = user_temp[train_percentage:]
            id_test = id_temp[train_percentage:]

            train_data.append(train)
            train_label.extend(user_train)
            test_data.append(test)
            test_label.extend(user_test)

    train_data = np.concatenate(train_data, axis=0)
    if val_data != []:
        val_data = np.concatenate(val_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)
    train_label = np.asarray(train_label)
    if val_data != []:
        val_label = np.asarray(val_label)
    test_label = np.asarray(test_label)

    train_data, train_label = skutils.shuffle(train_data, train_label)
    if val_data != []:
        val_data, val_label = skutils.shuffle(val_data, val_label)
    test_data, test_label = skutils.shuffle(test_data, test_label)

    return train_data, val_data, test_data, train_label, val_label, test_label

def normalize_data(train, val, test=None, return_mean_std=False):

    mean = np.mean(np.reshape(train, [-1, train.shape[2]]), axis=0)
    std = np.std(np.reshape(train, [-1, train.shape[2]]), axis=0)

    train = (train - mean)/std
    val = (val - mean)/std

    train = np.expand_dims(train, 3)
    val = np.expand_dims(val,  3)

    if test is not None:
        test = (test - mean)/std
        test = np.expand_dims(test, 3)
        
    if not return_mean_std:
        return train, val, test
    else:
        return train, val, test, mean, std

############################################
### From paper Data Augmentation for gait ##
############################################


def scale(x, out_range):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def denoiseData(signal, plot=False):
    denoise = np.empty_like(signal)
    original_shape = signal.shape[0]
    for dim in np.arange(signal.shape[1]):
        original_extent = tuple(slice(s) for s in signal[:,dim].shape)
        coeffs = pywt.wavedec(signal[:,dim], wavelet='db6', level=2)
        coeffs[-1] == np.zeros_like(coeffs[-1])
        coeffs[-2] == np.zeros_like(coeffs[-2])
        denoise[:,dim] = pywt.waverec(coeffs, 'db6')[original_extent]
    if plot:
        plt.figure(figsize=(12, 3))
        plt.style.use('seaborn-darkgrid')
        plt.subplot(3, 2, 1)
        plt.title(f'noise')
        plt.plot(np.arange(original_shape), signal[:, 0], 'b-', label='x')     
        plt.legend(loc='upper right')       
        plt.subplot(3, 2, 3)
        plt.plot(np.arange(original_shape), signal[:, 1], 'r-', label='y')   
        plt.legend(loc='upper right')         
        plt.subplot(3, 2, 5)
        plt.plot(np.arange(original_shape), signal[:, 2], 'g-', label='z')
        plt.legend(loc='upper right')
        plt.subplot(3, 2, 2)
        plt.title(f'denoise')
        plt.plot(np.arange(original_shape), denoise[:, 0], 'b-', label='x')         
        plt.legend(loc='upper right')   
        plt.subplot(3, 2, 4)
        plt.plot(np.arange(original_shape), denoise[:, 1], 'r-', label='y')    
        plt.legend(loc='upper right')       
        plt.subplot(3, 2, 6)
        plt.plot(np.arange(original_shape), denoise[:, 2], 'g-', label='z')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    return denoise


def calAutoCorrelation(data):
    
    n = len(data)
    autocorrelation_coeff = np.zeros((n,3))

    for i in range(3):
        autocorrelation_coeff[0,i] = np.sum(data[:,i]**2)/n

    for i in range(3):
        for t in range(1, n):
            for j in range(1, n-t):
                autocorrelation_coeff[t,i] = autocorrelation_coeff[t,i] + \
                    data[j,i]*data[j+t,i]
            autocorrelation_coeff[t,i] = autocorrelation_coeff[t,i] / (n-t)
        autocorrelation_coeff[:,i] = autocorrelation_coeff[:,i]/autocorrelation_coeff[0,i]

    return np.mean(autocorrelation_coeff, axis=1)


def find_peaks(peaks, data, gcLen, autocorr):
    
    # find first possible peak to start search
    first_peak = [peaks[0]]
    for i,_ in enumerate(peaks[1:]):
        if abs(peaks[i] - peaks[0]) <= 0.8*gcLen:
            first_peak.append(peaks[i])
        else:
            break

    first_peak = peaks.index(first_peak[np.argmin(data[first_peak])])

    # splice peaks from first possible detected peak
    peaks = peaks[first_peak:]

    # neighbour search of minimum at given gcLen from the first peak 
    peak_filtered = [peaks[0]]
    i = 0
    while i < len(peaks[:-1]):
        peak_cluster = []
        j = 1
        while i + j < len(peaks):
            if abs(peaks[i] - peaks[i + j]) > 1.1*gcLen:
                break
            if abs(peaks[i] - peaks[i + j]) >= 0.5*gcLen and abs(peaks[i] - peaks[i + j]) <= 1.1*gcLen:
                peak_cluster.append(peaks[i + j])
            j += 1
        if i + j >= len(peaks) and peak_cluster == []:
            break
        if peak_cluster == []:
            j = 1
            while i + j < len(peaks):
                if abs(peaks[i] - peaks[i + j]) > 1.6*gcLen:
                    break
                if abs(peaks[i] - peaks[i + j]) >= 0.5*gcLen and abs(peaks[i] - peaks[i + j]) <= 1.6*gcLen:
                    peak_cluster.append(peaks[i + j])
                j += 1
        if peak_cluster == []:
            j = 1
            while i + j < len(peaks):
                if abs(peaks[i] - peaks[i + j]) > 2.5*gcLen:
                    break
                if abs(peaks[i] - peaks[i + j]) >= 0.5*gcLen and abs(peaks[i] - peaks[i + j]) <= 2.5*gcLen:
                    peak_cluster.append(peaks[i + j])
                j += 1
        if peak_cluster == []:
            break

        index_min = np.argmin(data[peak_cluster])
        min_peak = peak_cluster[index_min]

        # from min peak found peak on the right if they are at max 0.1*gcLen
        '''
        for peak in peak_cluster[peak_cluster.index(min_peak):]:
            if abs(peak - min_peak) <= 0.05*gcLen:
                min_peak = peak
        '''
        peak_filtered.append(min_peak)
        i = peaks.index(min_peak)

    # check on first-second peak distance, and least two peaks distance
    if abs(peak_filtered[0] - peak_filtered[1]) > 1.2*gcLen:
        peak_filtered = peak_filtered[1:]
    if abs(peak_filtered[-1] - peak_filtered[-2]) > 1.2*gcLen:
        peak_filtered = peak_filtered[:-1]

    if len(peak_filtered) < 5 or len(peak_filtered) > 15:
        to_plot = True
    else:
        to_plot = False
    
    return peak_filtered, to_plot

def find_gcLen(data):

    # compute autcorrelation to estimate len cycle
    auto_corr_coeff = calAutoCorrelation(data)

    # smooth the auto_correlation_coefficient
    for i in range(10):
        auto_corr_coeff = smooth(auto_corr_coeff)

    # approximate the length of a gait cycle by selecting the 2nd peak (positive) in the auto correlation signal
    peak_auto_corr = []
    gcLen = 0
    flag = 0
    mean_auto_corr = np.mean(auto_corr_coeff[:200])
    for i in range(1, 200):
        if auto_corr_coeff[i] > auto_corr_coeff[i-1] and \
           auto_corr_coeff[i] > auto_corr_coeff[i+1] and \
           auto_corr_coeff[i] > mean_auto_corr:
            flag += 1
            peak_auto_corr.append(i)
            if flag == 2:
                gcLen = i - 1
                break
    
    if gcLen < 10:
        peak_auto_corr = []
        flag = 0
        for i in range(1, len(auto_corr_coeff)-1):
            if auto_corr_coeff[i] > auto_corr_coeff[i-1] and \
            auto_corr_coeff[i] > auto_corr_coeff[i+1]:
                if flag == 0 or (flag == 1 and i > 10):
                    flag += 1
                    peak_auto_corr.append(i)
                    if flag == 2:
                        gcLen = i - 1
                        break        

    return gcLen, auto_corr_coeff, peak_auto_corr


def find_thresh_peak(data):

    plot = False

    # all peaks
    all_peak_pos = []
    for i in range(0, data.shape[0]-1):
        if i==0 and data[i] <= data[i+1]:
            all_peak_pos.append(i)
        if data[i] <= data[i-1] and data[i] <= data[i+1]:
            all_peak_pos.append(i)

    # filter list of peaks based on mean and standard deviation of detected peaks
    _mean = np.mean(data)
    _std = np.std(data)
    filter_peaks_pos = []
    for peak in all_peak_pos:
        if(data[peak] < _mean - 0.6*_std):
            filter_peaks_pos.append(peak)

        
    if plot:
        plt.figure(figsize=(12, 3))
        plt.style.use('seaborn-darkgrid')
        plt.plot(np.arange(len(data)), data, 'b-')
        plt.scatter(all_peak_pos, data[all_peak_pos], c='red')
        plt.scatter(filter_peaks_pos, data[filter_peaks_pos], c='black')
        plt.tight_layout()
        plt.show()

    return filter_peaks_pos

def remove_g_component(signal, sampling_rate, plot):

    # get gravity component g(t)
    sos = butter_lowpass(cutoff=0.3, nyq_freq=sampling_rate*0.5, order=3, sampling_rate=sampling_rate)
    g = butter_lowpass_filter(signal, sos)

    # get linear acceleration s(t) = s(t) - g(t)
    no_g = signal - g

    if plot:
        plt.figure(figsize=(12, 3))
        plt.style.use('seaborn-darkgrid')
        plt.subplot(2,1,1)
        plt.title('With gravity component')
        plt.plot(np.arange(signal.shape[0]), signal[:,0], 'g-', label='x')
        plt.plot(np.arange(signal.shape[0]), signal[:,1], 'r-', label='y')
        plt.plot(np.arange(signal.shape[0]), signal[:,2], 'b-', label='z')
        plt.legend(loc='upper right')
        plt.subplot(2,1,2)
        plt.title('No gravity component')
        plt.plot(np.arange(no_g.shape[0]), no_g[:,0], 'g-', label='x')
        plt.plot(np.arange(no_g.shape[0]), no_g[:,1], 'r-', label='y')
        plt.plot(np.arange(no_g.shape[0]), no_g[:,2], 'b-', label='z')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()       

    return no_g

def butter_lowpass(cutoff, nyq_freq, order, sampling_rate):
    normal_cutoff = float(cutoff) / nyq_freq
    sos = signal.butter(order, normal_cutoff, btype='lowpass', output='sos', fs=sampling_rate)
    return sos

def butter_lowpass_filter(data, sos):
    y = signal.sosfiltfilt(sos, data, axis=0, padtype=None)
    return y

def interpolated(cycles, to_interp, plot_interpolated):
    cycles_interpolated = []
    for cycle in cycles:
        interpolated = np.empty((to_interp, cycle.shape[1]))
        for dim in np.arange(cycle.shape[1]):
            '''
            interpolated[:, dim] = CubicSpline(np.arange(0, cycle.shape[0]), cycle[:, dim])(
                np.linspace(0, cycle.shape[0]-1, to_interp))
            '''
            interpolated[:, dim] = interp1d(np.arange(0, cycle.shape[0]), cycle[:, dim])(np.linspace(0, cycle.shape[0]-1, to_interp))
        if plot_interpolated:
            plt.figure(figsize=(12, 3))
            plt.style.use('seaborn-darkgrid')
            plt.subplot(1, 2, 1)
            plt.title(f'original')
            plt.plot(np.arange(cycle.shape[0]),
                     cycle[:, 0], 'b-', label='noise')
            plt.plot(np.arange(cycle.shape[0]),
                     cycle[:, 1], 'r-', label='noise')
            plt.plot(np.arange(cycle.shape[0]),
                     cycle[:, 2], 'g-', label='noise')
            plt.subplot(1, 2, 2)
            plt.title(f'interpolated')
            plt.plot(
                np.arange(interpolated.shape[0]), interpolated[:, 0], 'b-', label='denoise')
            plt.plot(
                np.arange(interpolated.shape[0]), interpolated[:, 1], 'r-', label='denoise')
            plt.plot(
                np.arange(interpolated.shape[0]), interpolated[:, 2], 'g-', label='denoise')
            plt.tight_layout()
            plt.show()
        cycles_interpolated.append(interpolated)
    return cycles_interpolated