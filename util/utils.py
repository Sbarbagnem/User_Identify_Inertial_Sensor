import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pywt
from pprint import pprint
from sklearn import utils as skutils
from sklearn.model_selection import train_test_split
import math

from scipy.signal import find_peaks as find_peaks_scipy
from sklearn.preprocessing import scale as scale_sklearn


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
    if 'unimib' in dataset_name:
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
    autocorr = False if gcLen != None else True

    # plot data
    if False:
        plt.figure(figsize=(12, 3))
        plt.style.use('seaborn-darkgrid')
        plt.plot(np.arange(data.shape[0]), data[:,0], 'b-')
        plt.plot(np.arange(data.shape[0]), data[:,1], 'r-')
        plt.plot(np.arange(data.shape[0]), data[:,2], 'y-')
        plt.tight_layout()
        plt.show()

    t = 0.4

    peaks = find_thresh_peak(selected_data, t)

    # compute gcLen based on autocorrelation of signal if not given by default
    if autocorr:
        gcLen, auto_corr_coeff, peak_auto_corr = find_gcLen(selected_data)
        if plot_auto_corr_coeff:
            plt.figure(figsize=(12, 3))
            plt.style.use('seaborn-darkgrid')
            plt.plot(np.arange(len(auto_corr_coeff)), auto_corr_coeff, 'b-')
            plt.scatter(peak_auto_corr, auto_corr_coeff[peak_auto_corr], c='red')
            plt.tight_layout()
            plt.show()

    peaks, to_plot = find_peaks(peaks, selected_data, gcLen, autocorr)

    selected_data = scale_sklearn(selected_data, axis=0, with_mean=True, with_std=True)
    peaks_scipy, _ = find_peaks_scipy(np.negative(selected_data), height=np.mean(np.negative(selected_data)) + 0.5*np.std(np.negative(selected_data)), distance=gcLen*0.7)

    if plot_peak:# or to_plot: 
        plt.figure(figsize=(12, 3))

        plt.subplot(3,1,1)
        plt.style.use('seaborn-darkgrid')
        plt.plot(np.arange(data.shape[0]), data[:,0], 'g-', label='x')
        plt.plot(np.arange(data.shape[0]), data[:,1], 'r-', label='y')
        plt.plot(np.arange(data.shape[0]), data[:,2], 'b-', label='z')
        plt.legend(loc='upper right')

        plt.subplot(3,1,2)
        plt.style.use('seaborn-darkgrid')
        plt.plot(np.arange(data.shape[0]), data[:,2], 'b-', label='z')
        plt.vlines(peaks, ymin=min(data[:,2]), ymax=max(data[:,2]), color='black', ls='dotted')
        plt.legend(loc='upper right')

        plt.subplot(3,1,3)
        plt.style.use('seaborn-darkgrid')
        plt.plot(np.arange(selected_data.shape[0]), selected_data, 'b-', label='z')
        plt.vlines(peaks_scipy, ymin=min(selected_data), ymax=max(selected_data), color='red', ls='--')
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    return peaks_scipy


def segment2GaitCycle(peaks, segment):
    cycles = []
    for i in range(0, len(peaks)-1):
        cycle = segment[peaks[i]:peaks[i+1]+1]
        cycles.append(cycle[np.newaxis, :, :])
    return cycles


def split_data_train_val_test_gait(data,
                                   label_user,
                                   id_window,
                                   train_gait=8,
                                   val_test=0.5,
                                   gait_2_cycles=False,
                                   method='cycle_based',
                                   plot=False,
                                   overlap=None,
                                   split='standard'):

    data_for_user = []
    label_for_user = []
    id_for_user = []

    for user in np.unique(label_user):
        idx_user = np.where(label_user == user)
        data_for_user.append(data[idx_user])
        label_for_user.append(label_user[idx_user])
        if method == 'window_based':
            id_for_user.append(id_window[idx_user])

    train_data = []
    val_data = []
    test_data = []
    train_label = []
    val_label = []
    test_label = []

    if method == 'cycle_based':

        for cycles, label in zip(data_for_user, label_for_user):

            # to take random gait cycle from user
            cycles = skutils.shuffle(cycles)
            samples = cycles.shape[0]

            if samples <= 4:
                print(f'There only {samples} cycles for user {label[0]}')

            if gait_2_cycles:
                train_gait = int(samples*0.7)

            if split == 'standard':
                train_gait = int(samples*0.7)
                val_gait = int(samples*0.2)
                if samples < 10 and samples >= 4:
                    train_gait = samples - 2    
                    val_gait = int((samples - train_gait)/2)        
                   

            elif split == 'paper':
                if samples < 10:
                    train_gait = samples - 2

                val_gait = int((samples - train_gait)/2)

            # train
            train_data.append(cycles[:train_gait])
            train_label.extend(label[:train_gait])

            # val
            val_data.append(
                cycles[train_gait:val_gait+train_gait])
            val_label.extend(
                label[train_gait:val_gait+train_gait])

            # test
            test_data.append(cycles[val_gait+train_gait:])
            test_label.extend(label[val_gait+train_gait:])

    elif method == 'window_based':
        if overlap == None:
            raise Exception('Overlap must not be empty for window base method')
        # 70% train, 20% val, 10% test
        for cycles, labels, ID in zip(data_for_user, label_for_user, id_for_user):

            samples = cycles.shape[0]

            # take 90% of data for train
            train_percentage = int(samples*0.9)
            train = [cycles[:train_percentage],
                     labels[:train_percentage],
                     ID[:train_percentage]]

            # take 10% of data for test
            test_percentage = samples - train_percentage
            test = [cycles[train_percentage:train_percentage+test_percentage],
                    labels[train_percentage:train_percentage+test_percentage],
                    ID[train_percentage:train_percentage+test_percentage]]

            # delete overlap between train and test
            if overlap == 50:
                distances_to_delete = [1]
            elif overlap == 75:
                distances_to_delete = [1, 2, 3]
            overlap_idx = delete_overlap(
                train[2], test[2], distances_to_delete)
            train[0] = np.delete(train[0], overlap_idx, axis=0)
            train[1] = np.delete(train[1], overlap_idx, axis=0)

            # split train in train and val (78%, 22%)
            x_train, x_val, y_train, y_val = train_test_split(
                train[0], train[1], test_size=0.22)

            # train
            train_data.append(x_train)
            train_label.extend(y_train)

            # val
            val_data.append(x_val)
            val_label.extend(y_val)

            # test
            test_data.append(test[0])
            test_label.extend(test[1])

    train_data = np.concatenate(train_data, axis=0)
    val_data = np.concatenate(val_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)
    train_label = np.asarray(train_label)
    val_label = np.asarray(val_label)
    test_label = np.asarray(test_label)

    return train_data, val_data, test_data, train_label, val_label, test_label,


def normalize_data(train, val, test):

    mean = np.mean(np.reshape(train, [-1, train.shape[2]]), axis=0)
    std = np.std(np.reshape(train, [-1, train.shape[2]]), axis=0)

    train = (train - mean)/std
    val = (val - mean)/std
    test = (test - mean)/std

    train = np.expand_dims(train, 3)
    val = np.expand_dims(val,  3)
    test = np.expand_dims(test, 3)

    return train, val, test

############################################
### From paper Data Augmentation for gait ##
############################################


def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def denoiseData(data, plot=False):
    for i, x in enumerate(data):
        for dim in np.arange(x.shape[1]):
            # decompostion
            d1 = pywt.downcoef('d', x[:, dim], 'db6', level=1)
            a1 = pywt.downcoef('a', x[:, dim], 'db6', level=1)
            d2 = pywt.downcoef('d', x[:, dim], 'db6', level=2)
            a2 = pywt.downcoef('a', x[:, dim], 'db6', level=2)
            # set deatails coef to 0
            d1 = np.zeros_like(d1)
            d2 = np.zeros_like(d2)
            # recostruction
            temp = pywt.upcoef('a', a1, 'db6', level=1, take=x.shape[0]) + \
                pywt.upcoef('d', d1, 'db6', level=1, take=x.shape[0]) + \
                pywt.upcoef('a', a2, 'db6', level=2, take=x.shape[0]) + \
                pywt.upcoef('d', d2, 'db6', level=2, take=x.shape[0])
            # rescale in initial range
            temp = scale(temp, out_range=(min(x[:, dim]), max(x[:, dim])))
            if plot:
                plt.figure(figsize=(12, 3))
                plt.style.use('seaborn-darkgrid')
                plt.subplot(1, 2, 1)
                plt.title(f'noise')
                plt.plot(np.arange(x.shape[0]), x[:, dim], 'b-', label='noise')
                plt.subplot(1, 2, 2)
                plt.title(f'denoise')
                plt.plot(np.arange(temp.shape[0]), temp, 'b-', label='denoise')
                plt.tight_layout()
                plt.show()
            data[i][:, dim] = temp[:x.shape[0]]
    return data


def calAutoCorrelation(data):
    n = len(data)
    autocorrelation_coeff = np.zeros(n)
    autocorrelation_coeff[0] = np.sum(data[:]**2)/n

    for t in range(1, n):
        for j in range(1, n-t):
            autocorrelation_coeff[t] = autocorrelation_coeff[t] + \
                data[j]*data[j+t]
        autocorrelation_coeff[t] = autocorrelation_coeff[t] / (n-t)
    autocorrelation_coeff = autocorrelation_coeff/np.max(autocorrelation_coeff)
    return autocorrelation_coeff


def find_peaks(peaks, data, gcLen, autocorr):

    alpha = 0.25  # 0.25
    beta = 0.75   # 0.75
    gamma = 0.2  # 0.16

    plot_peak = False
    peaks_copy = peaks.copy()

    # find first candidate peak
    i = 1
    while i < len(peaks)-1:
        if peaks[i] - peaks[i-1] < 0.2*gcLen and data[peaks[i]] < data[peaks[i-1]]:
            peaks.remove(peaks[i-1])
        else:
            break

    # find all candidate peak
    i = 1
    while i < len(peaks)-1:
        if peaks[i]-peaks[i-1] < alpha*gcLen:
            if data[peaks[i]] <= data[peaks[i-1]]:
                peaks.remove(peaks[i-1])
                continue
            else:
                peaks.remove(peaks[i])
                continue
        elif peaks[i]-peaks[i-1] < beta*gcLen:
            if peaks[i+1] - peaks[i] < gamma*gcLen:
                if data[peaks[i+1]] <= data[peaks[i]]:
                    peaks.remove(peaks[i])
                    continue
                else:
                    peaks.remove(peaks[i+1])
                    continue
            else:
                peaks.remove(peaks[i])
                continue
        else:
            i += 1

    # if there is a gait grater then gcLen*1.5 must be probabily divided in two gait
    if autocorr:
        i = 1
        j = 0
        peak_pos_modified = peaks[:]
        while i < len(peaks):
            if peaks[i] - peaks[i-1] > 1.2*gcLen:
                temp = int((peaks[i] - peaks[i-1])/2) + peaks[i-1]
                most_close = sorted(peaks_copy, key=lambda x: abs(x-temp))
                try:
                    most_close_before = list(
                        filter(lambda x: x <= temp + 0.20*gcLen and x >= temp - 0.20*gcLen, most_close))
                    if most_close_before != []:
                        idx = np.argmin(data[most_close_before])
                        peak_pos_modified[i+j:i+j] = [most_close_before[idx]]
                        j += 1
                except:
                    plot_peak = True
            i += 1

        peaks = sorted(peak_pos_modified)
    
    if peaks[-1]-peaks[-2] < 0.5*gcLen:
        peaks.remove(peaks[-1])
        
    # check if there a minimum next to detected peaks, take it if it's lower
    thresh = 0.4 if autocorr else 0.2
    for i, peak in enumerate(peaks):
        idx = np.where(data[peak-int(thresh*gcLen):peak +
                            int(thresh*gcLen)] < data[peak])
        idx = idx[0] + peak-int(thresh*gcLen)
        if len(idx) > 0:
            peaks[i] = idx[np.argmin(data[idx])]

    # filter last two peaks
    if peaks[-1]-peaks[-2] < 0.3*gcLen:
        if peaks[-1] < peaks[-2]:
            peaks.remove(peaks[-2])
        else:
            peaks.remove(peaks[-1])
    elif peaks[-1]-peaks[-2] > 1.5*gcLen:
        peaks.remove(peaks[-1])

    if len(peaks) <= 4 or len(peaks)>30:
        plot_peak = True

    return sorted(peaks), plot_peak


def find_gcLen(data):

    # compute autcorrelation to estimate len cycle
    auto_corr_coeff = calAutoCorrelation(data)

    # smooth the auto_correlation_coefficient
    for i in range(7):
        auto_corr_coeff = smooth(auto_corr_coeff)

    # approximate the length of a gait cycle by selecting the 2nd peak (positive) in the auto correlation signal
    peak_auto_corr = []
    gcLen = 0
    flag = 0
    #mean_auto_corr = np.mean(auto_corr_coeff)
    #std_auto_corr = np.std(auto_corr_coeff)
    for i in range(1, 200):
        if auto_corr_coeff[i] > auto_corr_coeff[i-1] and \
           auto_corr_coeff[i] > auto_corr_coeff[i+1]: #and \
           #auto_corr_coeff[i] > (mean_auto_corr + std_auto_corr*0.4):
            flag += 1
            peak_auto_corr.append(i)
            if flag == 2:
                gcLen = i - 1
                break

    return gcLen, auto_corr_coeff, peak_auto_corr


def find_thresh_peak(data, t):

    plot = False

    # all peaks
    all_peak_pos = []
    for i in range(1, data.shape[0]-1):
        if(data[i] <= data[i-1] and data[i] <= data[i+1]):
            all_peak_pos.append(i)

    # filter list of peaks based on mean and standard deviation of detected peaks
    _mean = np.mean(data[all_peak_pos])
    _std = np.std(data[all_peak_pos])
    threshold = _mean - t*_std
    #threshold = _mean
    filter_peaks_pos = []
    for peak in all_peak_pos:
        if(data[peak] < threshold):
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