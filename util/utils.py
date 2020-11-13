import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pywt
from pprint import pprint
from sklearn import utils as skutils
from sklearn.model_selection import train_test_split
import math


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


def detectGaitCycle(data, plot_peak=False, plot_auto_corr_coeff=False, use_2_step=False):

    data = data[:, 2]  # z axis
    t = 0.4

    #peaks = find_thresh_peak(data, t)

    gcLen, auto_corr_coeff, peak_auto_corr = find_gcLen(data, use_2_step)

    ############################################
    ### From paper Biometric Walk Recognizer ###
    ############################################

    step_equilibrium, step_threshold = find_equilibrium_threshold(data, gcLen)

    peaks_steps = find_peaks_steps(data, step_equilibrium, step_threshold)
    '''
    plt.figure(figsize=(12, 3))
    plt.style.use('seaborn-darkgrid')
    plt.plot(np.arange(data.shape[0]), data, 'b-')
    plt.hlines([step_equilibrium, step_threshold], xmin=0, xmax=data.shape[0], color=['b', 'r'], ls='--')
    plt.scatter(x=peaks_steps, y=data[peaks_steps], c='r', marker='*')
    plt.tight_layout()
    plt.show()
    '''
    peaks = peaks_steps

    ############################################
    ################## End #####################
    ############################################

    # plot coefficients autocorrelation and 2nd peak used to estimated gait length
    if plot_auto_corr_coeff:
        plt.figure(figsize=(12, 3))
        plt.style.use('seaborn-darkgrid')
        plt.plot(np.arange(len(auto_corr_coeff)), auto_corr_coeff, 'b-')
        plt.scatter(peak_auto_corr, auto_corr_coeff[peak_auto_corr], c='red')
        plt.tight_layout()
        plt.show()

    #peaks, to_plot = find_peaks(peaks, data, gcLen, use_2_step)

    if plot_peak:  # or to_plot:
        plt.figure(figsize=(12, 3))
        plt.style.use('seaborn-darkgrid')
        plt.plot(np.arange(data.shape[0]), data, 'b-')
        plt.vlines(peaks, ymin=min(data)*0.95,
                   ymax=max(data)*0.95, color='r', ls='--')
        plt.tight_layout()
        plt.show()

    return peaks


def segment2GaitCycle(peaks, segment):
    cycles = []
    for i in range(0, len(peaks)-1):
        cycle = segment[peaks[i]:peaks[i+1], :]
        cycles.append(cycle[np.newaxis, :, :])
    return cycles


def split_data_train_val_test_gait(data, 
                                   label_user, 
                                   label_sequences, 
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
    sequences_for_user = []
    id_for_user = []

    for user in np.unique(label_user):
        idx_user = np.where(label_user == user)
        data_for_user.append(data[idx_user])
        label_for_user.append(label_user[idx_user])
        sequences_for_user.append(label_sequences[idx_user])
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

            if gait_2_cycles:
                train_gait = int(samples*0.7)

            if split == 'standard':
                train_gait = math.ceil(
                    samples*0.7) if ((samples*0.7) % 1) >= 0.5 else round(samples*0.7)
                val_gait = math.ceil(
                    samples*0.2) if ((samples*0.2) % 1) >= 0.5 else round(samples*0.2)
            elif split == 'paper':
                if samples < 10:
                    if samples == 9:
                        train_gait = 7
                    elif samples == 8:
                        train_gait = 8
                    elif samples == 7:
                        train_gait = 5

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


def find_peaks(peaks, data, gcLen, use_2_step=False):

    if use_2_step:
        alpha = 1/4  # 1/4
        beta = 0.7  # 0.7
        gamma = 2/6  # 2/6
    else:
        alpha = 0.5  # 1/4
        beta = 3/4
        gamma = 1/6

    plot_peak = False
    peaks_copy = peaks.copy()

    # find first candidate peak
    i = 1
    while i < len(peaks)-1:
        if peaks[i] - peaks[i-1] < 0.5*gcLen and data[peaks[i]] < data[peaks[i-1]]:
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
    if not use_2_step:
        i = 1
        j = 0
        peak_pos_modified = peaks[:]
        while i < len(peaks):
            if peaks[i] - peaks[i-1] > 1.1*gcLen:
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

    # filter last two peaks
    filtered = False
    if peaks[-1]-peaks[-2] < 0.3*gcLen:
        if peaks[-1] < peaks[-2]:
            peaks.remove(peaks[-2])
        else:
            peaks.remove(peaks[-1])
        filtered = True
    elif peaks[-1]-peaks[-2] < 0.5*gcLen and not filtered:
        peaks.remove(peaks[-1])
    elif peaks[-1]-peaks[-2] > 1.3*gcLen and not filtered:
        peaks.remove(peaks[-1])

    # check if there a minimum next to detected peaks, take it if it's lower
    if not use_2_step:
        for i, peak in enumerate(peaks):
            idx = np.where(data[peak-int(0.2*gcLen):peak +
                                int(0.2*gcLen)] < data[peak])
            idx = idx[0] + peak-int(0.2*gcLen)
            if len(idx) > 0:
                peaks[i] = idx[np.argmin(data[idx])]

    if len(peaks) <= 4:
        plot_peak = True

    return peaks, plot_peak


def find_gcLen(data, use_2_step):

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
    std_auto_corr = np.std(auto_corr_coeff[:200])
    for i in range(1, 200):
        if auto_corr_coeff[i] > auto_corr_coeff[i-1] and auto_corr_coeff[i] > auto_corr_coeff[i+1] and auto_corr_coeff[i] > (mean_auto_corr + std_auto_corr*0.4):
            flag += 1
            peak_auto_corr.append(i)
            if flag == 3 and use_2_step:
                gcLen = i - 1
                break
            if flag == 2 and not use_2_step:
                gcLen = i - 1
                break

    return gcLen, auto_corr_coeff, peak_auto_corr


def find_thresh_peak(data, t):

    # all peaks
    all_peak_pos = []
    for i in range(1, data.shape[0]-1):
        if(data[i] < data[i-1] and (data[i] < data[i+1] or data[i] == data[i+1])):
            all_peak_pos.append(i)

    # filter list of peaks based on mean and standard deviation of detected peaks
    _mean = np.mean(data)
    _std = np.mean(data)
    filter_peaks_pos = []
    for peak in all_peak_pos:
        if(data[peak] < _mean - 0.5*_std):
            filter_peaks_pos.append(peak)

    return filter_peaks_pos

############################################
### From paper Biometric Walk Recognizer ###
############################################


def find_equilibrium_threshold(data, gcLen):

    # step equilibrium
    temp_data = np.around(data[np.where(data < np.mean(data))], decimals=2)
    value, counts = np.unique(temp_data, return_counts=True)
    step_equilibrium = value[np.argmax(counts)]

    # step threshold
    maxima = []
    for i in range(1, data.shape[0]-1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            maxima.append(data[i])
    k = int((data.shape[0] / 50)*1.5)
    step_threshold = np.around(sorted(maxima, reverse=True)[k], decimals=2)
    step_threshold = 0.9 * step_threshold

    if step_threshold < step_equilibrium:
        return step_threshold, step_equilibrium

    return step_equilibrium, step_threshold


def find_peaks_steps(data, step_equilibrium, step_threshold):

    peaks_steps = []

    maxima = []
    for i in range(1, data.shape[0]-1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            maxima.append(i)

    start = 0
    while True:
        # find first peak greater than step threshold
        if start == 0:
            idx_step = min(
                list(set(np.where(data[start:] > 0.9*step_threshold)[0]) & set(maxima)))
            peaks_steps.append(idx_step)
            start = idx_step
        # find first value lower than step equilibrium
        idx = np.where(data[start:] < 0.9*step_equilibrium)[0]
        if idx.shape[0] == 0:
            break
        else:
            start = idx[0] + start
        # find next maximum greater than stap threshold
        idx = np.where(data[start:] > step_threshold)[0]
        if idx.shape[0] == 0:
            break
        try:
            idx_step = min(list(set(idx+start) & set(maxima)))
        except:
            break
        peaks_steps.append(idx_step)
        start = idx_step

    _mean = int(np.sum([peaks_steps[i] - peaks_steps[i-1]
                        for i in range(1, len(peaks_steps))]) / (len(peaks_steps)-1))

    # refine peaks based on value in neighboor
    for i in range(0, len(peaks_steps)):
        idx_r = np.where(
            data[peaks_steps[i]:peaks_steps[i]+int(0.3*_mean)] > data[peaks_steps[i]])[0]
        if idx_r.shape[0] > 0:
            max_neigh = np.argmax(
                data[peaks_steps[i]:peaks_steps[i]+int(0.3*_mean)])
            peaks_steps[i] = max_neigh + peaks_steps[i]

    # filter based on average len
    j = 0
    peaks = peaks_steps.copy()
    for i in range(1, len(peaks_steps)):
        if peaks_steps[i] - peaks_steps[i-1] < 0.7*_mean:
            if data[peaks_steps[i]] > data[peaks_steps[i-1]]:
                try:
                    peaks.remove(peaks_steps[i-1])
                except:
                    pass
            else:
                try:
                    peaks.remove(peaks_steps[i])
                except:
                    pass
        elif peaks_steps[i] - peaks_steps[i-1] > 1.7*_mean:
            temp = int(
                (peaks_steps[i] - peaks_steps[i-1])/2) + peaks_steps[i-1]
            most_close = sorted(
                np.arange(peaks_steps[i-1], peaks_steps[i]), key=lambda x: abs(x-temp))
            most_close_before = list(
                filter(lambda x: x <= temp + 0.20*_mean and x >= temp - 0.20*_mean, most_close))
            if most_close_before != []:
                idx = np.argmax(data[most_close_before])
                peaks[i+j:i+j] = [most_close_before[idx]]
                j += 1

    peaks = sorted(peaks)

    return peaks
