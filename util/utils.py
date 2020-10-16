import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pywt
from pprint import pprint
from sklearn import utils as skutils


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
        indexes[str[i]] = []

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
                                indexes[str(folder)].append(temp_index_label_act[0])
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
                            indexes[str(folder)].append(temp_index_label_act[0])
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
        overlap_ID = np.concatenate((overlap_ID, val_id+distance, val_id-distance))             

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
    plt.yticks(np.arange(0,1.1,0.1))
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
            f=open(path_to_save + 'performance_based_act.txt', 'a+')
        else:
            f= open(path_to_save + 'performance_based_act.txt', 'w+')
        for l,p in zip(label_act, correct):
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
        f=open(path_to_save + 'mean_performance.txt', 'a+')
    else:
        f= open(path_to_save + 'mean_performance.txt', 'w+')

    for key in list(performances.keys()):
        f.write(f"{key}: {performances[key]}\r\n")
    f.close()

def denoiseData(data, plot=False):
    for i,x in enumerate(data):       
        if x.shape[0] % 2 != 0:
            x = x[1:,:]           
        n = x.shape[0]   
        for dim in np.arange(x.shape[1]):
            coeffs = pywt.wavedec(x[:,dim], 'db6', level=2)
            # set high coef to 0
            coeffs[-1] = np.zeros_like(coeffs[-1])
            coeffs[-2] = np.zeros_like(coeffs[-2])
            temp = pywt.waverec(coeffs, 'db6')
            if plot:
                plt.figure(figsize=(12, 3))
                plt.style.use('seaborn-darkgrid')
                plt.subplot(1, 2, 1)
                plt.title(f'noise')
                plt.plot(np.arange(n), x[:,dim], 'b-', label='noise')
                plt.subplot(1, 2, 2)
                plt.title(f'denoise')
                plt.plot(np.arange(temp.shape[0]), temp, 'b-', label='denoise')
                plt.tight_layout()
                plt.show()
            x[:,dim] = temp
        data[i] = x
    return data

def calAutoCorrelation(data):
    n = len(data)
    autocorrelation_coeff = np.zeros(n)

    autocorrelation_coeff[0]= np.sum(data[:]**2)/n
    for t in range(1,n-1):
        for j in range(1,n-t):
            autocorrelation_coeff[t]= autocorrelation_coeff[t] + data[j]*data[j+t]
        autocorrelation_coeff[t]= autocorrelation_coeff[t] /(n-t)
    
    autocorrelation_coeff = autocorrelation_coeff/np.max(autocorrelation_coeff)
    return autocorrelation_coeff

def smooth(coef):
    window_len = 5
    s=np.r_[coef[window_len-1:0:-1],coef,coef[-1:-window_len:-1]]
    w=np.ones(window_len,'d')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def detectGaitCycle(data, plot_peak=False):

    data = data[:,2] # z axis
    t = 0.4 #0.4 1/2
    alpha = 0.5 #0.5 1/4
    beta = 0.7 #0.7 3/4
    gamma = 0.35 #0.35 1/6

    # all peaks 
    all_peak_pos = []
    for i in range(1, data.shape[0]-1):
        if(data[i] < data[i-1] and data[i] < data[i+1]):
            all_peak_pos.append(i)

    # filter list of peaks based on mean and standard deviation
    std_allPeak = np.std(data[all_peak_pos])
    mean_allPeak = np.mean(data[all_peak_pos])
    threshold = mean_allPeak + std_allPeak*t
    filter_peaks_pos = []
    for peak in all_peak_pos:
        if(data[peak] < threshold):
            filter_peaks_pos.append(peak)

    # compute autcorrelation to estimate len cycle
    auto_corr_coeff = calAutoCorrelation(data)

    # smooth the auto_correlation_coefficient
    for i in range(5):
        auto_corr_coeff = smooth(auto_corr_coeff)

    # approximate the length of a gait cycle by selecting the 2nd peak (positive) in the auto correlation signal
    gcLen = 0
    flag = 0
    for i in range(1,auto_corr_coeff.shape[0]-1):
        if(auto_corr_coeff[i] > auto_corr_coeff[i-1] and auto_corr_coeff[i] > auto_corr_coeff[i+1]):
            flag += 1
            if flag ==1:
                continue
            elif flag ==2:
                gcLen = i-1
                break

    # find first candidate peak
    i=1
    while i < len(filter_peaks_pos)-1:
        if filter_peaks_pos[i] - filter_peaks_pos[i-1] < gcLen and data[filter_peaks_pos[i]] < data[filter_peaks_pos[i-1]]:
            filter_peaks_pos.remove(filter_peaks_pos[i-1])
        else:
            break
        i += 1

    # find all candidate peak
    i = 1
    while i < len(filter_peaks_pos)-1:
        if filter_peaks_pos[i]-filter_peaks_pos[i-1] < alpha*gcLen:
            if data[filter_peaks_pos[i]] <= data[filter_peaks_pos[i-1]]:
                filter_peaks_pos.remove(filter_peaks_pos[i-1])
                continue
            else:
                filter_peaks_pos.remove(filter_peaks_pos[i])
                continue
        elif filter_peaks_pos[i]-filter_peaks_pos[i-1] < beta*gcLen:
            if filter_peaks_pos[i+1] - filter_peaks_pos[i] < gamma*gcLen:
                if data[filter_peaks_pos[i+1]] <= data[filter_peaks_pos[i]]:
                    filter_peaks_pos.remove(filter_peaks_pos[i])
                    continue
                else:
                    filter_peaks_pos.remove(filter_peaks_pos[i+1])
                    continue
            else:
                filter_peaks_pos.remove(filter_peaks_pos[i])
                continue
        i=i+1

    if filter_peaks_pos[-1]-filter_peaks_pos[-2] < beta*gcLen:
        filter_peaks_pos.remove(filter_peaks_pos[-1])

    # final check on len on gait found
    # if there is a gait grater then gcLen*1.7 must be probabily divided in two gait
    med_len = [filter_peaks_pos[i+1] - filter_peaks_pos[i]  for i,_ in enumerate(filter_peaks_pos[:-1])]
    med_len = np.sum(med_len)/len(med_len)

    i = 1
    j = 0
    peak_pos_modified = filter_peaks_pos[:]
    while i < len(filter_peaks_pos):
        if filter_peaks_pos[i] - filter_peaks_pos[i-1] > 1.5*med_len:
            peak_pos_modified[i+j:i+j] = [int((filter_peaks_pos[i] - filter_peaks_pos[i-1])/2) + filter_peaks_pos[i-1]]
            i += 1
            j += 1
            continue
        i += 1
    filter_peaks_pos = peak_pos_modified

    if plot_peak:
        plt.figure(figsize=(12, 3))
        plt.style.use('seaborn-darkgrid')
        plt.plot(np.arange(data.shape[0]), data, 'b-')
        #plt.scatter(filter_peaks_pos, data[filter_peaks_pos], c='red')
        plt.vlines(filter_peaks_pos, ymin=min(data)*0.95, ymax=max(data)*0.95, color='r', ls='--')
        plt.tight_layout()
        plt.show()

    return filter_peaks_pos

def segment2GaitCycle(peaks,segment):
    cycles = []
    for i in range(0,len(peaks)-1):
        cycle = segment[peaks[i]:peaks[i+1],:]
        cycle = cycle[np.newaxis,:,:]
        cycles.append(cycle)
    return cycles


def split_data_tran_val_test_gait(data, label_user, train_gait=8, val_test=0.5):

    # shuffle data and label
    data, label_user = skutils.shuffle(data, label_user)
    data, label_user = skutils.shuffle(data, label_user)

    data_for_user = []
    label_for_user = []

    for user in np.unique(label_user):
        idx_user = np.where(label_user == user)
        data_for_user.append(data[idx_user])
        label_for_user.append(label_user[idx_user])

    train_data = []
    val_data = []
    test_data = []
    train_label = []
    val_label = []
    test_label = []

    # take a number of train_gait cycle for every user
    for i,gait in enumerate(data_for_user):
        train_data.append(gait[:train_gait])
        train_label.extend(label_for_user[i][:train_gait])
        data_for_user[i] = np.delete(gait, np.arange(train_gait), 0)
        label_for_user[i] = label_for_user[i][train_gait:]

    # val_test percentage for validation and test set
    for i,gait in enumerate(data_for_user):
        stop = int(gait.shape[0]/2)
        val_data.append(gait[:stop])
        val_label.extend(label_for_user[i][:stop])  
        test_data.append(gait[stop:])
        test_label.extend(label_for_user[i][stop:])  

    train_data = np.concatenate(train_data, axis=0)
    val_data = np.concatenate(val_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)
    train_label = np.asarray(train_label)
    val_label = np.asarray(val_label)
    test_label = np.asarray(test_label)

    return train_data, val_data, test_data, train_label, val_label, test_label

def normalize_data(train, val, test, axis):

    mean = np.mean(np.reshape(train, [-1, axis]), axis=0)
    std = np.std(np.reshape(train, [-1, axis]), axis=0)

    train = (train - mean)/std
    val = (val - mean)/std
    test = (test - mean)/std

    train = np.expand_dims(train, 3)
    val = np.expand_dims(val,  3)     
    test = np.expand_dims(test, 3)
    
    return train, val, test