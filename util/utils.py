import numpy as np
import matplotlib.pyplot as plt


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

def split_balanced_data(lu, la, folders=10):

    # dict to save indexes' example for every folder
    indexes = {}
    for i in np.arange(folders):
        indexes[str(i)] = []

    last_folder = 0

    # balance split label user-activity in every folders
    for user in np.unique(lu):  # specific user
        temp_index_label_user = [index for index, x in enumerate(
            lu) if x == user]  # index of specific user

        for act in np.unique(la):  # specific activity
            temp_index_label_act = [index for index, x in enumerate(
                la) if x == act and index in temp_index_label_user]  # index of specific activity of user

            # same percentage data in every folder
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
