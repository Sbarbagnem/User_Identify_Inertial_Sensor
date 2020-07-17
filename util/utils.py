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


def samples_to_down(labels_user, labels_activity, ratio=0.3):

    # distribution act for user
    distribution = []  # list of user and activity for user
    for user in set(labels_user):
        distribution.append([])
        for act in set(labels_activity):
            samples = len([i for i, (u, a) in enumerate(
                zip(labels_user, labels_activity)) if a == act and u == user])
            distribution[user].append(samples)

    min_freq = np.min(distribution[:])

    to_down = [((np.asarray(freq)-np.asarray(min_freq))*0.3).astype(int)
               for freq in distribution]

    return to_down
