import numpy as np
from util.utils import mean_cross_performance
from config import CONFIG

for dataset in ['unimib']:

    path_log_file = './' + CONFIG[dataset]['log_file']
    path_to_save_train = path_log_file + 'summary_mean_train.txt'
    path_to_save_pre_train = path_log_file + 'summary_mean_pre_train.txt'
    
    with open(path_to_save_train, 'w+') as f:
        for fold in range(10):
            history = np.loadtxt(fname=path_log_file+'log_history_train_{}.txt'.format(fold))
            aaccuracy, uaccuracy = mean_cross_performance(history)
            #print('Mean accuracy on fold {} \n Activity accuracy {} \n User accuracy {}'.format(fold,aaccuracy,uaccuracy))
            f.write('Mean accuracy on fold {} \n\t Activity accuracy {} \n\t User accuracy {} \n\n'.format(fold,aaccuracy,uaccuracy))

    with open(path_to_save_pre_train, 'w+') as f:
        for fold in range(10):
            history = np.loadtxt(fname=path_log_file+'log_history_pre_train_{}.txt'.format(fold))
            aaccuracy, uaccuracy = mean_cross_performance(history)
            #print('Mean accuracy on fold {} \n Activity accuracy {} \n User accuracy {}'.format(fold,aaccuracy,uaccuracy))
            f.write('Mean accuracy on fold {} \n\t Activity accuracy {} \n\t User accuracy {} \n\n'.format(fold,aaccuracy,uaccuracy))