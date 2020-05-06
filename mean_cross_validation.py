import numpy as np
from util.utils import mean_cross_performance
from config import CONFIG

for dataset in ['unimib', 'sbhar', 'realdisp']:

    path_log_file = './' + CONFIG[dataset]['log_file']
    path_to_save = path_log_file + 'summary_mean.txt'
    
    with open(path_to_save, 'w+') as f:
        for fold in range(10):
            history = np.loadtxt(fname=path_log_file+'log_history_train_{}.txt'.format(fold))
            aaccuracy, uaccuracy = mean_cross_performance(history)
            #print('Mean accuracy on fold {} \n Activity accuracy {} \n User accuracy {}'.format(fold,aaccuracy,uaccuracy))
            f.write('Mean accuracy on fold {} \n\t Activity accuracy {} \n\t User accuracy {} \n\n'.format(fold,aaccuracy,uaccuracy))