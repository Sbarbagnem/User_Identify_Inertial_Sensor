import numpy as np
from util.utils import mean_cross_performance

log_file_unimib = './data/datasets/UNIMIBDataset/record/'
path_to_save_unimib = './data/datasets/UNIMIBDataset/plot/'

log_file_sbhar = './data/datasets/SBHAR_processed/record/'
path_to_save_sbhar = './data/datasets/SBHAR_processed/plot/'

log_file_realdisp = './data/datasets/REALDISP_processed/record/'
path_to_save_realdisp = './data/datasets/REALDISP_processed/plot/'

for dataset in ['unimib', 'sbhar', 'realdisp']
for fold in range(10):
    history = np.loadtxt(fname=path_log_file+'log_history_train_{}.txt'.format(fold))
    aaccuracy, uaccuracy = mean_cross_performance(history)
    print('Mean accuracy on fold {} \n Activity accuracy {} \n User accuracy {}'.format(fold,aaccuracy,uaccuracy))