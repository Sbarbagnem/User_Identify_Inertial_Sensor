import numpy as np
import os
import matplotlib.pyplot as plt 
from util.plot_history import plot_performance

#path_log_file = './data/datasets/UNIMIBDataset/record/'
#path_to_save = './data/datasets/UNIMIBDataset/plot/'

path_log_file = './data/datasets/SBHAR_processed/record/'
path_to_save = './data/datasets/SBHAR_processed/plot/'

path_log_file = './data/datasets/REALDISP_processed/record/'
path_to_save = './data/datasets/REALDISP_processed/plot/'

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

for fold in range(10):
    history = np.loadtxt(   fname=path_log_file+'log_history_train_{}.txt'.format(fold),
                            usecols=(1,3))
    plot_performance(history[:,0], history[:,1], fold, path_to_save, True)