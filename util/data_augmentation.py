'''
    A cross-dataset deep learning-based classifier for people fall detection and identification (ADL, 2020)
'''

from configuration import config
import os
import numpy as np

def add_gaussian_noise(data):
    # data (batch, window_len, axes)
    print(data.shape)
    data_out = np.empty([data.shape[0], data.shape[1], data.shape[2]], dtype=np.float)

    for sequence in range(data.shape[0]):
        noise = np.random.normal(0,0.01,100)
        # add noise to every axis not magnitude
        for axes in range(data.shape[2]):
            if axes in [3,7,11]: # for magnitude calculate it on noisy data
                magnitude = np.apply_along_axis(lambda x: np.sqrt(np.sum(np.power(x,2))),axis=0,arr=data_out[sequence,:,range(axes-3, axes)])
                data_out[sequence,:,axes] = magnitude 
            else: 
                data_out[sequence,:,axes] = data[sequence,:,axes] + noise
   
    return data_out

def scaling_sequence(data):
    # scaled(S) = S*((1.1 − 0.7) * rand() + 0.7)
    random = (1.1 - 0.7)* np.random.uniform(low=0.0, high=1.0) + 0.7
    scaled_sequences = data * random
    return scaled_sequences

def inter_seuquence(data):
    pass


