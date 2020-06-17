'''
    A cross-dataset deep learning-based classifier for people fall detection and identification (ADL, 2020)
'''

from configuration import config
import os
import numpy as np

def add_gaussian_noise(data):
    # data (batch, window_len, axes)
    data_out = np.empty([data.shape[0], data.shape[1], data.shape[2]], dtype=np.float)

    noise = np.random.normal(0,0.01,100)
    # add noise to every axis (no magnitude)
    data_out[:,:,0] = data[:,:,0] + noise
    data_out[:,:,1] = data[:,:,1] + noise
    data_out[:,:,2] = data[:,:,2] + noise

    # calculate magnitude on noisy axes
    magnitude = np.apply_along_axis(lambda x: np.sqrt(np.sum(np.power(x,2))),axis=2,arr=data_out)
    data_out[:,:,2] = magnitude
   
    return data_out

def scaling_sequence(data):
    # scaled(S) = S*((1.1 − 0.7) * rand() + 0.7)
    random = (1.1 - 0.7)* np.random.uniform(low=0.0, high=1.0) + 0.7
    scaled_sequences = data * random
    return scaled_sequences

def inter_seuquence(data):
    pass


