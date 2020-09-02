'''
    from: A cross-dataset deep learning-based classifier for people fall detection and identification (ADL, 2020)
'''
import os
import numpy as np
import math
from tqdm import tqdm
import sys
import pprint

import matplotlib.pyplot as plt
from sklearn import utils as skutils
import seaborn as sns

from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat

##########################################
###### BASE FUNCTIONS FOR AUGMENTED ######
##########################################

def GenerateRandomCurves(x, sigma=0.2, knot=4):
    xx = (np.ones((x.shape[1],1))*(np.arange(0,x.shape[0], (x.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, x.shape[1]))
    x_range = np.arange(x.shape[0])
    cs = []
    for dim in range(x.shape[1]):
        cs.append(CubicSpline(xx[:,dim], yy[:,dim])(x_range))
    return np.array(cs).transpose()


def DistortTimesteps(x, sigma=0.2):
    tt = GenerateRandomCurves(x, sigma) 
    tt_cum = np.cumsum(tt, axis=0)       
    t_scale = [(x.shape[0]-1)/tt_cum[-1,0],(x.shape[0]-1)/tt_cum[-1,1],(x.shape[0]-1)/tt_cum[-1,2]]
    for dim in range(x.shape[1]):
        tt_cum[:,dim] = tt_cum[:,dim]*t_scale[dim]
    return tt_cum

def jitter(x, sigma=0.1): #0.1
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1): 
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,x.shape[2])) # shape=(1,3)
    myNoise = np.matmul(np.ones((x.shape[1],1)), scalingFactor)
    return x*myNoise

def magnitude_warp(x, sigma=0.2):
    x = x[0,:,:]
    return (x * GenerateRandomCurves(x, sigma))[np.newaxis,:,:]

def time_warp(x, sigma=0.2):
    x = x[0,:,:]
    tt_new = DistortTimesteps(x, sigma)
    X_new = np.zeros(x.shape)
    x_range = np.arange(x.shape[0])
    for dim in range(x.shape[1]):
        X_new[:,dim] = np.interp(x_range, tt_new[:,dim], x[:,dim])
    return X_new[np.newaxis,:,:]

def rotation(x):
    x = x[0,:,:]
    axis = np.random.uniform(low=-1, high=1, size=x.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(x , axangle2mat(axis,angle))[np.newaxis,:,:]

def permutation(x, nPerm=4, minSegLength=20):
    x = x[0,:,:]
    diff = False
    while diff == False:
        X_new = np.zeros(x.shape)
        idx = np.random.permutation(nPerm)
        bWhile = True
        while bWhile == True:
            segs = np.zeros(nPerm+1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, x.shape[0]-minSegLength, nPerm-1))
            segs[-1] = x.shape[0]
            if np.min(segs[1:]-segs[0:-1]) > minSegLength:
                bWhile = False
        pp = 0
        for ii in range(nPerm):
            x_temp = x[segs[idx[ii]]:segs[idx[ii]+1],:]
            X_new[pp:pp+len(x_temp),:] = x_temp
            pp += len(x_temp)
        if bool(np.asarray(x != X_new).any()):
            diff = True
    return X_new[np.newaxis,:,:]

def random_sampling(x, nSample=90):
    
    x = x[0,:,:]

    # random sampling timesteps
    tt = np.zeros((nSample,x.shape[1]), dtype=int)
    random_tt = np.sort(np.random.randint(1,x.shape[0]-1,nSample-2)) # tengo stessi timestamp per ogni asse
    for axis in range(x.shape[1]):
        tt[1:-1,axis] = random_tt
    tt[-1,:] = x.shape[0]-1

    # interpolate data based on sampled timesteps
    X_new = np.zeros(x.shape)
    for axis in range(x.shape[1]):
        X_new[:,axis] = np.interp(np.arange(x.shape[0]), tt[:,axis], x[tt[:,axis],axis])

    return X_new[np.newaxis, :, :] 

'''
def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    flip = flip[:, np.newaxis, :]
    ret = np.empty([1, x.shape[1], x.shape[2]], dtype=np.float)
    for sensor in np.arange(0, x.shape[2], 3):
        sensor_axis = np.arange(sensor, sensor+3)
        rotate_axis = np.random.permutation(sensor_axis)
        ret[:, :, sensor_axis] = flip[:, :, sensor_axis] * x[:, :, rotate_axis]
    return ret

def permutation(x, max_segments=8, seg_mode='equal'):
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(2, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    diff = False
    while not diff:
        for i, pat in enumerate(x):
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        if bool(np.asarray(warp != orig_steps).any()):
            diff = True
    return ret


def magnitude_warp(x, sigma=0.2, knot=4): 

        # knot = complexity of the interpolation curves

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) *
                  (np.linspace(0, x.shape[1]-1., num=knot+2))).T

    ret = np.zeros_like(x)

    for i, pat in enumerate(x):
        # interpolation random warping step and original step
        warper = np.array([CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(
            orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret


def time_warp(x, sigma=0.2, knot=4):

        # knot = complexity of the interpolation curves

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) *
                  (np.linspace(0, x.shape[1]-1., num=knot+2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(
                warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(
                scale*time_warp, 0, x.shape[1]-1), pat[:, dim]).T
    return ret   
'''
# dict of base function from which choose 
BASE_FUNCTION = {
    'jitter': jitter,
    'scaling': scaling,
    'permutation': permutation,
    'rotation': rotation,
    'magnitude_warp': magnitude_warp,
    'time_warp': time_warp,
    'random_sampling': random_sampling
}

##########################################
###### CORE DATA AUGMENTED ###############
##########################################

def random_guided_warp_multivariate(x, labels_user, labels_activity, slope_constraint='symmetric', use_window=True, dtw_type='normal', magnitude=True, log=False, ratio=1):
    '''
        call random guided warp on every sensors' data
    '''

    idx_prototype = None
    total_added = 0

    
    to_add, distribution = samples_to_add(labels_user, labels_activity, random_warped=True)
    to_add = to_add + (np.max(distribution) * ratio)
    
    total_sample_to_add = np.sum(to_add).astype(int)

    data_aug = np.zeros([total_sample_to_add, x.shape[1],
                         x.shape[2]], dtype=np.float)
    la_aug = np.empty([total_sample_to_add], dtype=np.int)
    lu_aug = np.empty([total_sample_to_add], dtype=np.int)

    first = True

    if magnitude:
        step = 4
        offset = 3
    else:
        step = 3
        offset = 2

    print(f'Total sample to add: {total_sample_to_add}')

    while total_sample_to_add > 0:
        for i, idx in enumerate(np.arange(0, x.shape[2], step)):

            idx_sensor = np.arange(i+(offset*i), idx+step)
            ret, idx_prototype, la, lu, first, added, to_added = random_guided_warp(
                x[:, :, idx_sensor], labels_user, labels_activity, total_sample_to_add, slope_constraint, use_window, dtw_type, idx_prototype, first, to_add, log)

            idx_to_del = []

            # delete row with all zeros from data_aug and label
            for i, _ in enumerate(ret):
                if np.all(ret[i, :, :] == 0):
                    idx_to_del.append(i)

            ret = np.delete(ret, idx_to_del, 0)
            
            print('shape sensor data augmented {}'.format(ret.shape))
            data_aug[total_added:total_added+added, :, idx_sensor] = ret

        lu_aug[total_added:total_added+added] = lu
        la_aug[total_added:total_added+added] = la

        print(f'Sample added: {added}')

        first = True
        idx_prototype = None

        total_sample_to_add -= added
        total_added += added
        to_add = to_added

    idx_to_del = []

    # delete row with all zeros from data_aug and label
    for i, _ in enumerate(data_aug):
        if np.all(data_aug[i, :, :] == 0):
            idx_to_del.append(i)

    data_aug = np.delete(data_aug, idx_to_del, 0)
    la_aug = np.delete(la_aug, idx_to_del, 0)
    lu_aug = np.delete(lu_aug, idx_to_del, 0)

    print(f'Shape data aug: {data_aug.shape}')

    return data_aug, lu_aug, la_aug


def random_guided_warp(x, labels_user, labels_activity, sample_to_add=0, slope_constraint="symmetric", use_window=True, dtw_type="normal", idx_prototype=None, first=True, to_add=[], log=False):

    import util.dtw as dtw

    added = 0

    if idx_prototype != None:
        ret_idx_prototype = idx_prototype
    else:
        ret_idx_prototype = [-1 for i, _ in enumerate(x)]

    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None

    orig_steps = np.arange(x.shape[1])
    lu = np.argmax(
        labels_user, axis=1) if labels_user.ndim > 1 else labels_user
    la = np.argmax(labels_activity,
                   axis=1) if labels_activity.ndim > 1 else labels_activity

    la_ret = []
    lu_ret = []
    ret = np.zeros([np.sum(sample_to_add), x.shape[1],
                    x.shape[2]], dtype=np.float)

    last_index = 0

    for i, pat in enumerate(tqdm(x)):
        user = lu[i]
        activity = la[i]
        if (first and to_add[user][activity] > 0) or (not first and ret_idx_prototype[i] != -1):
            if log:
                plt.figure(figsize=(12, 8))
                plt.style.use('seaborn-darkgrid')
                plt.subplot(1, 3, 1)
                plt.title(
                    'original signal, user {} activity {}'.format(user, activity))
                plt.plot(orig_steps, pat[:, 0], 'b-', label='x')
                plt.plot(orig_steps, pat[:, 1], 'g-', label='y')
                plt.plot(orig_steps, pat[:, 2], 'r-', label='z')
                plt.legend(loc='upper left')

            # remove ones of different classes and add selection based on label activity
            temp_u = np.where(lu[np.arange(x.shape[0])] == user)[0]
            temp_a = np.where(la[np.arange(x.shape[0])] == activity)[0]
            choices = [a for u in temp_u for a in temp_a if a == u and a != i]
            if len(choices) > 0:
                # pick random intra-class pattern
                if first:
                    idx = np.random.choice(choices)
                    random_prototype = x[idx]
                    ret_idx_prototype[i] = idx
                else:
                    idx = idx_prototype[i]
                    random_prototype = x[idx]
                if log:
                    plt.subplot(1, 3, 2)
                    plt.title(
                        'prototype signal, user {} activity {}'.format(user, activity))
                    plt.plot(
                        orig_steps, random_prototype[:, 0], 'b-', label='x')
                    plt.plot(
                        orig_steps, random_prototype[:, 1], 'g-', label='y')
                    plt.plot(
                        orig_steps, random_prototype[:, 2], 'r-', label='z')
                    plt.legend(loc='upper left')

                if dtw_type == "shape":
                    path = dtw.shape_dtw(random_prototype[:, [0, 1, 2]], pat[:, [
                                         0, 1, 2]], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)  # add dtw only on axis and not magnitude
                else:
                    path = dtw.dtw(random_prototype[:, [0, 1, 2]], pat[:, [
                                   0, 1, 2]], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)  # add dtw only on axis and not magnitude

                warped = pat[path[1]]

                # magnitude
                if x.shape[2] == 4:
                    for dim in range(x.shape[2]):
                        if dim not in [3]:
                            ret[last_index, :, dim] = np.interp(orig_steps, np.linspace(
                                0, x.shape[1]-1., num=warped.shape[0]), warped[:, dim]).T
                        else:
                            # magnitude data
                            magnitude = np.apply_along_axis(lambda x: np.sqrt(np.sum(np.power(
                                x, 2))), axis=1, arr=ret[last_index, :, np.arange(dim-3, dim)].transpose([1, 0]))
                            ret[last_index, :, dim] = magnitude

                # no magnitude
                if x.shape[2] == 3:
                    for dim in range(x.shape[2]):
                        ret[last_index, :, dim] = np.interp(orig_steps, np.linspace(
                            0, x.shape[1]-1., num=warped.shape[0]), warped[:, dim]).T

                to_add[user][activity] -= 1
                la_ret.append(activity)
                lu_ret.append(user)
                added += 1

                if log:
                    plt.subplot(1, 3, 3)
                    plt.title('warped signal')
                    plt.plot(
                        orig_steps, ret[last_index, :, 0], 'b-', label='x')
                    plt.plot(
                        orig_steps, ret[last_index, :, 1], 'g-', label='y')
                    plt.plot(
                        orig_steps, ret[last_index, :, 2], 'r-', label='z')
                    plt.legend(loc='upper left')
                    plt.show()

                last_index += 1

            else:
                print("There is only one pattern of class user  {} and class activity {}, skipping timewarping".format(
                    lu[i], la[i]))

    first = False

    return ret, ret_idx_prototype, la_ret, lu_ret, first, added, to_add

def random_transformation(data, labels_user, labels_activity, log=False, n_axis=3, n_sensor=1, use_magnitude=True, ratio=1, compose=False, only_compose=False, function_to_apply=[], n_func_to_apply=3):
    '''
        Take orignal train data and apply randomly transformation between jitter, scaling, rotation, permutation
        magnitude warp and time warp
    '''

    if n_func_to_apply > len(function_to_apply):
        sys.exit('Fcuntion to apply must be <= to number of possible function choosen')

    to_add, distribution = samples_to_add(labels_user, labels_activity) # per avere lo stesso numero di esempi per ogni utente

    to_add = to_add + (np.max(distribution) * ratio)

    steps = np.arange(data.shape[1])

    sensor_dict = {'0': 'accelerometer', '1': 'gyrscope', '2': 'magnetometer'}

    functions_transformation = {str(func): BASE_FUNCTION[str(func)] for func in function_to_apply}

    idx, idx_flatten = compute_sub_seq(n_axis, n_sensor, use_magnitude)

    transformed_final = np.empty(
        [0, data.shape[1], data.shape[2]], dtype=np.float)
    lu_final = np.empty([0], dtype=np.int)
    la_final = np.empty([0], dtype=np.int)

    total_transformation = 1

    while total_transformation > 0:
        data, labels_user, labels_activity = skutils.shuffle(
            data, labels_user, labels_activity)

        number_transformation = [0] * len(data)
        random_transformation = [[]] * len(data)

        for i, _ in enumerate(data):

            user = labels_user[i]
            act = labels_activity[i]

            rng = np.random.default_rng()

            added = True

            # prob_augmentation = np.random.random()

            temp_to_add = to_add[user, act]

            if temp_to_add <= 0:
                added = False

            # apply n_func_to_apply function if it's possible
            if not only_compose:
               # if temp_to_add >= 4:
                if temp_to_add >= n_func_to_apply + 1:
                    number = n_func_to_apply
                    if compose and number>1:
                        to_add[user, act] -= number + 1
                        number_transformation[i] = number + 1
                    else:
                        to_add[user, act] -= number
                        number_transformation[i] = number 
                elif temp_to_add < n_func_to_apply + 1 and temp_to_add > 1:
                    number = int(temp_to_add) - 1
                    if compose and number>1:
                        to_add[user,act] -= number + 1
                        number_transformation[i] = number + 1
                    else:
                        to_add[user, act] -= number
                        number_transformation[i] = number
                elif temp_to_add == 1:
                    number = 1
                    to_add[user,act] -= number
                    number_transformation[i] = number
            # apply only compose transformations
            else:
                if temp_to_add > 0:
                    number = n_func_to_apply
                    to_add[user, act] -= 1
                    number_transformation[i] = 1


            if added:
                transformations = rng.choice(
                    np.arange(len(functions_transformation)), number, replace=False) # 3 trasformazioni random
                random_transformation[i] = transformations

        total_transformation = np.sum(number_transformation)

        if total_transformation > 0:

            # print('total transformations to apply: ', total_transformation)

            transformed = np.zeros(
                [total_transformation, data.shape[1], data.shape[2]], dtype=np.float)
            # print(f'shape transformed {transformed.shape}')
            lu = np.zeros([total_transformation], dtype=np.int)
            la = np.zeros([total_transformation], dtype=np.int)

            past = 0

            for i, (seq, transformations) in enumerate(zip(data, random_transformation)):

                if(len(transformations) > 0):

                    act = labels_activity[i]
                    user = labels_user[i]

                    applied = False
                    all_transf = []
                    all_transf_label = []

                    seq = np.reshape(seq, (1, seq.shape[0], seq.shape[1]))

                    # plot original signal
                    if log and len(transformations) > 0:
                        plt.figure(figsize=(12, 3))
                        for j, sensor_axis in enumerate(idx):
                            plt.style.use('seaborn-darkgrid')
                            if only_compose:
                                plt.subplot(len(idx), 2, 1+2*(j))
                            elif not only_compose and compose:
                                plt.subplot(len(idx), n_func_to_apply + 2, 1+((n_func_to_apply + 2)*(j)))
                            elif not compose and not only_compose:
                                plt.subplot(len(idx), n_func_to_apply + 1, 1+((n_func_to_apply + 1)*(j)))
                            plt.title(
                                f'original {sensor_dict[str(j)]} user {labels_user[i]} activity {labels_activity[i]}')
                            plt.plot(
                                steps, seq[0, :, sensor_axis[0]], 'b-', label='x')
                            plt.plot(
                                steps, seq[0, :, sensor_axis[1]], 'g-', label='y')
                            plt.plot(
                                steps, seq[0, :, sensor_axis[2]], 'r-', label='z')

                    # apply all transformations
                    for j, transformation in enumerate(transformations):
                        applied = True
                        key_func = list(functions_transformation.keys())[
                            transformation]
                        all_transf_label.append(key_func)

                        
                        if (number_transformation[i] > 1 and compose) or only_compose:
                            if all_transf == []:
                                all_transf = seq  # seq to apply all transformation on the same sequence
                                all_transf = functions_transformation[key_func](all_transf[:, :, idx_flatten]).reshape(
                                    1, seq.shape[1], len(idx_flatten))  # (1,100,6)
                            else:
                                all_transf = functions_transformation[key_func](all_transf).reshape(
                                    1, seq.shape[1], len(idx_flatten))  # (1,100,6)

                        # seq (1,100,axis)
                        # seq[:,:,idx_flatten] (1,100,axis-magnitude)
                        
                        if not only_compose:
                            ret = functions_transformation[key_func](
                                seq[:, :, idx_flatten])[0]  # (100, axis)
                            transformed[past, :, idx_flatten] = ret.transpose()

                            if use_magnitude:
                                # calculate magnitude
                                for sensor_axis in idx:
                                    magnitude = np.apply_along_axis(lambda x: np.sqrt(
                                        np.sum(np.power(x, 2))), axis=1, arr=ret[:, sensor_axis])
                                    transformed[past, :,
                                                sensor_axis[-1]+1] = magnitude
                            if log:
                                for h, sensor_axis in enumerate(idx):
                                    plt.style.use('seaborn-darkgrid')
                                    if compose:
                                        plt.subplot(len(idx), n_func_to_apply + 2, j+2+( (n_func_to_apply + 2) * h ))
                                    else:
                                        plt.subplot(len(idx), n_func_to_apply + 1, j+2+( (n_func_to_apply + 1) * h ))
                                    plt.title(f'{key_func}')
                                    plt.plot(
                                        steps, transformed[past, :, sensor_axis[0]], 'b-', label='x')
                                    plt.plot(
                                        steps, transformed[past, :, sensor_axis[1]], 'g-', label='y')
                                    plt.plot(
                                        steps, transformed[past, :, sensor_axis[2]], 'r-', label='z')

                            la[past] = act
                            lu[past] = user
                            past += 1

                    if applied and len(all_transf) > 0 and (compose or only_compose):
                        transformed[past, :, idx_flatten] = all_transf[0,:, :].transpose()
                        if use_magnitude:
                            # calculate magnitude
                            for sensor_axis in idx:
                                magnitude = np.apply_along_axis(lambda x: np.sqrt(
                                    np.sum(np.power(x, 2))), axis=0, arr=all_transf[0, :, sensor_axis])
                                transformed[past, :,
                                            sensor_axis[-1]+1] = magnitude

                        la[past] = act
                        lu[past] = user
                        past += 1
                    
                    if log and applied and len(all_transf) > 0:
                        if not only_compose:
                            for j, sensor_axis in enumerate(idx):
                                plt.style.use('seaborn-darkgrid')
                                plt.subplot(len(idx), n_func_to_apply + 2, (n_func_to_apply + 2) + (n_func_to_apply + 2)*(j))
                                plt.title(
                                    'all transformations on same sequence acc')
                                plt.plot(
                                    steps, all_transf[0, :, sensor_axis[0]], 'b-', label='x')
                                plt.plot(
                                    steps, all_transf[0, :, sensor_axis[1]], 'g-', label='y')
                                plt.plot(
                                    steps, all_transf[0, :, sensor_axis[2]], 'r-', label='z')
                        if only_compose:
                            for j, sensor_axis in enumerate(idx):
                                plt.style.use('seaborn-darkgrid')
                                plt.subplot(len(idx), 2, 2+2*(j))
                                plt.title(
                                    f'{", ".join(all_transf_label)} on the same sequence acc')
                                plt.plot(
                                    steps, all_transf[0, :, sensor_axis[0]], 'b-', label='x')
                                plt.plot(
                                    steps, all_transf[0, :, sensor_axis[1]], 'g-', label='y')
                                plt.plot(
                                    steps, all_transf[0, :, sensor_axis[2]], 'r-', label='z')                   

                    if log and applied:
                        plt.tight_layout()
                        plt.show()

            transformed_final = np.concatenate(
                (transformed_final, transformed), axis=0)
            lu_final = np.concatenate([lu_final, lu], axis=0)
            la_final = np.concatenate([la_final, la], axis=0)

    # print(f'shape original plus transformed {transformed_final.shape[0] + data.shape[0]}')

    return transformed_final, lu_final, la_final

############################
###### UTIL FUNCTIONS ######
############################

def add_percentage(distribution, ratio=1):

    to_add = np.zeros_like(distribution)

    for act in np.arange(distribution.shape[1]):
        to_add[:,act] = (distribution[:,act] * ratio).astype(int)

    return to_add

def samples_to_add(labels_user, labels_activity, random_warped=False):

    activities = np.unique(labels_activity)
    users = np.unique(labels_user)
    distribution = np.zeros(shape=(len(users), len(activities)))
    class_no_sample = []

    for user in users:
        for act in activities:
            samples = np.intersect1d(np.where(labels_user == user), np.where(
                labels_activity == act)).shape[0]
            distribution[user, act] = samples
            if samples == 0 or (random_warped and samples == 1): # there aren't samples for random_warped
                class_no_sample.append([user,act])

    max_freq = np.max(distribution)  # max freq for every act
    max_freq = np.repeat(max_freq, len(activities))

    to_add = np.zeros_like(distribution)

    for act, freq in enumerate(max_freq):
        to_add[:, act] = ((freq - distribution[:, act])*1).astype(int)

    for el in class_no_sample:
        to_add[el[0], el[1]] = 0

    return to_add, distribution

def compute_sub_seq(n_axis, n_sensor=1, use_magnitude=True):
    '''
        based on number of axis and using of magnitude return list of index for every sensor
    '''
    idx = []
    idx_flatten = []

    if use_magnitude:
        step = 4
    else:
        step = 3

    for i in np.arange(0, n_axis, step):
        # idx.append(list(np.arange(i,i+3)))
        idx_flatten.extend(list(np.arange(i, i+3)))

    for i in np.arange(0, n_sensor*3, 3):
        idx.append(list(np.arange(i, i+3)))

    return idx, idx_flatten
