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


def add_gaussian_noise(data):
    # data (batch, window_len, axes)

    data_out = np.empty([data.shape[0], data.shape[1],
                         data.shape[2]], dtype=np.float)

    for sequence in range(data.shape[0]):
        noise = np.random.normal(0, 0.01, 100)
        # add noise to every axis not magnitude
        for axes in range(data.shape[2]):
            if axes in [3, 7, 11]:  # for magnitude calculate it on noisy data
                magnitude = np.apply_along_axis(lambda x: np.sqrt(
                    np.sum(np.power(x, 2))), axis=0, arr=data_out[sequence, :, range(axes-3, axes)])
                data_out[sequence, :, axes] = magnitude
            else:
                data_out[sequence, :, axes] = data[sequence, :, axes] + noise

    return data_out


def scaling_sequence(data):
    # scaled(S) = S*((1.1 − 0.7) * rand() + 0.7)

    data_out = np.empty([data.shape[0], data.shape[1],
                         data.shape[2]], dtype=np.float)

    for sequence in range(data.shape[0]):
        random = (1.1 - 0.7) * np.random.uniform(low=0.0, high=1.0) + 0.7
        # mul random to every axis not magnitude
        for axes in range(data.shape[2]):
            if axes in [3, 7, 11]:  # for magnitude calculate it on noisy data
                magnitude = np.apply_along_axis(lambda x: np.sqrt(
                    np.sum(np.power(x, 2))), axis=0, arr=data_out[sequence, :, range(axes-3, axes)])
                data_out[sequence, :, axes] = magnitude
            else:
                data_out[sequence, :, axes] = data[sequence, :, axes] * random

    return data_out


def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(
        loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])


def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    flip = flip[:, np.newaxis, :]
    ret = np.empty([1, x.shape[1], x.shape[2]], dtype=np.float)
    for sensor in np.arange(0, x.shape[2], 3):
        sensor_axis = np.arange(sensor, sensor+3)
        rotate_axis = np.random.permutation(sensor_axis)
        ret[:, :, sensor_axis] = flip[:, :, sensor_axis] * x[:, :, rotate_axis]
    return ret


def permutation(x, max_segments=8, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(2, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(
                    x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            while np.all(warp == orig_steps):
                warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) *
                  (np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(
            orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret


def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
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


def window_slice(x, reduce_ratio=0.7):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(
        low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(
                target_len), pat[starts[i]:ends[i], dim]).T
    return ret


def window_warp(x, window_ratio=0.3, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(
        low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i], dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(
                warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i], dim])
            end_seg = pat[window_ends[i]:, dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[i, :, dim] = np.interp(np.arange(x.shape[1]), np.linspace(
                0, x.shape[1]-1., num=warped.size), warped).T
    return ret


def random_guided_warp_multivariate(x, labels_user, labels_activity, slope_constraint='symmetric', use_window=True, dtw_type='normal', magnitude=True, log=False):
    '''
        call random guided warp on every sensors' data
    '''

    idx_prototype = None

    to_add = samples_to_add(labels_user, labels_activity, 0.30)
    total_sample_to_add = np.sum(to_add)

    data_aug = np.zeros([total_sample_to_add, x.shape[1],
                         x.shape[2]], dtype=np.float)

    first = True

    if magnitude:
        step = 4
        offset = 3
    else:
        step = 3
        offset = 2

    for i, idx in enumerate(np.arange(0, x.shape[2], step)):
        idx_sensor = np.arange(i+(offset*i), idx+step)
        print(idx_sensor)
        ret, idx_prototype, la, lu, first = random_guided_warp(
            x[:, :, idx_sensor], labels_user, labels_activity, total_sample_to_add, slope_constraint, use_window, dtw_type, idx_prototype, first, to_add, log)
        print('shape sensor data augmented {}'.format(ret.shape))
        data_aug[:, :, idx_sensor] = ret

    idx_to_del = []

    # delete row with all zeros from data_aug and label
    for i, _ in enumerate(data_aug):
        if np.all(data_aug[i, :, :] == 0):
            idx_to_del.append(i)

    data_aug = np.delete(data_aug, idx_to_del, 0)
    labels_user = np.delete(lu, idx_to_del, 0)
    labels_activity = np.delete(la, idx_to_del, 0)

    return data_aug, labels_user, labels_activity


def random_guided_warp(x, labels_user, labels_activity, sample_to_add=0, slope_constraint="symmetric", use_window=True, dtw_type="normal", idx_prototype=None, first=True, to_add=[], log=False):

    import util.dtw as dtw

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
    ret = np.zeros([np.sum(sample_to_add), x.shape[1], x.shape[2]], dtype=np.float)

    last_index = 0
    
    for i, pat in enumerate(tqdm(x)):
        user = lu[i]
        activity = la[i]
        if (first and to_add[user][activity] > 0) or (not first and ret_idx_prototype[i] != -1):
            if log:
                print('sample: user {} activity {}'.format(user, activity))
                plt.figure(figsize=(12, 8))
                plt.style.use('seaborn-darkgrid')
                plt.subplot(1, 3, 1)
                plt.title(
                    'original signal, user {} activity {}'.format(user, activity))
                plt.plot(orig_steps, pat[:, 0], 'b-', label='x')
                plt.plot(orig_steps, pat[:, 1], 'g-', label='y')
                plt.plot(orig_steps, pat[:, 2], 'r-', label='z')
                plt.legend(loc='upper left')

            # remove ones of different classes and add selection based on label activity, different from pat
            temp_u = np.where(lu[np.arange(x.shape[0])] == user)[0]
            temp_a = np.where(la[np.arange(x.shape[0])] == activity)[0]
            choices = [a for u in temp_u for a in temp_a if a == u and a != i]
            if len(choices) > 0:
                # pick random intra-class pattern
                if first:
                    idx = np.random.choice(choices)
                    random_prototype = x[idx]
                    # ret_idx_prototype.append(idx)
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
                            #ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
                            ret[last_index, :, dim] = np.interp(orig_steps, np.linspace(
                                0, x.shape[1]-1., num=warped.shape[0]), warped[:, dim]).T
                        else:
                            # magnitude data
                            magnitude = np.apply_along_axis(lambda x: np.sqrt(np.sum(np.power(
                                x, 2))), axis=1, arr=ret[last_index, :, np.arange(dim-3, dim)].transpose([1, 0]))
                            #ret[i,:,dim] = magnitude
                            ret[last_index, :, dim] = magnitude

                # no magnitude
                if x.shape[2] == 3:
                    for dim in range(x.shape[2]):
                        #ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
                        ret[last_index, :, dim] = np.interp(orig_steps, np.linspace(
                            0, x.shape[1]-1., num=warped.shape[0]), warped[:, dim]).T

                to_add[user][activity] -= 1
                la_ret.append(activity)
                lu_ret.append(user)

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

    return ret, ret_idx_prototype, la_ret, lu_ret, first


def discriminative_guided_warp(x, labels_user, labels_activity, batch_size=6, slope_constraint="symmetric", use_window=True, dtw_type="normal", use_variable_slice=True):
    import util.dtw as dtw

    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    lu = np.argmax(
        labels_user, axis=1) if labels_user.ndim > 1 else labels_user
    la = np.argmax(labels_activity,
                   axis=1) if labels_activity.ndim > 1 else labels_activity

    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)

    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    for i, pat in enumerate(tqdm(x)):
        user = lu[i]
        act = la[i]
        # guarentees that same one isnt selected
        #choices = np.delete(np.arange(x.shape[0]), i)

        # remove ones of different classes
        positive_user = np.where(lu[choices] == lu[i])[0]
        positive_act = np.where(la[positive] == la[i])[0]
        negative = np.where(lu[choices] != lu[i])[0]
        negative = np.where(la[negative] == la[i])[
            0]  # altri utenti ma stessa azione

        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(
                positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(
                negative, neg_k, replace=False)]

            # vector embedding and nearest prototype in one
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.shape_dtw(pos_prot, pos_samp,
                                                                         dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.shape_dtw(pos_prot, neg_samp,
                                                                dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.shape_dtw(
                    positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.dtw(pos_prot, pos_samp,
                                                                   dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.dtw(pos_prot, neg_samp, dtw.RETURN_VALUE,
                                                          slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH,
                               slope_constraint=slope_constraint, window=window)

            # Time warp
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(
                0, x.shape[1]-1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps-warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(
                    0, x.shape[1]-1., num=warped.shape[0]), warped[:, dim]).T
        else:
            #print("There is only one pattern of class %d"%l[i])
            ret[i, :] = pat
            warp_amount[i] = 0.
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # unchanged
            ret = window_slice(ret, reduce_ratio=0.95)
        else:
            for i, pat in enumerate(ret):
                # Variable Sllicing
                ret[i] = window_slice(
                    pat[np.newaxis, :, :], reduce_ratio=0.95+0.05*warp_amount[i]/max_warp)[0]
    return ret


def samples_to_add(labels_user, labels_activity, ratio=0.3):

    # distribution act for user
    distribution = []  # list of user and activity for user
    for user in set(labels_user):
        distribution.append([])
        for act in set(labels_activity):
            samples = len([i for i, (u, a) in enumerate(
                zip(labels_user, labels_activity)) if a == act and u == user])
            distribution[user].append(samples)

    max_freq = np.max(distribution[:])

    to_add = [((np.asarray(max_freq)-np.asarray(freq))*ratio).astype(int)
              for freq in distribution]

    return to_add


def random_transformation(data, labels_user, labels_activity, log=False, n_axis=3, n_sensor=1, use_magnitude=True, ratio=0.5):
    '''
        Take orignal train data and apply randomly transformation between jitter, scaling, rotation, permutation
        magnitude warp and time warp
    '''

    to_add = samples_to_add(labels_user, labels_activity, ratio)

    steps = np.arange(data.shape[1])

    sensor_dict = {'0': 'accelerometer', '1': 'gyrscope', '2': 'magnetometer'}

    functions_transformation = {
        'jitter': jitter,
        'window slice': window_slice,
        'permutation': permutation,
        'rotation': rotation,
        'magnitude warp': magnitude_warp,
        'time warp': time_warp
    }

    idx, idx_flatten = compute_sub_seq(n_axis, n_sensor, use_magnitude)

    transformed_final = np.empty(
        [0, data.shape[1], data.shape[2]], dtype=np.float)
    lu_final = np.empty([0], dtype=np.int)
    la_final = np.empty([0], dtype=np.int)

    max_cycle = 10

    while max_cycle > 0:
        data, labels_user, labels_activity = skutils.shuffle(
            data, labels_user, labels_activity)
        number_transformation = []
        random_transformation = []
        for i, _ in enumerate(data):

            rng = np.random.default_rng()

            added = False

            prob_augmentation = np.random.random()

            if to_add[labels_user[i]][labels_activity[i]] > 5 and prob_augmentation > 0.5:
                number = np.random.randint(1, 3, 1)[0]
                to_add[labels_user[i]][labels_activity[i]] -= number + 1
                added = True
                number_transformation.append(number+1)

            if to_add[labels_user[i]][labels_activity[i]] > 0 and added == False and prob_augmentation > 0.5:
                number = 1
                to_add[labels_user[i]][labels_activity[i]] -= number
                added = True
                number_transformation.append(number)

            if added:
                transformations = rng.choice(
                    np.arange(len(functions_transformation)), number, replace=False)
                random_transformation.append(transformations)
            else:
                number_transformation.append(0)
                random_transformation.append([])

        total_transformation = np.sum(number_transformation)
        print('total transformations to apply: ', total_transformation)

        transformed = np.zeros(
            [total_transformation, data.shape[1], data.shape[2]], dtype=np.float)
        lu = np.zeros([total_transformation], dtype=np.int)
        la = np.zeros([total_transformation], dtype=np.int)

        past = 0

        for i, (seq, transformations) in enumerate(zip(data, random_transformation)):
            applied = False
            all_transf = []

            seq = np.reshape(seq, (1, seq.shape[0], seq.shape[1]))

            if log and len(transformations) > 0:
                plt.figure(figsize=(12, 3))
                for j, sensor_axis in enumerate(idx):
                    plt.style.use('seaborn-darkgrid')
                    plt.subplot(len(idx), 5, 1+5*(j))
                    plt.title(
                        f'original {sensor_dict[str(j)]} user {labels_user[i]} activity {labels_activity[i]}')
                    plt.plot(
                        steps, seq[0, :, sensor_axis[0]+j], 'b-', label='x')
                    plt.plot(
                        steps, seq[0, :, sensor_axis[1]+j], 'g-', label='y')
                    plt.plot(
                        steps, seq[0, :, sensor_axis[2]+j], 'r-', label='z')

            for j, transformation in enumerate(transformations):
                applied = True
                key_func = list(functions_transformation.keys())[
                    transformation]

                if number_transformation[i] > 1:
                    if all_transf == []:
                        all_transf = seq  # seq to apply all transformation on the same sequence
                        all_transf = functions_transformation[key_func](all_transf[:, :, idx_flatten]).reshape(
                            1, seq.shape[1], len(idx_flatten))  # (1,100,6)
                    else:
                        all_transf = functions_transformation[key_func](all_transf).reshape(
                            1, seq.shape[1], len(idx_flatten))  # (1,100,6)

                # seq (1,100,axis)
                # seq[:,:,idx_flatten] (1,100,axis-magnitude)
                ret = functions_transformation[key_func](
                    seq[:, :, idx_flatten])[0]  # (100, axis)
                transformed[past, :, idx_flatten] = ret.transpose()

                if use_magnitude:
                    # calculate magnitude
                    for sensor_axis in idx:
                        magnitude = np.apply_along_axis(lambda x: np.sqrt(
                            np.sum(np.power(x, 2))), axis=1, arr=ret[:, sensor_axis])
                        transformed[past, :, sensor_axis[-1]+1] = magnitude
                if log:
                    for h, sensor_axis in enumerate(idx):
                        plt.style.use('seaborn-darkgrid')
                        plt.subplot(len(idx), 5, j+2+5*(h))
                        plt.title(f'{key_func}')
                        plt.plot(
                            steps, transformed[past, :, sensor_axis[0]+h], 'b-', label='x')
                        plt.plot(
                            steps, transformed[past, :, sensor_axis[1]+h], 'g-', label='y')
                        plt.plot(
                            steps, transformed[past, :, sensor_axis[2]+h], 'r-', label='z')

                la[past] = labels_activity[i]
                lu[past] = labels_user[i]

                past += 1

            if applied and len(all_transf) > 0:
                transformed[past, :, idx_flatten] = all_transf[0,
                                                               :, :].transpose()
                if use_magnitude:
                    # calculate magnitude
                    for sensor_axis in idx:
                        magnitude = np.apply_along_axis(lambda x: np.sqrt(
                            np.sum(np.power(x, 2))), axis=0, arr=all_transf[0, :, sensor_axis])
                        transformed[past, :, sensor_axis[-1]+1] = magnitude

                la[past] = labels_activity[i]
                lu[past] = labels_user[i]
                past += 1

            if log and applied and len(all_transf) > 0:
                for j, sensor_axis in enumerate(idx):
                    plt.style.use('seaborn-darkgrid')
                    plt.subplot(len(idx), 5, len(transformations)+2+5*(j))
                    plt.title('all transformations on same sequence acc')
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

        max_cycle -= 1

    final = np.concatenate((labels_activity, la_final), axis=0)

    # calculate activity class with minor sample in train
    distribution_activity = []
    for act in np.unique(labels_activity):
        samples = len([i for i in final if i == act])
        distribution_activity.append(samples)

    print('new distribution: {}'.format(distribution_activity))

    return transformed_final, lu_final, la_final


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
