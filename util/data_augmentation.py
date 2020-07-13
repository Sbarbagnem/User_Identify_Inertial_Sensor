'''
    from: A cross-dataset deep learning-based classifier for people fall detection and identification (ADL, 2020)
'''
import os
import numpy as np
import math
from tqdm import tqdm
import sys

import matplotlib.pyplot as plt

def add_gaussian_noise(data):
    # data (batch, window_len, axes)

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

    data_out = np.empty([data.shape[0], data.shape[1], data.shape[2]], dtype=np.float)

    for sequence in range(data.shape[0]):
        random = (1.1 - 0.7)* np.random.uniform(low=0.0, high=1.0) + 0.7
        # mul random to every axis not magnitude
        for axes in range(data.shape[2]):
            if axes in [3,7,11]: # for magnitude calculate it on noisy data
                magnitude = np.apply_along_axis(lambda x: np.sqrt(np.sum(np.power(x,2))),axis=0,arr=data_out[sequence,:,range(axes-3, axes)])
                data_out[sequence,:,axes] = magnitude 
            else: 
                data_out[sequence,:,axes] = data[sequence,:,axes] * random
   
    return data_out

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)    
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

def rotation2d(x, sigma=0.2):
    thetas = np.random.normal(loc=0, scale=sigma, size=(x.shape[0]))
    c = np.cos(thetas)
    s = np.sin(thetas)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        rot = np.array(((c[i], -s[i]), (s[i], c[i])))
        ret[i] = np.dot(pat, rot)
    return ret

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret

def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

# Proposed

def random_guided_warp_multivariate(x, labels_user, labels_activity, slope_constraint='symmetric', use_window=True, dtw_type='normal', magnitude=True, log=False):
    '''
        call random guided warp on every sensors' data
    '''

    idx_prototype = None

    data_aug = np.zeros_like(x)
    #print('shape input {}'.format(data_aug.shape))

    if magnitude:
        step = 4
        offset = 3
    else:
        step = 3
        offset = 2

    for i,idx in enumerate(np.arange(0,x.shape[2],step)):
        idx_sensor = np.arange(i+(offset*i),idx+step)
        #print('idx scelti: {}'.format(idx_sensor))
        ret, idx_prototype = random_guided_warp(x[:,:,idx_sensor], labels_user, labels_activity, slope_constraint, use_window, dtw_type, idx_prototype, log)
        #print('shape sensor\' data augmented {}'.format(ret.shape))
        data_aug[:,:,idx_sensor] = ret

    idx_to_del = []

    # delete rw with all zeros from data_aug and label
    for i,_ in enumerate(data_aug):
        if np.all(data_aug[i,:,:] == 0):
            idx_to_del.append(i)

    data_aug = np.delete(data_aug, idx_to_del, 0)
    labels_user = np.delete(labels_user, idx_to_del, 0)
    labels_activity = np.delete(labels_activity, idx_to_del, 0)

    return data_aug, labels_user, labels_activity

    
def random_guided_warp(x, labels_user, labels_activity, slope_constraint="symmetric", use_window=True, dtw_type="normal", idx_prototype=None, log=False):

    import util.dtw as dtw

    if idx_prototype != None:
        ret_idx_prototype = idx_prototype
    else:
        ret_idx_prototype = []
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    lu = np.argmax(labels_user, axis=1) if labels_user.ndim > 1 else labels_user
    la = np.argmax(labels_activity, axis=1) if labels_activity.ndim > 1 else labels_activity

    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        if np.random.random() > 0.5 or (idx_prototype != None and idx_prototype[i] != -1):
            user = lu[i]
            activity = la[i]
            if log:
                print('sample: user {} activity {}'.format(user, activity))
                plt.figure(figsize=(12, 8))
                plt.style.use('seaborn-darkgrid')
                plt.subplot(1,3,1)
                plt.title('original signal, user {} activity {}'.format(lu[i], la[i]))
                plt.plot(orig_steps, pat[:,0], 'b-', label='x')
                plt.plot(orig_steps, pat[:,1], 'g-', label='y')
                plt.plot(orig_steps, pat[:,2], 'r-', label='z')
                plt.legend(loc='upper left')
                
            # remove ones of different classes and add selection based on label activity, different from pat
            temp_u = np.where(lu[np.arange(x.shape[0])] == lu[i])[0]
            temp_a = np.where(la[np.arange(x.shape[0])] == la[i])[0]
            choices = [a for u in temp_u for a in temp_a if a == u and a != i] 
            if len(choices) > 0:        
                # pick random intra-class pattern 
                if idx_prototype == None:
                    if log:
                        print('idx prototype not define yet')
                    idx = np.random.choice(choices)
                    random_prototype = x[idx]
                    ret_idx_prototype.append(idx)
                else:
                    idx = idx_prototype[i]
                    random_prototype = x[idx]
                if log:
                    plt.subplot(1,3,2)
                    plt.title('prototype signal, user {} activity {}'.format(lu[idx], la[idx]))
                    plt.plot(orig_steps, random_prototype[:,0], 'b-', label='x')
                    plt.plot(orig_steps, random_prototype[:,1], 'g-', label='y')
                    plt.plot(orig_steps, random_prototype[:,2], 'r-', label='z')
                    plt.legend(loc='upper left')
    
                if dtw_type == "shape":
                    path = dtw.shape_dtw(random_prototype[:,[0,1,2]], pat[:,[0,1,2]], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window) # add dtw only on axis and not magnitude
                else:
                    path = dtw.dtw(random_prototype[:,[0,1,2]], pat[:,[0,1,2]], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window) # add dtw only on axis and not magnitude
                
                warped = pat[path[1]]

                # magnitude
                if x.shape[2] == 4:
                    for dim in range(x.shape[2]):
                        if dim not in [3]:
                            ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
                        else:
                            # magnitude data
                            magnitude = np.apply_along_axis(lambda x: np.sqrt(np.sum(np.power(x,2))),axis=1,arr=ret[i,:,np.arange(dim-3, dim)].transpose([1,0]))
                            ret[i,:,dim] = magnitude

                # no magnitude
                if x.shape[2] == 3:
                    for dim in range(x.shape[2]):
                        ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
                if log:
                    plt.subplot(1,3,3)
                    plt.title('warped signal')
                    plt.plot(orig_steps, ret[i,:,0], 'b-', label='x')
                    plt.plot(orig_steps, ret[i,:,1], 'g-', label='y')
                    plt.plot(orig_steps, ret[i,:,2], 'r-', label='z')
                    plt.legend(loc='upper left')
                    plt.show()

            else:
                print("There is only one pattern of class user  {} and class activity {}, skipping timewarping".format(lu[i], la[i]))
                if idx_prototype == None:
                    ret_idx_prototype.append(-1)
        else:
            if idx_prototype == None:
                ret_idx_prototype.append(-1)

    return ret, ret_idx_prototype

def discriminative_guided_warp(x, labels_user, labels_activity, batch_size=6, slope_constraint="symmetric", use_window=True, dtw_type="normal", use_variable_slice=True):
    import util.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    lu = np.argmax(labels_user, axis=1) if labels_user.ndim > 1 else labels_user
    la = np.argmax(labels_activity, axis=1) if labels_activity.ndim > 1 else labels_activity
    
    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)
        
    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        
        # remove ones of different classes
        positive = np.where(lu[choices] == lu[i])[0]
        positive = np.where(la[positive] == la[i])[0]
        negative = np.where(lu[choices] != lu[i])[0]
        negative = np.where(la[negative] == la[i])[0] # altri utenti ma stessa azione
        
        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]
                        
            # vector embedding and nearest prototype in one
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.shape_dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.shape_dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.shape_dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                   
            # Time warp
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps-warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            #print("There is only one pattern of class %d"%l[i])
            ret[i,:] = pat
            warp_amount[i] = 0.
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # unchanged
            ret = window_slice(ret, reduce_ratio=0.95)
        else:
            for i, pat in enumerate(ret):
                # Variable Sllicing
                ret[i] = window_slice(pat[np.newaxis,:,:], reduce_ratio=0.95+0.05*warp_amount[i]/max_warp)[0]
    return ret


def random_transformation(data, labels_user, labels_activity, log=False, n_axis=3, n_sensor=1, use_magnitude=True):
    '''
        Take orignal train data and apply randomly transformation between jitter, scaling, rotation, permutation
        magnitude warp and time warp
    '''

    # calculate activity class with minor sample in train
    distribution_activity = []
    for act in np.unique(labels_activity):
        samples = len([i for i in labels_activity if i == act])
        distribution_activity.append(samples)

    print('old distribution: {}'.format(distribution_activity))

    max_freq = max(distribution_activity)

    to_add = [int((max_freq-freq)) for freq in distribution_activity]

    number_transformation = []
    random_transformation = []
    steps = np.arange(data.shape[1])

    functions_transformation = {
        'jitter': jitter,
        'scaling': scaling,
        'permutation': permutation,
        'rotation': rotation,
        'magnitude warp': magnitude_warp,
        'time warp': time_warp
    }

    idx, idx_flatten = compute_sub_seq(n_axis, n_sensor, use_magnitude)

    # cycle to define output data shape and improve speed augmentation
    for i,_ in enumerate(data):

        rng = np.random.default_rng()

        added = False

        if to_add[labels_activity[i]] > 5:
            number = np.random.randint(2,4,1)
            to_add[labels_activity[i]] -= number + 1
            added = True

        if to_add[labels_activity[i]] > 0 and added == False:
            number = 1
            to_add[labels_activity[i]] -= 2
            added = True

        if added:
            number_transformation.append(number+1)
            transformations = rng.choice(np.arange(len(functions_transformation)), number, replace=False)
            random_transformation.append(transformations)
        else:
            random_transformation.append([])

    total_transformation = np.sum(number_transformation)[0]

    transformed = np.zeros([total_transformation, data.shape[1], data.shape[2]], dtype=np.float)
    lu = np.zeros([total_transformation], dtype=np.int)
    la = np.zeros([total_transformation], dtype=np.int)

    past = 0

    for i, seq in enumerate(tqdm(data)):
        applied = False
        transformations = random_transformation[i]

        seq = np.reshape(seq, (1, seq.shape[0], seq.shape[1]))
        
        if log:
            plt.figure(figsize=(12, 3))
            plt.style.use('seaborn-darkgrid')
            plt.subplot(1,6,1)
            plt.title('original accelerometer')
            plt.plot(steps, seq[0,:,0], 'b-', label='x')
            plt.plot(steps, seq[0,:,1], 'g-', label='y')
            plt.plot(steps, seq[0,:,2], 'r-', label='z')
            plt.legend(loc='upper left')
            
        all_transf = []
        #function_apply = [] # list of index of function to apply

        for j,transformation in enumerate(transformations):
            applied = True
            key_func = list(functions_transformation.keys())[transformation]

            if all_transf == []:
                all_transf = seq # seq to apply all transformation on the same sequence
                all_transf = functions_transformation[key_func](all_transf[:,:,idx_flatten]).reshape(1,seq.shape[1],len(idx_flatten)) # (1,100,6)
            else:
                all_transf = functions_transformation[key_func](all_transf).reshape(1,seq.shape[1],len(idx_flatten)) # (1,100,6)

            # seq (1,100,axis)
            # seq[:,:,idx_flatten] (1,100,axis-magnitude)
            ret = functions_transformation[key_func](seq[:,:,idx_flatten]).reshape((seq.shape[1],len(idx_flatten))) # (100, axis)
            transformed[past,:,idx_flatten] = ret.transpose()

            if use_magnitude:
                # calculate magnitude
                for sensor_axis in idx:
                    magnitude = np.apply_along_axis(lambda x: np.sqrt(np.sum(np.power(x,2))),axis=1,arr=ret[:,sensor_axis])
                    transformed[past,:,sensor_axis[-1]+1] = magnitude
            if log:
                plt.style.use('seaborn-darkgrid')
                plt.subplot(1, 6, j+2)
                plt.title('{}'.format(key_func))
                plt.plot(steps, ret[:,0], 'b-', label='x')
                plt.plot(steps, ret[:,1], 'g-', label='y')
                plt.plot(steps, ret[:,2], 'r-', label='z')
                plt.legend(loc='upper left')
        
            la[past] = labels_activity[i]
            lu[past] = labels_user[i]

            past += 1
        if applied:
            transformed[past,:,idx_flatten] = all_transf[0,:,:].transpose()
            if use_magnitude:
                # calculate magnitude
                for sensor_axis in idx:
                    magnitude = np.apply_along_axis(lambda x: np.sqrt(np.sum(np.power(x,2))),axis=0,arr=all_transf[0,:,sensor_axis])
                    transformed[past,:,sensor_axis[-1]+1] = magnitude

            la[past] = labels_activity[i]
            lu[past] = labels_user[i]
            past +=1
        if log and applied:
            plt.style.use('seaborn-darkgrid')
            plt.subplot(1, 6, len(transformations)+2)
            plt.title('all transformations on same sequence')
            plt.plot(steps, all_transf[0,:,0], 'b-', label='x')
            plt.plot(steps, all_transf[0,:,1], 'g-', label='y')
            plt.plot(steps, all_transf[0,:,2], 'r-', label='z')
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.show()

    #print('shape data augmented after radom tranformation {}'.format(transformed.shape))

    final = np.concatenate((labels_activity, la), axis=0)

    # calculate activity class with minor sample in train
    distribution_activity = []
    for act in np.unique(la):
        samples = len([i for i in final if i == act])
        distribution_activity.append(samples)

    print('new distribution: {}'.format(distribution_activity))

    return transformed, lu, la

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

    for i in np.arange(0,n_axis,step):
        #idx.append(list(np.arange(i,i+3)))
        idx_flatten.extend(list(np.arange(i,i+3)))

    for i in np.arange(0,n_sensor*3, 3):
        idx.append(list(np.arange(i, i+3)))

    return idx, idx_flatten