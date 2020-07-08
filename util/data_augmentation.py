'''
    from: A cross-dataset deep learning-based classifier for people fall detection and identification (ADL, 2020)
'''
import os
import numpy as np
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

def spawner(x, labels, sigma=0.05, verbose=0):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    
    import util.dtw as dtw
    from helper import dtw_graph1d
    random_points = np.random.randint(low=1, high=x.shape[1]-1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:     
            random_sample = x[np.random.choice(choices)]
            # SPAWNER splits the path into two randomly
            path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            combined = np.concatenate((np.vstack(path1), np.vstack(path2+random_points[i])), axis=1)
            if verbose:
                print(random_points[i])
                dtw_value, cost, DTW_map, path = dtw.dtw(pat, random_sample, return_flag = dtw.RETURN_ALL, slope_constraint="symmetric", window=window)
                dtw_graph1d(cost, DTW_map, path, pat, random_sample)
                dtw_graph1d(cost, DTW_map, combined, pat, random_sample)
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=mean.shape[0]), mean[:,dim]).T
        else:
            print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = pat
    return jitter(ret, sigma=sigma)

def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True):
    # https://ieeexplore.ieee.org/document/8215569
    
    import util.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        
    ret = np.zeros_like(x)
    for i in tqdm(range(ret.shape[0])):
        # get the same class as i
        choices = np.where(l == l[i])[0]
        if choices.size > 0:        
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]
            
            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                        
            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]
            
            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern 
                    weighted_sums += np.ones_like(weighted_sums) 
                else:
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(np.log(0.5)*dtw_value/dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight 
            
            ret[i,:] = average_pattern / weighted_sums[:,np.newaxis]
        else:
            print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = x[i]
    return ret

# Proposed

def random_guided_warp_multivariate(x, labels_user, labels_activity, slope_constraint='symmetric', use_window=True, dtw_type='normal', magnitude=True):
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
        ret, idx_prototype = random_guided_warp(x[:,:,idx_sensor], labels_user, labels_activity, slope_constraint, use_window, dtw_type, idx_prototype)
        #print('shape sensor\' data augmented {}'.format(ret.shape))
        data_aug[:,:,idx_sensor] = ret

    return data_aug

    
def random_guided_warp(x, labels_user, labels_activity, slope_constraint="symmetric", use_window=True, dtw_type="normal", idx_prototype=None):
    import util.dtw as dtw

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
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes and add selection based on label activity
        choices = np.where(lu[choices] == lu[i])[0]
        choices = np.where(la[choices] == la[i])[0]
        if choices.size > 0:        
            # pick random intra-class pattern 
            if idx_prototype == None:
                idx = np.random.choice(choices)
                random_prototype = x[idx]
                ret_idx_prototype.append(idx)
            else:
                random_prototype = x[idx_prototype[i]]

            #print('shape prototype and sample {} {}'.format(random_prototype.shape, pat.shape))
   
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

        else:
            print("There is only one pattern of class user  {} and class activity {}, skipping timewarping".format(lu[i], la[i]))
            ret_idx_prototype.append(-1)
            ret[i,:] = pat
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


def random_transformation(data, labels_user, labels_activity, log=False, use_magnitude=True):
    '''
        Take orignal train data and apply randomly transformation between jitter, scaling, rotation, permutation
        magnitude warp and time warp
    '''

    steps = np.arange(data.shape[1])

    transformed = np.empty([0, data.shape[1], data.shape[2]], dtype=np.float)
    lu = np.empty([0], dtype=np.int)
    la = np.empty([0], dtype=np.int)

    functions_transformation = {
        'jitter': jitter,
        'scaling': scaling,
        'rotation': rotation,
        'permutation': permutation,
        'magnitude warp': magnitude_warp,
        'time warp': time_warp
    }

    for i, seq in enumerate(tqdm(data)):
        seq = np.reshape(seq, (1, seq.shape[0], seq.shape[1]))
        number_transformations = np.random.randint(0, len(functions_transformation))
        rng =np.random.default_rng()
        random_transformation = rng.choice(len(functions_transformation), size=number_transformations, replace=False) if number_transformations > 0 else []

        if len(random_transformation) != 0 :
            
            if log:
                plt.figure(figsize=(12, 8))
                plt.subplot(2,4,1)
                plt.title('original')
                plt.plot(steps, seq[0,:,3], '-')
                
            for i,transformation in enumerate(random_transformation):
                key_func = list(functions_transformation.keys())[transformation]
                ret = functions_transformation[key_func](seq).reshape((1,100,seq.shape[2])) # apply random transformation only on axis not magnitude
                transformed = np.concatenate((transformed, ret), axis=0)
                lu = np.concatenate((lu, labels_user[i].reshape((1))), axis=0)
                la = np.concatenate((la, labels_activity[i].reshape((1))), axis=0)
                if log:
                    plt.subplot(2, 4, i+2)
                    plt.title('{}'.format(key_func))
                    plt.plot(steps, ret[0,:,3], '-')
        if log:
            plt.tight_layout()
            plt.show()

    print('shape data augmented after radom tranformation {}'.format(transformed.shape))
    return transformed, lu, la
