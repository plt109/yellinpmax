import numpy as np
import scipy.stats as sps
from copy import deepcopy
import pickle

from pmax import * # not nice, change this later

def cal_split_pmax(sub_elements, mus):
    '''
    Function that determines all intervals given the concatenated dataset 
    and computes pmax ts on each interval
    
    Returns:
    p_bag[ind_max] :
    k_bag[ind_max] :
    '''
    ### intervals time
    p_bag = []
    k_bag = []
    # for each dataset
    for ii, this_sub in enumerate(sub_elements):
        # for interval containing k events
        for this_k in range(len(this_sub)-1):
            window_size = this_k+2
            interval_length = []
            # sliding window of k events through all events
            for i in range(len(this_sub) - window_size + 1):
                start = this_sub[i]
                end = this_sub[i+window_size-1]
                interval_length.append(end-start)
            interval_length = np.asarray(interval_length)
            p_bag.append(max(ts(interval_length*mus[ii], this_k)))
            k_bag.append(this_k)
    ind_max = np.nanargmax(p_bag) # find interval with pmax
    
    return p_bag[ind_max], k_bag[ind_max]
