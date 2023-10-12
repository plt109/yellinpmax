import numpy as np
import scipy.stats as sps
from copy import deepcopy
import pickle

from .pmax import * # not nice, change this later

def concatenated_pmax(sub_elements, mus):
    '''
    Function that determines all intervals given the concatenated dataset 
    without mixing the events from different datasets, computes pmax ts 
    on each interval and returns largest pmax ts out of all the intervals 
    considered from all datasets.

    Arguments:
    sub_elements (list): List of arrays of probability integral transformed 
                         events in each dataset, including ROI boundaries 
                         of each dataset.
                         E.g.: [array([0., 0.02, 0.15 ,0.25]),
                         array(0.25, 0.49, 1.)]
    mus (np.array): Array of total signal expectations in units of [events] 
                    in each dataset.
                    E.g.: array([ 43., 129.])

    Keyword arguments: None, for now
    
    Returns:
    p_bag[ind_max] (np.float64): pmax ts value out of all intervals considered
                                 from all datasets but without mixing events 
                                 from different datasets.
    k_bag[ind_max] (int): Number of events contained by the final interval that 
                          gave the final pmax ts value.
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

def concatenated_pit(events_arr, f_cumulative_cat, roi_arr):
    """
    Performs probability integral transform for independent
    blah blah
    
    Arguments:
    events_arr (list): List of datasets of events.
                       E.g.: [adsfsadfdsf]
    f_cumulative_cat (scipy.interpolate.interp1d): interpolated cumulative function of signal spectrum
    roi_arr (np.ndarray, list, tuple): List of ROIs.
                                       E.g.: ((1., 140.), (150., 300.)).
    
    Returns:
    pitted_events (list): List or arrays of PIT-ed observed events asldfjsadlfkjds
    mu_arr (list): List of mus in each of the ROIs
    
    """
    # Full ROI
    starts = [aa[0] for aa in roi_arr]
    ends = [aa[1] for aa in roi_arr]
    roi_cat = (min(starts), max(ends))

    # PIT-ing and concatenating individual datasets
    mu_arr = []
    pitted_events = []
    for this_dataset, this_roi in zip(events_arr, roi_arr):
        this_pitted = probability_integral_transform(this_dataset,
                f_cumulative_cat, roi_cat)
        pitted_events.append(this_pitted)

        this_mu = f_cumulative_cat(this_roi[1])-f_cumulative_cat(this_roi[0])
        mu_arr.append(this_mu)

    # To-do: Check that events stay within their partition after PIT

    return pitted_events, mu_arr


