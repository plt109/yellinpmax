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


def concatenate_spectra(support1, rates1, roi1,
                        support2, rates2, roi2):
    """ Function that concatenates 2 spectra (event/observable as function of observable), 
    while padding the in between regions properly with zeros.
    
    Probably can generalise it to beyond 2 datasets but meh.
    
    Arguments:
    support1 (pd.series, np.array, tuple): Observeable at which the rates are defined
    rates1 (pd.series, np.array, tuple): Event rates [event/observable] with exposure multiplied in
    roi1 (np.array, tuple): ROI for first dataset. E.g.: (1., 140.), (150., 3000.)
    
    And the same for support2, rates2, roi2
    
    Returns:
    support_cat (np.array): Observable at which the concatenated rates are defined
    rates_cat (np.array): Event rates [event/observable]. 
    roi_cat (np.array): Concatenated ROI. E.g: (1., 3000.)
    
    Pueh Leng Tan, 13 Oct 2023
    """
    
    # Only grabbing bits of spectrum within roi
    mask = (support1>=roi1[0]) & (support1<=roi1[1])
    support1 = support1[mask]
    rates1 = rates1[mask]

    mask = (support2>=roi2[0]) & (support2<=roi2[1])
    support2 = support2[mask]
    rates2 = rates2[mask]

    # sibei important to null out the bits with totally no events
    num_padder = 3
    zero_start = min(max(support1), max(lower_roi)) # defo correct, think about it
    zero_end = max(min(support2), min(s2_roi)) # defo correct, think about it
    support_padder = np.linspace(zero_start, zero_end, num_padder, endpoint=True) # being explicit about end point
    rates_padder = np.zeros_like(support_padder)

    # Concatenating support, rates and ROI
    support_cat = np.concatenate([support1, support_padder, support2])
    rates_cat = np.concatenate([rates1, rates_padder, rates2])
    roi_cat = np.array([min(roi1), max(roi2)])
    
    return support_cat, rates_cat, roi_cat
