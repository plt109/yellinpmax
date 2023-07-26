import numpy as np
import scipy.stats as sps
import scipy.interpolate as spi
import pandas as pd 
import pickle


def get_cumulative(energies, rates, roi, num_cumulative_steps=1000):
    '''Returns unnormalised cumulative function, f_cdf, such that 
    f_cdf(b)-f_cdf(a) gives the integral of `energies` between [a,b].
    
    Arguments:
    energies: Energies at which the spectrum is defined, eg.: keV.
    rates: Rates at `energies` in units of events/keV. Can be events/(keV tonne year) as well, then scaling by
    exposure has to be done outside this function.
    roi: 2-element array or tuple indicating start and end point of region-of-interest. Same units as `energies`.
    
    Keyword arguments:
    num_cumulative_steps: Number of points used to compute cumulative function
    
    Returns:
    f_cdf: scipy.interpolate.interpolate.interp1d. Interpolated cumulative function of input spectrum
    '''
    
    # Interpolating spectrum, [events/keV]
    f = spi.interp1d(energies, rates,
                     kind='linear',
                     bounds_error=False, fill_value=(0., 0.))
    
    # Computing cumulative
    start_pt = roi[0]
    end_pt = roi[1]
    assert start_pt < end_pt, 'Dead: Start of ROI must be smaller than end of ROI.'
    
    if num_cumulative_steps<1000:
        num_cumulative_steps = 1000
        print('num_cumulative_steps too small, reverting to default value of 1000')

    test_x = np.linspace(start_pt, end_pt, num_cumulative_steps)
    test_y = f(test_x)

    step = np.unique(np.diff(test_x))[0]
    test_cumsum_y = np.cumsum(test_y)*step # [events]

    # Interpolating cumulative function, [events]
    f_cumulative = spi.interp1d(test_x, test_cumsum_y,
                         kind='linear',
                         bounds_error=False, fill_value=(0., 0.))
    return f_cumulative


# ### Functions required for probability integral transform
def probability_integral_transform(events, f_cumulative, roi):
    '''Performs probability integral transform.
    See https://en.wikipedia.org/wiki/Probability_integral_transform
    
    Arguments:
    events (np.array): array of observed events
    f_cumulative (scipy.interpolate.interpolate.interp1d): interpolated cumulative function of signal spectrum
    roi (np.array, tuple): 2-element array or tuple indicating start and end point of region of interest.
                           Same units as `energies`.
    
    Returns:
    test4 (np.array): array of observed events after probability integral transformation assuming that observed
                      events were drawn from the signal distribution under test
    '''
    # Checking ROI definition makes sense
    assert roi[0] < roi[1], 'Dead: Start of ROI must be strictly smaller than end of ROI.'
    
    # Total number of events expected in roi
    mu = f_cumulative(roi[1])-f_cumulative(roi[0])
    assert mu>0, 'Dead: Signal expectation in region of interest must be positive.'
    
    # Transformation itself
    test2 = f_cumulative(events) 
    # need this step cause f_cumulative isn't exactly cdf. not normalised
    test3 = (test2-f_cumulative(roi[0]))/mu

    # Add in ROI boundaries
    test4 = np.sort(np.concatenate([test3, [0., 1.]]))
    
    # Yellin pmax insensitive to the 'multiplicity' of events
    test4 = np.unique(test4)
    
    # might have to kill this check later got pmax with negative numbers
    assert (min(test4)>=0) & (max(test4)<=1),    'Dead: Smt wrong with probability integral transform'
    
    return test4


# ### Functions required for computing pmax test statistic itself
def compute_interval_length(elements, k):
    """
    Finds all intervals containing k events event the set of events including boundaries,
    and computes the normalised length of each interval.
    
    Arguments:
    elements (np.array): array of events after probability integral transformation.
                         must include boundaries of region of interest, 0 and 1.
    k (integer): number of events each interval contains.
    
    Returns:
    (np.array): array of each interval that contains k events.
    """
    
    window_size = k+2 # interval containing k events has length k+2
    
    elements = np.unique(elements)
    assert len(elements) >= window_size,    f"window size of {window_size} larger than array of length {len(elements)}"
    
    interval_length = []
    for i in range(len(elements) - window_size + 1):
        #print(elements[i:i+window_size]) # for debugging
        start = elements[i]
        end = elements[i+window_size-1]
        interval_length.append(end-start)
    return np.asarray(interval_length)


def ts(mu, n):
    """
    Computes the Poisson p-value of observing n events in intervals under test
    the pmax test statistic is the maximum of this ts over a set of intervals under test
    
    Arguments:
    mu (np.array): signal expectation of intervals under test
    n (int): number of observed events of interval under test
    
    Returns:
    (float): test statistic in intervals under test
    """
    return 1.-sps.poisson.cdf(n, mu=mu)


def pmax(events, mu):
    """
    Computes the pmax test statistic given a set of events and signal expectation
    in the entire region of interest.
    
    Arguments:
    events (np.array): 1d array of events after probability integral transformation.
            array has to include edges of region of interest, 0 and 1.
    mu (float64): signal expectation in entire region of interest
    
    Returns:
    p_bag[ind] (float64): pmax test statistic in optimal interval
    ind (integer): number of events the optimal interval contains
    
    """
    # events only contains ROI boundaries
    if len(events)==2:
        return ts(mu, 0)
    else:
        p_bag = []
        # Intervals containing 0..all events
        for k in range(len(events)-1):
            
            # Length of interval containing k events
            aa = compute_interval_length(events, k)
            
            # Length of interval then becomes the poisson mu
            p_bag.append(max(ts(aa*mu, k))) # max p_n for n=k
        ind = np.nanargmax(p_bag)
        
        # final chosen interval contained `ind` events
        return p_bag[ind], ind


# ### Functions required for computing precentile given pmax test statistic and overall mu in ROI
def nearest_mu(test_mu, mu_bag, is_verbose=False):
    """
    Finds the indices of the elements inside mu_bag that flank test_mu.    
    This function assumes test_mu is definitely in between two points in mu_bag
    
    Arguments:
    test_mu (float): signal expectation of the interval under test
    mu_bag (np.array): array of mu's for which the pmax ts distribution is available
    
    Returns:
    sel_ind (list): indices of mu_bag such that mu_bag[sel_ind[0]] <= test_mu <= mu_bag[sel_ind[1]]
    """
    
    aa = np.argmax(mu_bag>test_mu) # first element in mu_bag greater than test_mu
    sel_ind = [aa-1, aa]
    
    sel_mu = mu_bag[sel_ind]
    
    if is_verbose:
        print(f'is {test_mu} between {mu_bag[sel_ind]}?')
    assert (test_mu >= min(sel_mu)) & (test_mu <= max(sel_mu)), 'Problem with finding nearest mu''s.'
    
    return sel_ind


def compute_percentile(test_pmax, test_mu, mu_bag, pmax_distribution):
    """
    Computes the percentile of test_pmax. 0<= percentile <= 100.
    eg: when performing a hypothesis test at 90% confidence level, you can reject 
    the null hypothesis if this test_pmax percentile is > 90.
    
    Arguments:
    test_pmax (float64): pmax test statistic of interval under test
    test_mu (float64): signal expectation in interval under test
    mu_bag (np.array): array of mu's for which the pmax ts distribution is available
    pmax_distribution (list): list of arrays containing the toy pmax test stastic at 
                              a particular signal expectation
                              
    Returns:
    percentile (float): Percentile of the pmax test statistic.
    """
    
    # grab the nearest 2 mu's
    if (test_mu > min(mu_bag)) & (test_mu < max(mu_bag)):
        sel_ind = nearest_mu(test_mu, mu_bag)
    elif test_mu<min(mu_bag):
        sel_ind = [0]
    else:
        sel_ind = [len(mu_bag)-1]

    pmax_percentile_bag = []
    for this_ind in sel_ind:
        this_pmax_distribution = pmax_distributions[this_ind]
        pmax_percentile_bag.append(sps.percentileofscore(this_pmax_distribution, test_pmax, kind='strict'))

    if len(sel_ind)>1:
        f_percentile = spi.interp1d(mu_bag[sel_ind], pmax_percentile_bag, kind='linear',
                               bounds_error=None, fill_value=np.nan) # strictly for mu within range
        percentile = f_percentile(test_mu)
    else:
        percentile = pmax_percentile_bag[0]
    
    return percentile
