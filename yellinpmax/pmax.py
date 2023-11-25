import gzip
import pickle
import importlib.util
from pathlib import Path

import numpy as np
import scipy.stats as sps
import scipy.interpolate as spi


#### Functions for handling input spectrum
def get_cumulative(energies, rates, roi, num_cumulative_steps=1000):
    """Returns unnormalised cumulative function, f_cdf, such that 
    f_cdf(b)-f_cdf(a) gives the integral of `energies` between [a,b].
    
    Arguments:
    energies (np.ndarray): Energies at which the spectrum is defined, eg.: keV.
    rates (np.ndarray): Rates at `energies` in units of events/keV. Can be events/(keV tonne year) as well, then scaling by
    exposure has to be done outside this function.
    roi (np.ndarray, list, tuple): 2-element array or tuple indicating start
                                   and end point of region-of-interest. Same units as `energies`.
    
    Keyword arguments:
    num_cumulative_steps (integer): Number of points used to compute cumulative function
    
    Returns:
    f_cumulative (scipy.interpolate.interpolate.interp1d): Interpolated cumulative function of input spectrum
    """
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


#### Functions required for probability integral transform
def probability_integral_transform(events, f_cumulative, roi):
    """Performs probability integral transform.
    See https://en.wikipedia.org/wiki/Probability_integral_transform
    
    Arguments:
    events (np.ndarray): array of observed events
    f_cumulative (scipy.interpolate.interpolate.interp1d): interpolated cumulative function of signal spectrum
    roi (np.ndarray, list, tuple): 2-element array or tuple indicating start and end point of region of interest.
                                   Same units as `energies`.
    
    Returns:
    test4 (np.ndarray): array of observed events after probability integral transformation assuming that observed
                      events were drawn from the signal distribution under test
    """
    # No events to PIT anyway
    if len(events)==0:
        return events

    # Checking ROI definition makes sense
    assert roi[0] < roi[1], 'Dead: Start of ROI must be strictly smaller than end of ROI.'

    # Checking that events are indeed within ROI
    assert np.sum((events<roi[0])|(events>roi[1]))==0, 'Dead: Why are there events outside ROI?'
    
    # Total number of events expected in roi
    mu = f_cumulative(roi[1])-f_cumulative(roi[0])
    assert mu>0, 'Dead: Signal expectation in region of interest must be positive.'
    
    # Transformation itself
    test2 = f_cumulative(events) 
    # need this step cause f_cumulative isn't exactly cdf. not normalised
    test3 = (test2-f_cumulative(roi[0]))/mu

    # random variable is always positive after PIT
    assert (min(test3)>=0) & (max(test3)<=1), 'Dead: Smt wrong with probability integral transform'

    return test3


#### Functions required for computing pmax test statistic itself
def compute_interval_length(elements, k):
    """Finds all intervals containing k events event the set of events including boundaries,
    and computes the normalised length of each interval.
    
    Arguments:
    elements (np.ndarray): array of sorted events after probability integral transformation.
                         must include boundaries of region of interest, 0 and 1.
    k (integer): number of events each interval contains.
    
    Returns:
    (np.ndarray): array of each interval that contains k events.
    """
    # interval containing k events has length k+2
    window_size = k+2

    # np.unique returns sorted array, adding ROI boundaries just in case
    elements = np.unique(np.concatenate([elements, [0., 1.]]))

    assert len(elements) >= window_size, f"window size of {window_size} larger than array of length {len(elements)}"
    
    interval_length = []
    for i in range(len(elements) - window_size + 1):
        #print(elements[i:i+window_size]) # for debugging
        start = elements[i]
        end = elements[i+window_size-1]
        interval_length.append(end-start)
    return np.asarray(interval_length)


def ts(mu, n):
    """Computes the Poisson p-value of observing n events in intervals under test
    the pmax test statistic is the maximum of this ts over a set of intervals under test
    
    Arguments:
    mu (np.ndarray): signal expectation of intervals under test
    n (integer): number of observed events of interval under test
    
    Returns:
    (float): test statistic in intervals under test
    """
    return 1.-sps.poisson.cdf(n, mu=mu)


def pmax(events, mu):
    """Computes the pmax test statistic given a set of events and signal expectation
    in the entire region of interest.
    
    Arguments:
    events (np.ndarray): 1d array of events after probability integral transformation.
            array has to include edges of region of interest, 0 and 1.
    mu (float): signal expectation in entire region of interest
    
    Returns:
    p_bag[ind] (float): pmax test statistic in optimal interval
    ind (integer): number of events the optimal interval contains
    """
    # Yellin insensitive to 'multiplicity' of events. Do this to save some
    # computational time.
    events = np.unique(events)

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


#### Functions required for computing precentile given pmax test statistic and overall mu in ROI
def nearest_mu(test_mu, mu_bag, is_verbose=False):
    """Finds the indices of the elements inside mu_bag that flank test_mu.    
    This function assumes test_mu is definitely in between two points in mu_bag
    
    Arguments:
    test_mu (float): signal expectation of the interval under test
    mu_bag (np.ndarray): array of mu's for which the pmax ts distribution is available
    pmax_null_ts (dictionary): dictionary containing pmax ts null distribution
                               and the various mu's
    
    Returns:
    sel_ind (list): indices of mu_bag such that mu_bag[sel_ind[0]] <= test_mu <= mu_bag[sel_ind[1]]
    """
    # first element in mu_bag greater than test_mu
    aa = np.argmax(mu_bag>test_mu)
    sel_ind = [aa-1, aa]
    
    sel_mu = mu_bag[sel_ind]
    
    if is_verbose:
        print(f'is {test_mu} between {mu_bag[sel_ind]}?')
    assert (test_mu >= min(sel_mu)) & (test_mu <= max(sel_mu)), 'Problem with finding nearest mu''s.'
    
    return sel_ind


def compute_percentile(test_pmax, test_mu, pmax_null_ts):
    """Computes the percentile of test_pmax. 0<= percentile <= 100.
    eg: when performing a hypothesis test at 90% confidence level, you can reject 
    the null hypothesis if this test_pmax percentile is > 90.
    
    Arguments:
    test_pmax (float): pmax test statistic of interval under test
    test_mu (float): signal expectation in interval under test
    pmax_null_ts (dictionary): dictionary containing pmax ts null distribution
                               and the various mu's it was defined for
                              
    Returns:
    percentile (float): Percentile of the pmax test statistic.
    """
    mu_bag = pmax_null_ts['mu_bag']
    pmax_distributions = pmax_null_ts['pmax_distributions']

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


#### Functions for loading default pmax ts distributions
def default_pmax_distributions():
    """Loads the default dictionary containing the pmax ts null distribution.
    The dictionary contains 3 keys, 'readme', 'mu_bag', 'pmax_distributions'.

    Returns:
    pmax_ts (dictionary): Dictionary containing the pmax ts null distribution
                          and the various mu's it was toyed out for
    """
    # Path to file with a dictionary of pmax null ts toys (mu from [2.5, 70.], 1e4 toy pmax ts
    # each.
    spec = importlib.util.find_spec('yellinpmax')
    _ts_fname = 'pmax_distributions_v2.pickle.gz'
    pmax_ts_file = Path(spec.origin).parents[1] / 'data' / _ts_fname

    assert pmax_ts_file.exists(), \
           f'Dead: Unable to find default pmax distribution pickle, {_ts_fname}.'

    with gzip.open(pmax_ts_file) as fid:
        pmax_ts = pickle.load(fid)

    return pmax_ts


#### Function to toy out pmax ts distributions
def get_pmax_distribution(poisson_mu, num_toys=int(1e4)):
    """ Function to toyMC out the pmax ts distribution given a particular 
    signal expectation in the analysis region of interest (ROI).

    Arguments:
    poisson_mu (float64): Signal expectation in analysis ROI
    num_toys (int): Number of toyMCs. Default of 1e4 is abit of an overkill,
                    but why not?

    Returns:
    pmax_bag (np.array): Array of pmax ts for signal expectation of `poisson_mu`
    """
    n_observed = sps.poisson.rvs(poisson_mu, size=num_toys)

    pmax_bag = np.ones(num_toys)*-1.

    for ind_toy, this_n in enumerate(n_observed):
        events = np.sort(sps.uniform.rvs(size=this_n))
        events = np.concatenate([[0.], events, [1.]]) # add in boundaries
        pmax_bag[ind_toy] = pmax(events, poisson_mu)[0]

    return pmax_bag
