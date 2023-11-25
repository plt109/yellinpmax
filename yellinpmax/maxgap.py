import numpy as np

#### Functions to compute yellin max gap ts
def compute_c0_inner(x, mu):
    """
    Helper function to compute the probability of the max gap size, x_max, 
    being smaller than some particular value of x, P(x_max<x) = c0 = 1-pvalue.
    
    Essentially computing the c0(x,mu) object in eq 2 of arxiv 0203002v2. Paper
    says, 'For a 90% CL UL, increase xsec until mu and the observed x are such
    that c0 reaches 0.90.'

    Eg. If c0 = 0.9, one can reject that particular hypothesis (usually some
    xsec that gives that particular signal spectrum) at 90% CL.

    Arguments:
    x (np.float64): Maximum gap size. Ie. Signal expectation in max gap
    mu (np.float64): Total signal expectation in ROI

    Returns:
    bag (np.float64): Probability of the maximum gap size, x, being smaller 
                      than a particular value of x
    """

    k_max = int(np.floor(mu/x))
    bag = 0.
    for k in range(k_max+1): # because `range` doesn't include endpoint, but summation in yellin does.
        aa = (1+k/(mu-k*x))
        bb = np.exp(-k*x)*(k*x-mu)**k
        bb /= np.math.factorial(k)
        bag += (aa*bb)
        
    return bag


def compute_c0(f_cdf, input_events, s2_roi):
    """
    Function to find the maximum gap given a signal spectrum and a set of
    events, and compute the probability of that particular max gap being smaller
    than some particular value of x.

    Eg. If c0 = 0.9, one can reject that particular hypothesis (usually some
    xsec that gives that particular signal spectrum) at 90% CL.

    Arguments:
    f_cdf (scipy.interpolate.interpolate.interp1d): Interpolated cumulative
                                                    function of input spectrum
    input_events (np.array): Array of of observed events 
    s2_roi (np.array, list, tuple): 2-element array or tuple indicating start and end point of
                                    region-of-interest.

    Returns:
    c0 (np.float64): P(x_max<x) = 1-pvalue
    overall_mu (np.float64): Total signal expectation in ROI
    """

    # Compute overall mu
    overall_mu = f_cdf(s2_roi[1])-f_cdf(s2_roi[0])
    
    # Only consider events within roi
    mask = (input_events>s2_roi[0]) & (input_events<s2_roi[1]) # strictly within. gonna add edges in later anyway
    events = input_events[mask]
    
    if overall_mu == 0:
        c0 = np.nan
    else:
        # Compute gap definition
        aa = np.concatenate(([s2_roi[0]], events, [s2_roi[1]]))
        gap_start = aa[0:-1]
        gap_end = aa[1:]

        # Compute x for each gap
        x = np.ones_like(gap_start)*np.nan
    
        for ind_gap, this_gap_start in enumerate(gap_start):
            x[ind_gap] = f_cdf(gap_end[ind_gap])-f_cdf(this_gap_start)
                
        # Find max gap
        x_max = np.max(x)

        c0 = compute_c0_inner(x_max, overall_mu)
    
    return c0, overall_mu
