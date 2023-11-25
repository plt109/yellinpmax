import numpy as np

#### Functions to compute yellin max gap ts
def compute_c0_inner(x, mu):
    """
    Helper function to compute Yellin max gap, see eq 2 of arxiv 0203002v2

    Arguments:
    x ():
    mu ():

    Returns:
    bag ():

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

    Arguments:
    f_cdf ():
    input_events ():
    s2_roi ():

    Returns:


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
