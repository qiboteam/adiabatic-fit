import numpy as np

def generate_gamma(xarr, nsteps, e0, e1, shape=10, scale=0.5, swap=False, nvals=5000):
    """
    Generate a sample of data following a Gamma distribution.
    
    Args:
        xarr (list), nsteps (int), e0 (float), e1 (float): adiabatic evolution 
        parameters, important to map the physical problem into the evolutionary approach.
        shape (float), scale (float): Gamma dist. parameters.
        swap (bool): True if one want to swap the problem wrt y axis.
        nvals (int)

    Returns:
        cdf (np.ndarray): cdf values.
        sample (np.ndarray), normed_sample (np.ndarray): original generated sample
            and same sample normalized into the range [0, 1].
    """
    sample = np.random.gamma(shape, scale, nvals)

    if swap:
        sample *= -1

    normed_sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample)) 
    h, b = np.histogram(normed_sample, bins=nsteps, range=[0,1], density=False)
    # Sanity check
    np.testing.assert_allclose(b, xarr)
    cdf_raw = np.insert(np.cumsum(h)/len(h), 0, 0)

    # Translate the CDF such that it goes from 0 to 1
    cdf_norm = (cdf_raw - np.min(cdf_raw)) / (np.max(cdf_raw) - np.min(cdf_raw))
    # And now make it go from the E_initial to E_final (E0 to E1)
    cdf = e0 + cdf_norm*(e1 - e0)
    return np.array(cdf), np.array(sample), np.array(normed_sample)