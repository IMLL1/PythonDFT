def dft(y):
    import numpy as np
    from math import tau
    """_summary_
        Runs a discrete fourier transform on a set of data
    Args:
        y (real numpy array): a numpy array of real y values, evenly distributed in x
    """
    N = len(y)  # number of samples
    # initialize empty complex amplitudes array
    dftObj = np.empty(N, dtype=complex)
    for idx in range(N):
        # we want to calculate e^(i*expArg). Exists because numpy.exp can't do complex
        expArg = tau * idx * np.arange(N) / N
        # set the nth element equal to te integral of y*e^(i*expArg) = int(y*cis(expArg)). The integrals are really Reimann sums, and expArg is a function of n
        dftObj[idx] = np.sum(y * (np.cos(expArg) + np.sin(expArg) * 1j))
    
    return dftObj