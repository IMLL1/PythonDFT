def dft1d(y, n=None):
    import numpy as np
    from math import tau

    """_summary_
        Runs a discrete fourier transform on a 1 dimensional set of data. The Fourier transform length 
    Args:
        y (real numpy array): a numpy array of real y values, evenly distributed in t
        n (int):              number of frequencies to use, including the zero frequency. Defaults to the length of the signal.
    """
    y = np.array(y)  # convert to array if list
    lenY = len(y)

    if n == None:
        n = lenY  # if no n is provided, n is the length of y
    if n < lenY:
        y = y[:n]  # truncate y if n is less than the length of y
    if n > lenY:
        y = np.append(y, [0] * (n - lenY))  # append 0s if n is bigger than y's length

    # initialize empty complex amplitudes array
    dftObj = np.empty(n, dtype=complex)
    for idx in range(n):
        # set the nth element equal to te integral of y*e^(2pi(frequency)(x between 0 and 1)i). The integrals are really Reimann sums
        dftObj[idx] = np.sum(y * np.exp(tau * idx * np.arange(n) * 1j / n))

    return dftObj
