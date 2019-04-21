import random as rn

import numba
import numpy as np


def log_transform(x, inverse=False, inplace=True):
    """


    Parameters
    ----------
    x : numpy.ndarray
        An array which
    inverse : bool
        inverse transform

    inplace : bool

    Returns
    -------

    References
    ----------
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    https://stackoverflow.com/questions/21610198/runtimewarning-divide-by-zero-encountered-in-log
    """
    if inplace:
        # method 1: use boolean mask
        if inverse:
            mask = (x >= 0)
            x[mask] = np.power(10, x[mask]) - 1
            x[~mask] = -np.power(10, -x[~mask]) + 1
        else:
            mask = (x >= 0)
            x[mask] = np.log10(x[mask] + 1)
            x[~mask] = -np.log10(np.abs(x[~mask] - 1))

        # method 2: use index
        # ge0 = np.where(x >= 0)  # greater equal 0
        # lt0 = np.where(x < 0)  # less than 0
        # ge0 = np.asarray(x >= 0).nonzero()
        # lt0 = np.asarray(x < 0).nonzero()
        # x[ge0] = np.log10(x[ge0] + 1)
        # x[lt0] = -np.log10(np.abs(x[lt0] - 1))

        # method 3: use numpy.where(condition[, x, y])
        # An array with elements from x where condition is True, and elements from y elsewhere.
        # Note: numpy.log10(prob) is being evaluated before the numpy.where is being evaluated.
        # x = np.where(x >= 0, np.log10(x + 1), -np.log10(np.abs(x - 1)))
        return
    else:
        new_x = x.copy()
        if inverse:
            mask = (new_x >= 0)
            new_x[mask] = np.power(10, new_x[mask]) - 1
            new_x[~mask] = -np.power(10, -new_x[~mask]) + 1
        else:
            mask = (new_x >= 0)
            new_x[mask] = np.log10(new_x[mask] + 1)
            new_x[~mask] = -np.log10(np.abs(new_x[~mask] - 1))
        return new_x


@numba.njit()
def add_noise(x, ratio=0.05):
    """Add noise to each element of the array by a certain percentage.

    In order to handle large arrays under memory constraints, this function uses in-place mode.

    Parameters
    ----------
    x : numpy.ndarray
        Array that you wanted to add noise.
    ratio : float, default 0.05
        Noise added to element is proportional to this value.

    Returns
    -------
    None

    References
    ----------
    https://stackoverflow.com/questions/44257931/fastest-way-to-add-noise-to-a-numpy-array
    """
    x = x.reshape(-1)  # flat view
    for i in range(len(x)):
        x[i] += x[i] * rn.uniform(-1, 1) * ratio
