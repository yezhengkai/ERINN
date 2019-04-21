"""Numpy-related utilities."""
from __future__ import division, absolute_import, print_function

import numpy as np


def rand_replace_elem(x, ratio, fill=0, inplace=True):
    """Random replace array with specific value at certain ratio.

    Parameters
    ----------
    x : numpy.ndarray
        2D array.
    ratio : float
        Filled value's ratio. 
        It must be between 0 and 1.
    fill : x.dtype.type, default 0
        Filled value.
    inplace : bool, default True
        Whether to create a copy of `x` (False) or to replace values
        in-place (True).

    Returns
    -------
    new_x : numpy.ndarray
        When inplace is False, it would return a array copied from
        x and replace elements.
    """
    shape = x.shape
    nb_true = round(ratio * x.size)
    mask = np.zeros(shape, dtype=bool)
    for i in range(shape[0]):
        mask[i, :] = rand_bool_array(nb_true, (1, shape[1]))
    if inplace:
        x[mask] = fill
        return
    else:
        new_x = x.copy()
        new_x[mask] = fill
        return new_x


def rand_bool_array(nb_true, out_shape):
    """Generate random bool array.

    Parameters
    ----------
    nb_true : int
        Number of True.
    out_shape : tuple
        Random bool array's shape.

    Returns
    -------
    arr : numpy.array
        Random bool array.
    """
    nb_element = 1
    for i in out_shape:
        nb_element = nb_element * i
    arr = np.zeros(nb_element, dtype=bool)
    nb_true = int(nb_true)
    arr[:nb_true] = True
    np.random.shuffle(arr)
    arr = arr.reshape(out_shape)

    return arr
