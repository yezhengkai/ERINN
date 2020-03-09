"""Numpy-related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import numpy as np
from numpy.lib.arraysetops import _unique1d
from numpy.lib.arraysetops import _unpack_tuple


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


def rand_bool_array(num_true, out_shape):
    """Generate random bool array.

    Parameters
    ----------
    num_true : int
        Number of True.
    out_shape : tuple
        Random bool array's shape.

    Returns
    -------
    arr : numpy.array
        Random bool array.
    """
    num_element = 1
    for i in out_shape:
        num_element = num_element * i
    arr = np.zeros(num_element, dtype=bool)
    num_true = int(num_true)
    arr[:num_true] = True
    np.random.shuffle(arr)
    arr = arr.reshape(out_shape)

    return arr


def crop_zeros(array, remain=0, return_bound=False):
    """
    Crop the edge zero of the input array.

    Parameters
    ----------
    array : numpy.ndarray
        2D numpy array.
    remain : int
        The number of edges of all zeros which you want to remain.
    return_bound : str or bool
        Select the mode to manipulate the drawing.
        True: return array and bound.
        'only_bound': return bound.
        Others: return array.

    Returns
    -------
    out : np.ndarray, optional
        Cropped array.
    left_bound : int, optional
        The edge of the left cropping.
    right_bound : int, optional
        The edge of the right cropping.
    upper_bound : int, optional
        The edge of the upper cropping.
    lower_bound : int, optional
        The edge of the lower cropping.

    References
    ----------
    https://stackoverflow.com/questions/48987774/how-to-crop-a-numpy-2d-array-to-non-zero-values
    """
    row = array.any(1)
    if row.any():
        row_size, col_size = array.shape
        col = array.any(0)

        left_bound = np.max([col.argmax() - remain, 0])
        right_bound = np.min([col_size - col[::-1].argmax() + remain, col_size - 1])  # col[::-1] is reverse of col
        upper_bound = np.max([row.argmax() - remain, 0])
        lower_bound = np.min([row_size - row[::-1].argmax() + remain, row_size - 1])  # row[::-1] is reverse of row
        out = array[upper_bound:lower_bound, left_bound:right_bound]
    else:
        left_bound = None
        right_bound = None
        upper_bound = None
        lower_bound = None
        out = np.empty((0, 0), dtype=bool)

    if isinstance(return_bound, bool) and return_bound:
        return out, (left_bound, right_bound, upper_bound, lower_bound)
    elif return_bound == 'only_bound':
        return left_bound, right_bound, upper_bound, lower_bound
    else:
        return out


def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None, stable=True):
    """
    Find the unique elements of an array.
    The only difference from the numpy version is that this version will return a stable unique array (after sorting).
    For more details, see numpy's unique function.

    References
    ----------
    https://github.com/numpy/numpy/blob/v1.17.0/numpy/lib/arraysetops.py#L151-L294
    """
    ar = np.asanyarray(ar)

    if axis is None:
        ret = _unique1d(ar, return_index, return_inverse, return_counts)
        return _unpack_tuple(ret)

    # axis was specified and not None
    try:
        ar = np.swapaxes(ar, axis, 0)
    except np.AxisError:
        # this removes the "axis1" or "axis2" prefix from the error message
        raise np.AxisError(axis, ar.ndim)

    # Must reshape to a contiguous 2D array for this to work...
    orig_shape, orig_dtype = ar.shape, ar.dtype
    ar = ar.reshape(orig_shape[0], -1)
    ar = np.ascontiguousarray(ar)

    # asvoid can make sorting stable, but This method might avoid the warnings mentioned in
    # https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
    if stable:
        dtype = np.dtype((np.void, ar.dtype.itemsize * ar.shape[1]))
    else:
        dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    try:
        consolidated = ar.view(dtype)
    except TypeError:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype))

    def reshape_uniq(uniq):
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(-1, *orig_shape[1:])
        uniq = np.swapaxes(uniq, 0, axis)
        return uniq

    output = _unique1d(consolidated, return_index,
                       return_inverse, return_counts)
    output = (reshape_uniq(output[0]),) + output[1:]
    return _unpack_tuple(output)


def _new_view(arr, asvoid=False):
    """
    View the array as a dtype that contains the entire column per row.

    Parameters
    ----------
    arr : numpy.ndarray, Iterable, int, float

    Returns
    -------
    new_arr : numpy.ndarray
        New view of the input arr.

    References
    ----------
    https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    https://github.com/simpeg/simpeg/blob/46c6619c3368dd0000c3b2fe27bdbd042fe8c17a/SimPEG/Utils/matutils.py#L76
    """
    new_arr = np.ascontiguousarray(arr)  # Return a contiguous array (ndim >= 1) in memory (C order)
    num_cols = arr.shape[-1]  # Number of columns (if arr is a 2D array)
    if asvoid:
        dtype = np.dtype((np.void, arr.dtype.itemsize * num_cols))
        return new_arr.view(dtype)
    else:
        # This method might avoid the warnings mentioned in
        # https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
        dtype = {'names': ['f{}'.format(i) for i in range(num_cols)],
                 'formats': [arr[0, i].dtype for i in range(num_cols)]}
        return new_arr.view(dtype)


def in2d(arr1, arr2, assume_unique=False, invert=False):
    """
    Test whether each row of a 2-D array is also present in a second array.

    Returns a boolean array the same length as number of row of arr1 that is True
    where an row of arr1 is in arr2 and False otherwise.

    Parameters
    ----------
    arr1 : array_like
        Input 2D array.
    arr2 : array_like
        The 'rows' against which to test each row of arr1.
    assume_unique : bool, optional
        If True, the 'rows' of input arrays are both assumed to be unique, which can speed up the calculation.
        Default is False.
    invert : bool, optional
        If True, the 'rows' in the returned array are inverted
        (that is, False where an element of arr1 is in arr2 and True otherwise).
        Default is False.

    Returns
    -------
    arr3 : numpy.ndarray
        The rows arr1[in2d, :] are in arr2.

    References
    ----------
    https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    """
    # view the array as a dtype that contains the entire column per row.
    tmp1, tmp2 = map(_new_view, (arr1, arr2))
    return np.in1d(tmp1, tmp2, assume_unique=assume_unique, invert=invert)


def intersect2d(arr1, arr2, assume_unique=False, return_indices=False):
    """
    Find the intersection of two arrays.

    Return the sorted, unique 'rows' that are in both of the input arrays.

    Parameters
    ----------
    arr1, arr2 : array_like
        Input 2D arrays.
    assume_unique : bool
        If True, the 'rows' of input arrays are both assumed to be unique, which can speed up the calculation.
        Default is False.
    return_indices : bool
        If True, the indices which correspond to the intersection of 'rows' of the two arrays are returned.
        The first instance of a value is used if there are multiple. Default is False.

    Returns
    -------
    arr3 : numpy.ndarray
        Sorted 2D array of common and unique 'row'.
    comm1 : numpy.ndarray
        The indices of the first occurrences of the common values in arr1. Only provided if return_indices is True.
    comm2 : numpy.ndarray
        The indices of the first occurrences of the common values in arr2. Only provided if return_indices is True.

    References
    ----------
    https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    """
    # view the array as a dtype that contains the entire column per row.
    tmp1, tmp2 = map(_new_view, (arr1, arr2))
    if return_indices:
        arr3, comm1, comm2 = np.intersect1d(tmp1, tmp2,
                                            assume_unique=assume_unique, return_indices=return_indices)
        # reshape the structured array with original view.
        arr3 = arr3.view(arr1.dtype).reshape(-1, arr1.shape[-1])
        return arr3, comm1, comm2
    else:
        arr3 = np.intersect1d(tmp1, tmp2, assume_unique=assume_unique)
        # reshape the structured array with original view.
        arr3 = arr3.view(arr1.dtype).reshape(-1, arr1.shape[-1])
        return arr3


def setdiff2d(arr1, arr2, assume_unique=False):
    """
    Find the set difference of two arrays.

    Return the sorted, unique 'rows' in arr1 that are not in arr2.

    Parameters
    ----------
    arr1 : array_like
        Input 2D array.
    arr2 : array_like
        Input comparison 2D array.
    assume_unique : bool
        If True, the 'rows' of input arrays are both assumed to be unique, which can speed up the calculation.
        Default is False.

    Returns
    -------
    arr3 : numpy.ndarray
        Sorted 2D array of 'rows' in arr1 that are not in arr2.

    References
    ----------
    https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    """
    # view the array as a dtype that contains the entire column per row.
    tmp1, tmp2 = map(_new_view, (arr1, arr2))
    arr3 = np.setdiff1d(tmp1, tmp2, assume_unique=assume_unique)
    # reshape the structured array with original view.
    arr3 = arr3.view(arr1.dtype).reshape(-1, arr1.shape[-1])
    return arr3


def setxor2d(arr1, arr2, assume_unique=False):
    """
    Find the set exclusive-or of rows of two arrays.

    Return the sorted, unique 'rows' that are in only one (not both) of the input arrays.

    Parameters
    ----------
    arr1, arr2 : array_like
        Input 2D arrays.
    assume_unique : bool
        If True, the 'rows' of input arrays are both assumed to be unique, which can speed up the calculation.
        Default is False.
    Returns
    -------
    arr3 : numpy.ndarray
        Sorted 2D array of unique 'rows' that are in only one of the input arrays.

    References
    ----------
    https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    """
    # view the array as a dtype that contains the entire column per row.
    tmp1, tmp2 = map(_new_view, (arr1, arr2))
    arr3 = np.setxor1d(tmp1, tmp2, assume_unique=assume_unique)
    # reshape the structured array with original view.
    arr3 = arr3.view(arr1.dtype).reshape(-1, arr1.shape[-1])
    return arr3


def union2d(arr1, arr2):
    """
    Return the unique, sorted array of 'rows' that are in either of the two input arrays.

    Parameters
    ----------
    arr1, arr2 : array_like
        Input 2D arrays.

    Returns
    -------
    arr3 : numpy.ndarray
        Unique, sorted union of the input arrays.

    References
    ----------
    https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    """
    # view the array as a dtype that contains the entire column per row.
    tmp1, tmp2 = map(_new_view, (arr1, arr2))
    arr3 = np.union1d(tmp1, tmp2)
    # reshape the structured array with original view.
    arr3 = arr3.view(arr1.dtype).reshape(-1, arr1.shape[-1])
    return arr3


def set_ops(arr1, arr2, relation, dim=2, **kwargs):
    """
    Set operation.
    The operation contains 'in', 'intersection', 'difference ', 'exclusive-or' and 'union'.

    Parameters
    ----------
    arr1, arr2 : array_like
        Input array.
        For more details, see 'in1d', 'intersect1d', 'isin',
        'setdiff1d', 'setxor1d', 'union1d' and the corresponding 2d version.
    relation : str
        Set relationship.
    dim : int, optional
        1D or 2D set operation.
    kwargs : bool, optional
        For more details, see 'in1d', 'intersect1d', 'isin',
        'setdiff1d', 'setxor1d', 'union1d' and the corresponding 2d version.

    Returns
    -------
    For more details, see 'in1d', 'intersect1d', 'isin',
    'setdiff1d', 'setxor1d', 'union1d' and the corresponding 2d version.

    References
    ----------
    https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays

    Examples
    --------
    >>> import numpy as np
    >>> from erinn.python.utils.np_utils import set_ops
    >>> a = np.array([[5, 1, 2, 3, -1.6, 1000, 0],
    ...               [5, 1, 2, 4, -3.2, 1000, 0],
    ...               [5, 1, 2, 6, -3.6, 1000, 0],
    ...               [5, 1, 2, 7, -2.5, 1000, 0],
    ...               [5, 1, 2, 8, -2.1, 1000, 0],
    ...               [5, 1, 2, 10, -1.5, 1000, 0],
    ...               [5, 1, 2, 11, -1.8, 1000, 0],
    ...               [5, 1, 2, 12, -1.8, 1000, 0],
    ...               [5, 1, 2, 14, -1.8, 1000, 0],
    ...               [5, 1, 2, 15, -1.6, 1000, 0]])
    >>> b = a[::2, :].copy()
    >>> b = np.vstack((b, a[-1, :], a[0, :]))
    >>> print(set_ops(a, b, 'in', dim=2))
    [ True False  True False  True False  True False  True  True]
    """
    # default kwargs
    assume_unique = False
    invert = False
    return_indices = False
    # parse kwargs
    for k, v in kwargs.items():
        if k == 'assume_unique':
            assume_unique = v
        if k == 'invert':
            invert = v
        if k == 'return_indices':
            return_indices = v

    if dim == 1:
        if relation.lower() == 'in':
            return np.in1d(arr1, arr2, assume_unique=assume_unique, invert=invert)
        elif relation.lower() == 'intersect':
            return np.intersect1d(arr1, arr2, assume_unique=assume_unique, return_indices=return_indices)
        elif relation.lower() == 'isin':
            np.isin(arr1, arr2, assume_unique=assume_unique, invert=invert)
        elif relation.lower() == 'diff':
            return np.setdiff1d(arr1, arr2, assume_unique=assume_unique)
        elif relation.lower() == 'xor':
            return np.setxor1d(arr1, arr2, assume_unique=assume_unique)
        elif relation.lower() == 'union':
            return np.union1d(arr1, arr2)
        else:
            raise ValueError("The positional argument 'relation' is wrong."
                             + " Please input 'in', 'intersect', 'isin', 'diff', 'xor' or 'union'.")
    elif dim == 2:
        if relation.lower() == 'in':
            return in2d(arr1, arr2, assume_unique=assume_unique, invert=invert)
        elif relation.lower() == 'intersect':
            return intersect2d(arr1, arr2, assume_unique=assume_unique, return_indices=return_indices)
        elif relation.lower() == 'diff':
            return setdiff2d(arr1, arr2, assume_unique=assume_unique)
        elif relation.lower() == 'xor':
            return setxor2d(arr1, arr2, assume_unique=assume_unique)
        elif relation.lower() == 'union':
            return union2d(arr1, arr2)
        else:
            raise ValueError("The positional argument 'relation' is wrong."
                             + " Please input 'in', 'intersect', 'diff', 'xor' or 'union'.")
    else:
        raise ValueError("The keyword argument 'dim' is wrong. Please input 1 or 2.")
