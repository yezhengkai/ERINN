"""Custom preprocessing functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import os
import re

import numba
import numpy as np
import tensorflow as tf


def log_transform(arr, inverse=False, inplace=True):
    """
    Perform a logarithmic transformation or an inverse logarithmic transformation.

    new_array[i] = log10(arr[i] + 1), arr[i] >= 0
    new_array[i] = -log10(abs(arr[i] - 1)), arr[i] < 0

    Parameters
    ----------
    arr : numpy.ndarray
        An array which you want to perform logarithmic transformation or inverse logarithmic transformation.
    inverse : bool
        Whether to perform an inverse transformation.
    inplace : bool
        Whether to use inplace mode.

    Returns
    -------
    new_arr : numpy.ndarray, optional
        If `inplace` is False, then a transformed array is returned.

    References
    ----------
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    https://stackoverflow.com/questions/21610198/runtimewarning-divide-by-zero-encountered-in-log
    """
    if inplace:
        # method 1: use boolean mask
        if inverse:
            mask = (arr >= 0)
            arr[mask] = np.power(10, arr[mask]) - 1
            arr[~mask] = -np.power(10, -arr[~mask]) + 1
        else:
            mask = (arr >= 0)
            arr[mask] = np.log10(arr[mask] + 1)
            arr[~mask] = -np.log10(np.abs(arr[~mask] - 1))

        # method 2: use index
        # ge0 = np.where(arr >= 0)  # greater equal 0
        # lt0 = np.where(arr < 0)  # less than 0
        # ge0 = np.asarray(arr >= 0).nonzero()
        # lt0 = np.asarray(arr < 0).nonzero()
        # arr[ge0] = np.log10(arr[ge0] + 1)
        # arr[lt0] = -np.log10(np.abs(arr[lt0] - 1))

        # method 3: use numpy.where(condition[, x, y])
        # An array with elements from x where condition is True, and elements from y elsewhere.
        # Note: numpy.log10(prob) is being evaluated before the numpy.where is being evaluated.
        # arr = np.where(arr >= 0, np.log10(arr + 1), -np.log10(np.abs(arr - 1)))
    else:
        new_arr = arr.copy()
        if inverse:
            mask = (new_arr >= 0)
            new_arr[mask] = np.power(10, new_arr[mask]) - 1
            new_arr[~mask] = -np.power(10, -new_arr[~mask]) + 1
        else:
            mask = (new_arr >= 0)
            new_arr[mask] = np.log10(new_arr[mask] + 1)
            new_arr[~mask] = -np.log10(np.abs(new_arr[~mask] - 1))
        return new_arr


@numba.njit()
def add_noise(x, ratio=0.1):
    """
    Add noise to each element of the array by a certain percentage.

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
        x[i] += x[i] * np.random.uniform(-1, 1) * ratio


def source_receiver_midpoints(SRCLOC, RECLOC):
    """
    Calculate source receiver midpoints.

    Parameters
    ----------
    SRCLOC : numpy.ndarray
        Source locations.
    RECLOC : numpy.ndarray
        Receiver locations.

    Returns
    -------
    midx : numpy.ndarray
        midpoints x location
    midz : numpy.ndarray
        midpoints z location

    References
    ----------
    https://github.com/simpeg/simpeg/blob/b8d716f86a4ea07ba3085fabb24c2bc974788040/SimPEG/EM/Static/Utils/StaticUtils.py#L128
    """

    # initialize midx and midz
    midx = []
    midz = []
    for i in range(len(SRCLOC)):
        Cmid = (SRCLOC[i, 0] + SRCLOC[i, 2]) / 2  # midpoint of the current electrode (Tx) pair (in x direction)
        Pmid = (RECLOC[i, 0] + RECLOC[i, 2]) / 2  # midpoint of the potential electrode (Rx) pair (in x direction)
        zsrc = (SRCLOC[i, 1] + SRCLOC[i, 3]) / 2  # midpoint of the current electrode (Tx) pair (in z direction)
        midx.append(((Cmid + Pmid) / 2))  # midpoint of the Cmid and Pmid (x direction)
        midz.append(np.abs(Cmid - Pmid) / 2 + zsrc)  # Half the length between Cmid and Pmid, then add zsrc (in z direction, positive down)
    midx = np.array(midx).reshape(-1, 1)  # form an 2D array
    midz = np.array(midz).reshape(-1, 1)  # form an 2D array

    return midx, midz


def to_midpoint(array, SRCLOC, RECLOC):
    """
    Reshape inputs tensor to midpoint image.
    shape = (accumulated number of same midpoint, number of midpoint, 1)

    Parameters
    ----------
    array : numpy.ndarray
        The array you want to reshape.
    SRCLOC : numpy.ndarray
        Source locations.
    RECLOC : numpy.ndarray
        Receiver locations.

    Returns
    -------
    new_array : numpy.ndarray
        Reshaped array.
    """

    array = array.reshape(-1)  # flatten input arrays
    midx, midz = source_receiver_midpoints(SRCLOC, RECLOC)  # calculate midpoint

    unique_midx, index_midx = np.unique(midx, return_inverse=True)
    num_unique_midx = len(unique_midx)
    num_midpoint = len(midx)  # number of midpoint
    new_array = [[] for i in range(num_unique_midx)]  # initialize new array (list of lists)
    # accumulate at same midpoint
    for i in range(num_midpoint):
        new_array[index_midx[i]].append([array[i], midz[i]])
    # sort by depth at the same midpoint
    for i in range(num_unique_midx):
        new_array[i].sort(key=lambda x: x[1])  # sort by midz (depth)
        new_array[i] = [ii[0] for ii in new_array[i]]  # drop midz

    # pad the list of lists to form an array
    new_array = tf.keras.preprocessing.sequence.pad_sequences(new_array,
                                                              dtype='float64',
                                                              padding='post')
    new_array = np.expand_dims(new_array.T, axis=2)  # reshape to 3D array

    return new_array


def to_txrx(array, SRCLOC, RECLOC):
    """
    Reshape inputs tensor to Tx-Rx image.
    shape = (number of Tx pair, number of Rx pair, 1)

    Parameters
    ----------
    array : numpy.ndarray
        The array you want to reshape.
    SRCLOC : numpy.ndarray
        Source locations.
    RECLOC : numpy.ndarray
        Receiver locations.

    Returns
    -------
    new_array : numpy.ndarray
        Reshaped array.
    """

    array = array.reshape(-1)  # flatten input arrays
    # find unique Tx pair and unique Rx pair
    unique_srcloc, index_src = np.unique(SRCLOC, return_inverse=True, axis=0)
    unique_recloc, index_rec = np.unique(RECLOC, return_inverse=True, axis=0)
    num_index = len(index_src)  #
    new_array = np.zeros((unique_srcloc.shape[0], unique_recloc.shape[0]))
    for i in range(num_index):
        new_array[index_src[i], index_rec[i]] = array[i]

    new_array = np.expand_dims(new_array, axis=2)  # reshape to 3D array

    return new_array


# TODO: When deprecating npz support, we should extract the import command.
# TODO: improve import statement, use multiprocess
def make_processed_dataset(config_file):
    """
    Preprocess raw dataset and save it to processed directory.

    Parameters
    ----------
    config_file : str, pathlib.Path or dict
        The path to the configured yaml file or the dictionary for configuration.

    Returns
    -------
    None
    """
    from erinn.utils.io_utils import read_config_file, read_pkl, write_pkl

    config = read_config_file(config_file)
    preprocess = config['preprocess']
    do_preprocess = any(value['perform'] for action, value in preprocess.items())
    # read SRCLOC and RECLOC
    if preprocess['to_midpoint']['perform']\
            or preprocess['to_txrx']['perform']:
        glob_para = read_pkl(config['glob_para_pkl'])
        SRCLOC = glob_para['SRCLOC']
        RECLOC = glob_para['RECLOC']

    if do_preprocess:
        raw_data_dir = config['raw_data_dir']
        processed_data_dir = config['processed_data_dir']
        pattern = re.compile(raw_data_dir)
        pattern_name = r'raw'
        replace_name = r'processed'

        for root_dir, sub_dirs, files in os.walk(raw_data_dir):
            for filename in files:
                if filename.endswith('pkl'):
                    pkl_name = os.path.join(root_dir, filename)
                    sub_dir_in_processed = re.sub(pattern, processed_data_dir, root_dir)
                    new_pkl_name = os.path.join(sub_dir_in_processed, re.sub(pattern_name, replace_name, filename))
                    data = read_pkl(pkl_name)

                    # preprocess
                    for k, v in preprocess.items():
                        if k == 'add_noise' and v.get('perform'):
                            add_noise(data['inputs'], **v.get('kwargs'))
                        elif k == 'log_transform' and v.get('perform'):
                            log_transform(data['inputs'], **v.get('kwargs'))
                        elif k == 'to_midpoint' and v.get('perform'):
                            data['inputs'] = to_midpoint(data['inputs'], SRCLOC, RECLOC)
                        elif k == 'to_txrx' and v.get('perform'):
                            data['inputs'] = to_txrx(data['inputs'], SRCLOC, RECLOC)

                    # save pickle in processed dir
                    os.makedirs(sub_dir_in_processed, exist_ok=True)
                    write_pkl(data, new_pkl_name)
        try:
            print(f"The shape of inputs: {data['inputs'].shape}")
            print(f"The shape of targets: {data['targets'].shape}")
        except NameError as err:
            pass  # no pickle files
