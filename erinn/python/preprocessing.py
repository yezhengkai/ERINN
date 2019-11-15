import os
import random as rn
import re

import numba
import numpy as np


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
        x[i] += x[i] * rn.uniform(-1, 1) * ratio


# TODO: When deprecating npz support, we should extract the import command
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
    from .utils.io_utils import read_config_file, read_pkl, write_pkl

    config = read_config_file(config_file)
    preprocess = config['preprocess']
    do_preprocess = any(value['perform'] for action, value in preprocess.items())

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

                    # save pickle in processed dir
                    os.makedirs(sub_dir_in_processed, exist_ok=True)
                    write_pkl(data, new_pkl_name)
