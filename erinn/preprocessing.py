"""Custom preprocessing functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import multiprocessing as mp
import os
import re
from functools import partial

import numba
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from erinn.utils.io_utils import read_config_file
from erinn.utils.io_utils import read_pkl
from erinn.utils.io_utils import write_pkl


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
def add_noise(x, scale=0.05, noise_type='normal'):
    """Add noise to each element of the array by a certain percentage.

    In order to handle large arrays under memory constraints, this function uses in-place mode.

    Parameters
    ----------
    x : numpy.ndarray
        Array that you wanted to add noise.
    scale : float, default 0.05
        If noise_type is 'normal', scale is represent the standard deviation.
        If noise_type is 'uniform', the noise added to element is proportional to this variable.
    noise_type: str, {'normal', 'uniform'}, default normal
        Noise type.
        "normal" indicates that the noise is sampled from a Gaussian probability distribution function.
        "uniform" indicates that the noise is sampled from a uniform probability distribution function.

    Returns
    -------
    None

    References
    ----------
    .. [1] https://stackoverflow.com/questions/44257931/fastest-way-to-add-noise-to-a-numpy-array
    .. [2] https://github.com/simpeg/simpeg/blob/178b54417af0b892a3920685056a489ab2b6cda1/SimPEG/Survey.py#L501-L502
    .. [3] https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python/53688043#53688043
    .. [4] https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html
    """
    # Since version 0.28.0, the generator is thread-safe and fork-safe.
    # Each thread and each process will produce independent streams of random numbers.
    # x = x.reshape(-1)  # flat view
    x = x.ravel()  # flat view

    if noise_type == 'normal':
        for i in range(len(x)):
            x[i] += scale * abs(x[i]) * np.random.normal(0.0, 1.0)
    elif noise_type == 'uniform':
        for i in range(len(x)):
            x[i] += scale * abs(x[i]) * np.random.uniform(-1.0, 1.0)
    else:
        raise(NotImplementedError('noise_type is not supported.'))


def source_receiver_midpoints(Tx_locations, Rx_locations):
    """
    Calculate source receiver midpoints.

    Parameters
    ----------
    Tx_locations : numpy.ndarray
        Transmitter locations.
    Rx_locations : numpy.ndarray
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
    for i in range(len(Tx_locations)):
        # midpoint of the current electrode (Tx) pair (in x direction)
        Cmid = (Tx_locations[i, 0] + Tx_locations[i, 2]) / 2
        # midpoint of the potential electrode (Rx) pair (in x direction)
        Pmid = (Rx_locations[i, 0] + Rx_locations[i, 2]) / 2
        # midpoint of the current electrode (Tx) pair (in z direction)
        zsrc = (Tx_locations[i, 1] + Tx_locations[i, 3]) / 2
        # midpoint of the Cmid and Pmid (x direction)
        midx.append(((Cmid + Pmid) / 2))
        # Half the length between Cmid and Pmid, then add zsrc (in z direction, positive down)
        midz.append(np.abs(Cmid - Pmid) / 2 + zsrc)
    midx = np.array(midx).reshape(-1, 1)  # form an 2D array
    midz = np.array(midz).reshape(-1, 1)  # form an 2D array

    return midx, midz


def to_midpoint(array, Tx_locations, Rx_locations, value=0.0):
    """Reshape inputs tensor to midpoint image.

    shape = (accumulated number of same midpoint, number of midpoint, 1)

    Parameters
    ----------
    array : numpy.ndarray
        The array you want to reshape.
    Tx_locations : numpy.ndarray
        Transmitter locations.
    Rx_locations : numpy.ndarray
        Receiver locations.
    value : float
        The value of the blank element you want to fill in.

    Returns
    -------
    new_array : numpy.ndarray
        Reshaped array.
    """

    array = array.reshape(-1)  # flatten input arrays
    midx, midz = source_receiver_midpoints(Tx_locations, Rx_locations)  # calculate midpoint

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
                                                              padding='post',
                                                              value=value)
    new_array = np.expand_dims(new_array.T, axis=2)  # reshape to 3D array

    return new_array


def to_txrx(array, Tx_locations, Rx_locations, value=0.0):
    """Reshape inputs tensor to Tx-Rx image.

    shape = (number of Tx pair, number of Rx pair, 1)

    Parameters
    ----------
    array : numpy.ndarray
        The array you want to reshape.
    Tx_locations : numpy.ndarray
        Transmitter locations.
    Rx_locations : numpy.ndarray
        Receiver locations.
    value : float
        The value of the blank element you want to fill in.

    Returns
    -------
    new_array : numpy.ndarray
        Reshaped array.
    """

    array = array.reshape(-1)  # flatten input arrays
    # find unique Tx pair and unique Rx pair
    unique_Tx_locations, index_src = np.unique(Tx_locations, return_inverse=True, axis=0)
    unique_Rx_locations, index_rec = np.unique(Rx_locations, return_inverse=True, axis=0)

    # create new zero array and assign value
    num_index = len(index_src)
    new_array = np.ones((unique_Tx_locations.shape[0],
                         unique_Rx_locations.shape[0]),
                        dtype=np.float) * value
    for i in range(num_index):
        new_array[index_src[i], index_rec[i]] = array[i]

    new_array = np.expand_dims(new_array, axis=2)  # reshape to 3D array

    return new_array


def to_section(array, nCx, nCy):
    """Reshape inputs tensor to section image.

    shape = (
        number of cell center mesh in the z (y) direction,
        number of cell center mesh in the x direction,
        1
    )

    Parameters
    ----------
    array : numpy.ndarray
        The array you want to reshape.
    nCx : int
        Number of cell center mesh in the x direction.
    nCy : int
        Number of cell center mesh in the z (y) direction.

    Returns
    -------
    new_array : numpy.ndarray
        Reshaped array.
    """
    array = array.reshape(-1)  # flatten input arrays
    new_array = np.flipud(array.reshape((nCy, nCx)))
    new_array = np.expand_dims(new_array, axis=2)  # reshape to 3D array

    return new_array


# TODO: use tfRecord
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

    config = read_config_file(config_file)
    raw_data_dir = config['raw_data_dir']
    save_processed_data_dir = config['save_processed_data_dir']
    preprocess = config['preprocess']
    simulator_pkl = os.path.join(raw_data_dir, 'simulator.pkl')
    save_simulator_pkl = os.path.join(save_processed_data_dir, 'simulator.pkl')
    do_preprocess = any(value['perform'] for action, value in preprocess.items())

    simulator = read_pkl(simulator_pkl)
    # read nCx and nCy
    nCx = simulator.mesh.nCx  # number of cell center mesh in the x direction
    nCy = simulator.mesh.nCy  # number of cell center mesh in the z (y) direction
    # read Tx_locations and Rx_locations
    Tx_locations = simulator.urf.abmn_locations[:, :4]
    Rx_locations = simulator.urf.abmn_locations[:, 4:]
    # expand simulator.config and save it
    simulator.config = {
        'generate': simulator.config,  # config for generate data
        'preprocess': config  # config for preprocess data
    }
    os.makedirs(save_processed_data_dir, exist_ok=True)
    write_pkl(simulator, save_simulator_pkl)

    if do_preprocess:
        pattern_raw_pkl = re.compile('raw_data_\d{6}.pkl')

        for root_dir, sub_dirs, files in os.walk(raw_data_dir):
            # filter files list so the files list will contain pickle files that match the pattern
            files = list(filter(pattern_raw_pkl.match, files))
            # If the files list is empty, continue to the next iteration of the loop
            if not files:
                continue
            # make sub directory
            sub_dir_in_processed = re.sub(raw_data_dir, save_processed_data_dir, root_dir)
            os.makedirs(sub_dir_in_processed, exist_ok=True)

            # Parallel version!
            par = partial(
                _make_processed_dataset,
                preprocess=preprocess,
                root_dir=root_dir,
                sub_dir_in_processed=sub_dir_in_processed,
                Tx_locations=Tx_locations, Rx_locations=Rx_locations,
                nCx=nCx, nCy=nCy
            )
            pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
            for data in tqdm(pool.imap_unordered(par, files),
                             desc=f'Preprocess data and save to {sub_dir_in_processed}',
                             total=len(files)):
                pass
            pool.close()
            pool.join()

            # Serial version!
            # for filename in files:
            #     pkl_name = os.path.join(root_dir, filename)
            #     data = read_pkl(pkl_name)
            #     # check if the data is dict and have "resistance" and "resistivity_log10" keys
            #     if (not isinstance(data, dict)
            #             or data.get('resistance') is None
            #             or data.get('resistivity_log10') is None):
            #         continue

            #     # preprocess
            #     for k, v in preprocess.items():
            #         if k == 'add_noise' and v.get('perform'):
            #             add_noise(data['resistance'], **v.get('kwargs'))
            #         elif k == 'log_transform' and v.get('perform'):
            #             log_transform(data['resistance'], **v.get('kwargs'))
            #         elif k == 'to_midpoint' and v.get('perform'):
            #             data['resistance'] = to_midpoint(
            #                 data['resistance'], Tx_locations, Rx_locations
            #             )
            #         elif k == 'to_txrx' and v.get('perform'):
            #             data['resistance'] = to_txrx(
            #                 data['resistance'], Tx_locations, Rx_locations
            #             )
            #         elif k == 'to_section' and v.get('perform'):
            #             data['resistivity_log10'] = to_section(
            #                 data['resistivity_log10'], nCx, nCy
            #             )

            #     # save pickle in processed dir
            #     new_pkl_name = os.path.join(
            #         sub_dir_in_processed,
            #         re.sub(r'raw', r'processed', filename)
            #     )
            #     write_pkl(data, new_pkl_name)

        # show information about input / target tensor shape
        try:
            print("The shape of resistance (shape of NN input data): "
                  + f"{data['resistance'].shape}")
            print("The shape of resistivity (shape of NN target data): "
                  + f"{data['resistivity_log10'].shape}")
            print("IF YOU WANT TO GET THE RAW resistivity_log10, YOU SHOULD USE"
                  + " `raw_resistivity_log10 = np.flipud(resistivity_log10).flatten()`")
        except NameError as err:
            pass  # no pickle files


def _make_processed_dataset(filename, preprocess, root_dir, sub_dir_in_processed,
                            Tx_locations, Rx_locations, nCx, nCy):
    # for filename in files:
    pkl_name = os.path.join(root_dir, filename)
    data = read_pkl(pkl_name)
    # check if the data is dict and have "resistance" and "resistivity_log10" keys
    if (not isinstance(data, dict)
            or data.get('resistance') is None
            or data.get('resistivity_log10') is None):
        raise Exception('data is not a dict or dict does not contain essential keys')

    # preprocess
    for k, v in preprocess.items():
        if k == 'add_noise' and v.get('perform'):
            add_noise(data['resistance'], **v.get('kwargs'))
        elif k == 'log_transform' and v.get('perform'):
            log_transform(data['resistance'], **v.get('kwargs'))
        elif k == 'to_midpoint' and v.get('perform'):
            data['resistance'] = to_midpoint(
                data['resistance'], Tx_locations, Rx_locations
            )
        elif k == 'to_txrx' and v.get('perform'):
            data['resistance'] = to_txrx(
                data['resistance'], Tx_locations, Rx_locations
            )
        elif k == 'to_section' and v.get('perform'):
            data['resistivity_log10'] = to_section(
                data['resistivity_log10'], nCx, nCy
            )

    # save pickle in processed dir
    new_pkl_name = os.path.join(
        sub_dir_in_processed,
        re.sub(r'raw', r'processed', filename)
    )
    write_pkl(data, new_pkl_name)
    return data
