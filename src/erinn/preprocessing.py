"""Custom preprocessing functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import multiprocessing as mp
import os
import re
from collections import Counter
from functools import partial

import numpy as np
import tensorflow as tf
from numba import njit
from numba import types
from numba.typed import Dict
from tqdm import tqdm

from erinn.utils.io_utils import get_pkl_list
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


def add_noise(array, scale=0.05, noise_type='normal', seed=None, inplace=True):
    """Add noise to each element of the array by a certain percentage.

    In order to handle large arrays under memory constraints, this function uses in-place mode.

    Parameters
    ----------
    array : numpy.ndarray
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
    new_array : numpy.ndarray
        Noisy array.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/44257931/fastest-way-to-add-noise-to-a-numpy-array
    .. [2] https://github.com/simpeg/simpeg/blob/178b54417af0b892a3920685056a489ab2b6cda1/SimPEG/Survey.py#L501-L502
    .. [3] https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python/53688043#53688043
    .. [4] https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html
    """
    if type(noise_type) == bytes:
        noise_type = noise_type.decode('utf-8')
    array = np.asarray(array)
    raw_shape = array.shape

    if not inplace:
        new_array = array.copy()
    else:
        new_array = array

    rng = np.random.default_rng(seed)
    if noise_type == 'normal':
        new_array += scale * abs(new_array) * rng.normal(0.0, 1.0, size=raw_shape)
    elif noise_type == 'uniform':
        new_array += scale * abs(new_array) * rng.uniform(0.0, 1.0, size=raw_shape)
    else:
        raise(NotImplementedError('noise_type is not supported.'))

    if not inplace:
        return new_array.reshape(raw_shape)


# TODO: Check if z_positive implementation is correct
def source_receiver_midpoints(Tx_locations, Rx_locations, z_positive="down"):
    """
    Calculate source receiver midpoints.

    Parameters
    ----------
    Tx_locations : numpy.ndarray
        Transmitter locations.
    Rx_locations : numpy.ndarray
        Receiver locations.
    z_positive: {"up", "down"}
        Define positive is "up" or "down" in the z direction.

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

    # midpoint of the current electrode (Tx) pair (in x direction)
    Cmid = (Tx_locations[:, 0] + Tx_locations[:, 2]) / 2
    # midpoint of the potential electrode (Rx) pair (in x direction)
    Pmid = (Rx_locations[:, 0] + Rx_locations[:, 2]) / 2
    # midpoint of the current electrode (Tx) pair (in z direction)
    zsrc = (Tx_locations[:, 1] + Tx_locations[:, 3]) / 2
    # midpoint of the Cmid and Pmid (x direction)
    midx = (Cmid + Pmid) / 2
    # Half the length between Cmid and Pmid, then add zsrc
    if z_positive == "up":
        midz = -np.abs(Cmid - Pmid) / 2 + zsrc  # in z direction, positive up
    elif z_positive == "down":
        midz = np.abs(Cmid - Pmid) / 2 - zsrc  # in z direction, positive down
    else:
        raise ValueError('`z_positive` must be "up" or "down"')
    # form an 2D array
    midx = midx.reshape(-1, 1)
    midz = midz.reshape(-1, 1)

    return midx, midz


@njit
def _to_midpoint_heavy_part(new_array, z_array, midx, index_midx, midz, array, counter):
    for i in range(len(array)):
        z_array[counter[midx[i, 0]], index_midx[i]] = midz[i, 0]
        new_array[counter[midx[i, 0]], index_midx[i]] = array[i]
        counter[midx[i, 0]] += 1
    return new_array, z_array


def to_midpoint(array, Tx_locations, Rx_locations, value=0.0, dim=3):
    """Reshape inputs tensor to midpoint image.

    Default output shape = (accumulated number of same midpoint, number of midpoint, 1).
    If `dim` is 2, output shape = (accumulated number of same midpoint, number of midpoint, 1).

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
    dim : int, {2, 3} optional
        Dimension of reshaped array.

    Returns
    -------
    new_array : numpy.ndarray
        Reshaped array.

    References
    ----------
    .. [1] https://docs.python.org/3.6/library/collections.html
    .. [2] https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector
    .. [3] https://stackoverflow.com/questions/55078628/using-dictionaries-with-numba-njit-function
    .. [4] https://stackoverflow.com/questions/43218594/sort-each-column-of-an-numpy-ndarray-using-the-output-of-numpy-argsort
    .. [5] https://docs.scipy.org/doc/numpy/reference/generated/numpy.take_along_axis.html
    """

    array = array.reshape(-1)  # flatten input arrays
    midx, midz = source_receiver_midpoints(Tx_locations, Rx_locations)  # calculate midpoint
    unique_midx, index_midx = np.unique(midx, return_inverse=True)
    num_unique_midx = len(unique_midx)
    # use collections.Counter to count midx
    counter = Counter(midx.flatten())
    num_count = counter.most_common(1)[0][1]  # most_common(1) output [(value, count)]
    new_array = np.ones((num_count, num_unique_midx), dtype=np.float64) * np.nan
    z_array = np.ones((num_count, num_unique_midx), dtype=np.float64) * np.nan

    counter.subtract(counter)  # reset count to 0
    # initialize numba dictionary
    numba_counter = Dict.empty(
        key_type=types.float64,
        value_type=types.int64,
    )
    numba_counter.update(counter)  # convert counter to a numba dictionary
    new_array, z_array = _to_midpoint_heavy_part(
        new_array, z_array, midx, index_midx, midz, array, numba_counter
    )

    # sort new_array by midz of new_array element (stable sort!)
    sort_index = np.argsort(z_array, axis=0, kind='stable')
    np.take_along_axis(new_array, sort_index, axis=0)
    # replace np.nan with gived value
    new_array = np.where(np.isnan(new_array), value, new_array)

    if dim == 3:
        new_array = np.expand_dims(new_array, axis=2)  # reshape to 3D array
    elif dim == 2:
        new_array = new_array
    else:
        raise ValueError('`dim` accepts only 2 or 3')

    return new_array


def to_txrx(array, Tx_locations, Rx_locations, value=0.0, dim=3):
    """Reshape inputs tensor to Tx-Rx image.

    Default output shape = (number of Tx pair, number of Rx pair, 1).
    If `dim` is 2, output shape = (number of Tx pair, number of Rx pair, 1).

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
    dim : int, {2, 3} optional
        Dimension of reshaped array.

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

    if dim == 3:
        new_array = np.expand_dims(new_array, axis=2)  # reshape to 3D array
    elif dim == 2:
        new_array = new_array
    else:
        raise ValueError('`dim` accepts only 2 or 3')

    return new_array


def to_section(array, nCx, nCy, dim=3):
    """Reshape inputs tensor to section image.

    Default output shape = (
                       number of cell center mesh in the z (y) direction,
                       number of cell center mesh in the x direction,
                       1
                   )
    If `dim` is 2, output shape  = (
                       number of cell center mesh in the z (y) direction,
                       number of cell center mesh in the x direction
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

    if dim == 3:
        new_array = np.expand_dims(new_array, axis=2)  # reshape to 3D array
    elif dim == 2:
        new_array = new_array
    else:
        raise ValueError('`dim` accepts only 2 or 3')

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
    dataset_dir = config['dataset_dir']
    to_float32 = config['save_as_float32']
    # save_processed_data_dir = config['save_processed_data_dir']
    preprocess_resistance = config['preprocess']['resistance']
    preprocess_resistivity = config['preprocess']['resistivity']
    simulator_pkl = os.path.join(dataset_dir, 'simulator.pkl')
    # save_simulator_pkl = os.path.join(save_processed_data_dir, 'simulator.pkl')
    # do_preprocess = any(value['perform'] for action, value in preprocess.items())

    simulator = read_pkl(simulator_pkl)
    # read nCx and nCy
    nCx = simulator.mesh.nCx  # number of cell center mesh in the x direction
    nCy = simulator.mesh.nCy  # number of cell center mesh in the z (y) direction
    # read Tx_locations and Rx_locations
    Tx_locations = simulator.urf.abmn_locations[:, :4]
    Rx_locations = simulator.urf.abmn_locations[:, 4:]
    # expand simulator.config and save it
    # simulator.config = {
    #     'generating': simulator.config,  # config for generate data
    #     'preprocessing': config  # config for preprocess data
    # }
    # os.makedirs(save_processed_data_dir, exist_ok=True)
    # write_pkl(simulator, save_simulator_pkl)

    for sub_dir in ('training', 'validation', 'testing'):
        resistance_dir = os.path.join(dataset_dir, sub_dir, 'resistance')
        resistivity_dir = os.path.join(dataset_dir, sub_dir, 'resistivity')
        raw_resistance_dir = os.path.join(resistance_dir, 'raw')
        raw_resistivity_dir = os.path.join(resistivity_dir, 'raw')
        raw_resistance_list = get_pkl_list(raw_resistance_dir)
        raw_resistivity_list = get_pkl_list(raw_resistivity_dir)

        # create resistance directory
        save_resistance_dir_list = []
        for _, processes in preprocess_resistance.items():
            process_description_list = []
            for process, kwargs in processes.items():
                if process == 'add_noise':
                    process_description_list.append(
                        '['
                        + '_'.join(
                            [f"{int(kwargs['scale']*100):0>3}%",
                             kwargs['noise_type'],
                             'noise'])
                        + ']'
                    )
                elif process == 'log_transform':
                    process_description_list.append('[log_transform]')
                elif process == 'to_midpoint':
                    process_description_list.append('[midpoint]')
                elif process == 'to_txrx':
                    process_description_list.append('[txrx]')
            save_resistance_dir = os.path.join(
                resistance_dir, '_'.join(process_description_list)
            )
            os.makedirs(save_resistance_dir, exist_ok=True)
            save_resistance_dir_list.append(save_resistance_dir)

        # create resistivity directory
        save_resistivity_dir_list = []
        for _, processes in preprocess_resistivity.items():
            process_description_list = []
            for process, kwargs in processes.items():
                if process == 'to_section':
                    process_description_list.append('[section]')
            save_resistivity_dir = os.path.join(
                resistivity_dir,
                '_'.join(process_description_list)
            )
            os.makedirs(save_resistivity_dir, exist_ok=True)
            save_resistivity_dir_list.append(save_resistivity_dir)

        # preprocess resistance
        for i, (_, processes) in enumerate(preprocess_resistance.items()):
            save_resistance_dir = save_resistance_dir_list.pop(0)
            par = partial(
                _process_resistance,
                save_resistance_dir=save_resistance_dir,
                processes=processes,
                to_float32=to_float32,
                Tx_locations=Tx_locations, Rx_locations=Rx_locations,
                nCx=nCx, nCy=nCy
            )
            pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
            for data in tqdm(pool.imap_unordered(par, raw_resistance_list),
                             desc=f'Preprocess data and save to {save_resistance_dir}',
                             total=len(raw_resistance_list)):
                pass
            pool.close()
            pool.join()

            # Serial version
            # for raw_resistance_pkl in raw_resistance_list:
            #     raw_resistance = read_pkl(raw_resistance_pkl)
            #     pkl_name = os.path.basename(raw_resistance_pkl)
            #     save_resistance_pkl = os.path.join(
            #         save_resistance_dir, pkl_name
            #     )
            #     for process, kwargs in processes.items():
            #         if process == 'add_noise':
            #             add_noise(raw_resistance, **kwargs)
            #         elif process == 'log_transform':
            #             log_transform(raw_resistance, **kwargs)
            #         elif process == 'to_midpoint':
            #             raw_resistance = to_midpoint(
            #                 raw_resistance, Tx_locations, Rx_locations
            #             )
            #         elif process == 'to_txrx':
            #             raw_resistance = to_txrx(
            #                 raw_resistance, Tx_locations, Rx_locations
            #             )
            #     if to_float32:
            #         raw_resistance = raw_resistance.astype('float32')
            #     write_pkl(raw_resistance, save_resistance_pkl)

        # preprocess resistivity
        for i, (_, processes) in enumerate(preprocess_resistivity.items()):
            save_resistivity_dir = save_resistivity_dir_list.pop(0)
            par = partial(
                _process_resistivity,
                save_resistivity_dir=save_resistivity_dir,
                processes=processes,
                to_float32=to_float32,
                nCx=nCx, nCy=nCy
            )
            pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
            for data in tqdm(pool.imap_unordered(par, raw_resistivity_list),
                             desc=f'Preprocess data and save to {save_resistivity_dir}',
                             total=len(raw_resistivity_list)):
                pass
            pool.close()
            pool.join()
            # for raw_resistivity_pkl in raw_resistivity_list:
            #     raw_resistivity = read_pkl(raw_resistivity_pkl)
            #     pkl_name = os.path.basename(raw_resistivity_pkl)
            #     save_resistivity_pkl = os.path.join(
            #         save_resistivity_dir, pkl_name
            #     )
            #     for process, kwargs in processes.items():
            #         if process == 'to_section':
            #             raw_resistivity = to_section(
            #                 raw_resistivity, nCx, nCy
            #             )
            #     if to_float32:
            #         raw_resistivity = raw_resistivity.astype('float32')
            #     write_pkl(raw_resistivity, save_resistivity_pkl)
    print("IF YOU WANT TO GET THE RAW resistivity_log10, YOU SHOULD USE"
           + " `raw_resistivity_log10 = np.flipud(resistivity_log10).flatten()`")

    # if do_preprocess:
    #     pattern_raw_pkl = re.compile('raw_data_\d{6}.pkl')

    #     for root_dir, sub_dirs, files in os.walk(raw_data_dir):
    #         # filter files list so the files list will contain pickle files that match the pattern
    #         files = list(filter(pattern_raw_pkl.match, files))
    #         # If the files list is empty, continue to the next iteration of the loop
    #         if not files:
    #             continue
    #         # make sub directory
    #         sub_dir_in_processed = re.sub(raw_data_dir, save_processed_data_dir, root_dir)
    #         os.makedirs(sub_dir_in_processed, exist_ok=True)

    #         # Parallel version!
    #         par = partial(
    #             _make_processed_dataset,
    #             preprocess=preprocess,
    #             root_dir=root_dir,
    #             sub_dir_in_processed=sub_dir_in_processed,
    #             Tx_locations=Tx_locations, Rx_locations=Rx_locations,
    #             nCx=nCx, nCy=nCy
    #         )
    #         pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
    #         for data in tqdm(pool.imap_unordered(par, files),
    #                          desc=f'Preprocess data and save to {sub_dir_in_processed}',
    #                          total=len(files)):
    #             pass
    #         pool.close()
    #         pool.join()

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
        # try:
        #     print("The shape of resistance (shape of NN input data): "
        #           + f"{data['resistance'].shape}")
        #     print("The shape of resistivity (shape of NN target data): "
        #           + f"{data['resistivity_log10'].shape}")
        #     print("IF YOU WANT TO GET THE RAW resistivity_log10, YOU SHOULD USE"
        #           + " `raw_resistivity_log10 = np.flipud(resistivity_log10).flatten()`")
        # except NameError as err:
        #     pass  # no pickle files


def _process_resistance(filename, save_resistance_dir, processes,
                        to_float32, Tx_locations, Rx_locations, nCx, nCy):
    raw_resistance = read_pkl(filename)
    pkl_name = os.path.basename(filename)
    save_resistance_pkl = os.path.join(
        save_resistance_dir, pkl_name
    )
    for process, kwargs in processes.items():
        if process == 'add_noise':
            add_noise(raw_resistance, **kwargs)
        elif process == 'log_transform':
            log_transform(raw_resistance, **kwargs)
        elif process == 'to_midpoint':
            raw_resistance = to_midpoint(
                raw_resistance, Tx_locations, Rx_locations
            )
        elif process == 'to_txrx':
            raw_resistance = to_txrx(
                raw_resistance, Tx_locations, Rx_locations
            )
    if to_float32:
        raw_resistance = raw_resistance.astype('float32')
    write_pkl(raw_resistance, save_resistance_pkl)


def _process_resistivity(filename, save_resistivity_dir,
                         processes, to_float32, nCx, nCy):
    raw_resistivity = read_pkl(filename)
    pkl_name = os.path.basename(filename)
    save_resistivity_pkl = os.path.join(
        save_resistivity_dir, pkl_name
    )
    for process, kwargs in processes.items():
        if process == 'to_section':
            raw_resistivity = to_section(
                raw_resistivity, nCx, nCy
            )
    if to_float32:
        raw_resistivity = raw_resistivity.astype('float32')
    write_pkl(raw_resistivity, save_resistivity_pkl)
# def _make_processed_dataset(filename, preprocess, root_dir, sub_dir_in_processed,
#                             Tx_locations, Rx_locations, nCx, nCy):
#     # for filename in files:
#     pkl_name = os.path.join(root_dir, filename)
#     data = read_pkl(pkl_name)
#     # check if the data is dict and have "resistance" and "resistivity_log10" keys
#     if (not isinstance(data, dict)
#             or data.get('resistance') is None
#             or data.get('resistivity_log10') is None):
#         raise Exception('data is not a dict or dict does not contain essential keys')

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
#     return data
