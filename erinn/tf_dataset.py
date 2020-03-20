"""Functions used for tf.data.Dataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

from functools import partial

import tensorflow as tf

from erinn.preprocessing import add_noise
from erinn.preprocessing import log_transform
from erinn.preprocessing import to_midpoint
from erinn.preprocessing import to_txrx
from erinn.preprocessing import to_section
from erinn.utils.io_utils import read_pkl


def read_dataset(file_path, read_dataset_info):
    """Read dataset from pickle files and preprocess it.

    Parameters
    ----------
    file_path : str, os.PathLike or pathlib.Path
        The path of pickle file.
    read_dataset_info : dict

    Returns
    -------
    resistance : numpy.ndarray
        The input data of the neural network.
    resistivity_log10 : numpy.ndarray
        The target data of the neural network.
    """
    # read data and assign
    data = read_pkl(file_path.numpy().decode('utf-8'))
    resistance = data['resistance']
    resistivity_log10 = data['resistivity_log10']
    # parse read_dataset_info dictionary
    preprocess = read_dataset_info['preprocess']
    Tx_locations = read_dataset_info['Tx_locations']
    Rx_locations = read_dataset_info['Rx_locations']
    nCx = read_dataset_info['nCx']
    nCy = read_dataset_info['nCy']

    # preprocess
    for k, v in preprocess.items():
        if k == 'add_noise' and v.get('perform'):
            add_noise(resistance, **v.get('kwargs'))
        elif k == 'log_transform' and v.get('perform'):
            log_transform(resistance, **v.get('kwargs'))
        elif k == 'to_midpoint' and v.get('perform'):
            resistance = to_midpoint(
                resistance, Tx_locations, Rx_locations
            )
        elif k == 'to_txrx' and v.get('perform'):
            resistance = to_txrx(
                resistance, Tx_locations, Rx_locations
            )
        elif k == 'to_section' and v.get('perform'):
            resistivity_log10 = to_section(
                resistivity_log10, nCx, nCy
            )

    return resistance, resistivity_log10


def tf_read_dataset(file_path, read_dataset_info):
    """Reading dataset from pickle files and preprocess it (TensorFlow version).

    Parameters
    ----------
    file_path : str, os.PathLike or pathlib.Path
        The path of pickle file.
    read_dataset_info : dict

    Returns
    -------
    resistance : tensorflow.Tensor
        The input data of the neural network.
    resistivity_log10 : tensorflow.Tensor
        The target data of the neural network.

    References
    ----------
    .. [1] https://www.tensorflow.org/api_docs/python/tf/py_function
    .. [2] https://www.tensorflow.org/guide/data
    """
    # use partial to pass read_dataset_info
    par = partial(read_dataset, read_dataset_info=read_dataset_info)
    # wrapping par function with py_function
    [resistance, resistivity_log10] = tf.py_function(
        par, [file_path], [tf.float32, tf.float32]
    )
    # set tensor shape
    resistance.set_shape(read_dataset_info['input_shape'])
    resistivity_log10.set_shape(read_dataset_info['output_shape'])
    return resistance, resistivity_log10
