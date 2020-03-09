"""Utilities for manipulating data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import itertools
import json
import re

import h5py
import numpy as np
import numpy.matlib


def scan_hdf5(h5, grp='/', recursive=True, tab_step=2):
    """Recursively print the path of hdf5.

    Parameters
    ----------
    h5 : str
        Hdf5 file path.
    grp : str, default '/'
        The group path of hdf5 to be searched. Default is root group.
    recursive : bool, default True
        Whether to recursively search for groups of hdf5.
    tab_step : int, default 2
        Number of white space for alignment.

    Returns
    -------
    None

    References
    ----------
    https://stackoverflow.com/questions/43371438/how-to-inspect-h5-file-in-python
    """

    def scan_node(g, tabs=0):
        print(' ' * tabs, g.name)
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                print(' ' * tabs + ' ' * tab_step + ' -', v.name)
            elif isinstance(v, h5py.Group) and recursive:
                scan_node(v, tabs=tabs + tab_step)
    with h5py.File(h5, 'r') as f:
        scan_node(f[grp])


def search_hdf5(h5, pattern, grp='/', recursive=True):
    """Search for dataset paths of hdf5.

    Search for dataset paths of hdf5 that match
    a specific regular expression pattern.

    Parameters
    ----------
    h5 : str
        hdf5 file path.
    pattern :　compiled regular expression object
        Particular string in the dataset paths of hdf5 to be searched.
    grp : str, default '/'
        The group path of hdf5 to be searched. Default is root group.
    recursive : bool, default True
        Whether to recursively search for groups of hdf5.

    Retruns
    -------
    match_list : list
        A list contains the dataset paths of hdf5
        that matches the given pattern
    """

    match_list = []

    def scan_node(g):
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                if re.search(pattern, v.name):
                    match_list.append(v.name)
            elif isinstance(v, h5py.Group) and recursive:
                scan_node(v)
    with h5py.File(h5, 'r') as f:
        scan_node(f[grp])
    return match_list


# TODO: prepare_for_get_2_5Dpara 修改或是刪掉
def prepare_for_get_2_5Dpara(config_json, return_urf=False):
    """Generate essential variables for matlab function get_2_5Dpara().

    Parameters
    ----------
    config_json : str

    return_urf : bool, default False

    Returns
    -------
    srcloc : numpy.ndarray

    dx : numpy.ndarray

    dz : numpy.ndarray
    recloc : numpy.ndarray
    srcnum : numpy.ndarray

    """
    from .io_utils import read_urf

    with open(config_json) as f:
        config = json.load(f)

    urf = config['simulate']['geomatry_urf']
    Tx_id, Rx_id, RxP2_id, coord, data = read_urf(urf)
    if data.size == 0:
        C_pair = np.array(list(itertools.combinations(Tx_id.flatten(), 2)))
        P_pair = np.array(list(itertools.combinations(Rx_id.flatten(), 2)))
        CP_pair = np.hstack((np.repeat(C_pair, P_pair.shape[0], axis=0),
                             np.matlib.repmat(P_pair, C_pair.shape[0], 1)))
    else:
        CP_pair = data[:, 0:4]

    recloc = np.hstack(
        (coord[CP_pair[:, 2].T - 1, 1:4:2], coord[CP_pair[:, 3].T - 1, 1:4:2]))
    recloc[:, 1:4:2] = np.abs(recloc[:, 1:4:2])
    srcloc = np.hstack(
        (coord[CP_pair[:, 0].T - 1, 1:4:2], coord[CP_pair[:, 1].T - 1, 1:4:2]))
    srcloc, srcnum = np.unique(srcloc, return_inverse=True, axis=0)
    srcnum = np.reshape(srcnum, (-1, 1)) + 1  # matlab index is from 1

    array_len = max(coord[:, 1]) - min(coord[:, 1])
    srcloc[:, [0, 2]] = srcloc[:, [0, 2]] - array_len/2
    recloc[:, [0, 2]] = recloc[:, [0, 2]] - array_len/2
    dx = np.ones((config['nx'], 1))
    dz = np.ones((config['nz'], 1))

    if return_urf:
        return [[srcloc, dx, dz, recloc, srcnum],
                [Tx_id, Rx_id, RxP2_id, coord, data]]
    else:
        return srcloc, dx, dz, recloc, srcnum
