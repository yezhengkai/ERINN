"""Utilities related to disk I/O."""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import gc
import json
import os
import pickle
import re
import shutil
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Any

import h5py
import numpy as np
from ruamel.yaml import YAML
from tensorflow.keras.utils import plot_model

from erinn.utils.data_utils import prepare_for_get_2_5Dpara
from erinn.utils.time_utils import datetime_in_range
from erinn.utils.time_utils import datetime_range


def read_pkl(pkl: Union[str, Path]) -> Any:
    """
    Read pickle file.

    Parameters
    ----------
    pkl : str or pathlib.Path
    The path od pickle file.

    Returns
    -------
    obj : Any
    Restored object.
    """
    with open(pkl, "rb") as f:
        obj = pickle.load(f)
        return obj


def write_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_config_file(config_file: Union[str, Path, dict]) -> dict:
    if isinstance(config_file, dict):
        config = config_file
    elif isinstance(config_file, (str, Path)):
        # use SafeLoader/SafeDumper. Loading of a document without resolving unknown tags.
        yaml = YAML(typ='safe')
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.load(f)
    else:
        raise TypeError('Please input string or dictionary.')

    return config


def read_urf(urf_file: Union[str, Path]) -> dict:
    """
    Read urf file.

    Parameters
    ----------
    urf_file : str or pathlib.Path
        urf file path.

    Returns
    -------
    urf : dict
       Dictionary contains the information of urf file.

        * Tx_id : numpy.ndarray (1, t)
             Transmitter electrodes number.
        * Rx_id : numpy.ndarray (1, r)
             Receiver electrodes number.
        * RxP2_id : numpy.ndarray (1, r2)
             Common reference potential electrode number.
        * coord : numpy.ndarray (m, 4)
             The first column is the id of the electrode and
             the remaining columns are the 3-dimensional coordinates of the electrode.
        * data : numpy.ndarray (d, 7)
             Measurements of specific electrode dipole-dipole.

    Notes
    -----
    Please refer to the instruction manual of AGI EarthImager 2D.

    References
    ----------
    .. [1] Advanced Geosciences, Inc. (2009).
           Instruction Manual for EarthImager 2D, Version 2.4.0, Resistivity and IP Inversion Software.
    """

    with open(urf_file, encoding='utf-8') as f:
        # predefine output variables
        urf_info = {'Tx_id': None,
                    'Rx_id': None,
                    'RxP2_id': None,
                    'coord': None,
                    'data': None}
        for line in f:
            line = line.strip()
            if line == 'Tx':
                line = f.readline().strip()
                urf_info['Tx_id'] = np.array(
                    line.split(sep=','), dtype=np.int64, ndmin=2)
            elif line == 'Rx':
                line = f.readline().strip()
                urf_info['Rx_id'] = np.array(
                    line.split(sep=','), dtype=np.int64, ndmin=2)
            elif line == 'RxP2':
                line = f.readline().strip()
                if line != '':
                    urf_info['RxP2_id'] = np.array(line.split(sep=','),
                                                   dtype=np.int64, ndmin=2)
            elif line.startswith(':Geometry'):
                if urf_info['RxP2_id'] is None:
                    num_line = int(np.nanmax(
                        np.concatenate((urf_info['Tx_id'],
                                        urf_info['Rx_id']), axis=1)
                        )
                    )
                else:
                    num_line = int(np.nanmax(
                        np.concatenate((urf_info['Tx_id'],
                                        urf_info['Rx_id'],
                                        urf_info['RxP2_id']), axis=1)
                        )
                    )
                line = [f.readline().strip().split(',')
                        for _ in range(num_line)]
                urf_info['coord'] = np.array(line, dtype=np.float64, ndmin=2)
            elif line.startswith(':Measurements'):
                line = list(map(lambda l: l.strip().split(','), f.readlines()))
                urf_info['data'] = np.array(line, dtype=np.float64, ndmin=2)
    return urf_info


def write_urf(urf_file: Union[str, Path], urf_info: dict):
    """
    Write urf file.

    Parameters
    ----------
    urf_file : str or pathlib.Path
        urf file path.
    urf_info : dict
       Dictionary contains the information of urf file.

        * Tx_id : numpy.ndarray (1, t)
             Transmitter electrodes number.
        * Rx_id : numpy.ndarray (1, r)
             Receiver electrodes number.
        * RxP2_id : numpy.ndarray (1, r2)
             Common reference potential electrode number.
        * coord : numpy.ndarray (m, 4)
             The first column is the id of the electrode and
             the remaining columns are the 3-dimensional coordinates of the electrode.
        * data : numpy.ndarray (d, 7)
             Measurements of specific electrode dipole-dipole.

    Returns
    -------
    None

    Notes
    -----
    Please refer to the instruction manual of AGI EarthImager 2D.

    References
    ----------
    .. [1] Advanced Geosciences, Inc. (2009).
           Instruction Manual for EarthImager 2D, Version 2.4.0, Resistivity and IP Inversion Software.
    """
    with open(urf_file, mode='wt', encoding='utf-8') as f:
        keys = ['Tx_id', 'Rx_id', 'RxP2_id', 'coord', 'data']
        for i, key in enumerate(keys):
            value = urf_info.get(key)
            if key == 'Tx_id':
                f.write('Tx\n')
                if value is None:
                    np.savetxt(f, np.array([''], dtype=str), fmt="%s")
                else:
                    np.savetxt(f, value, delimiter=', ', fmt="%d")
            elif key == 'Rx_id':
                f.write('Rx\n')
                if value is None:
                    np.savetxt(f, np.array([''], dtype=str), fmt="%s")
                else:
                    np.savetxt(f, value, delimiter=', ', fmt="%d")
            elif key == 'RxP2_id':
                f.write('RxP2\n')
                if value is None:
                    np.savetxt(f, np.array([''], dtype=str), fmt="%s")
                else:
                    np.savetxt(f, value, delimiter=', ', fmt="%d")
                f.write('\n')
            elif key == 'coord':
                f.write(':Geometry\n')
                if value is None:
                    np.savetxt(f, np.array([''], dtype=str), fmt="%s")
                else:
                    np.savetxt(f, value, fmt="%2d, %10.6f, %10.6f, %10.6f")
                f.write('\n')
            elif key == 'data':
                f.write(':Measurements\n')
                if value is None:
                    np.savetxt(f, np.array([''], dtype=str), fmt="%s")
                else:
                    try:
                        np.savetxt(f, value, fmt="%2d, %2d, %2d, %2d, "
                                                 + "%11.6f, %11.6f, %11.6f")
                    except ValueError as e:
                        np.savetxt(f, value, fmt="%2d, %2d, %2d, %2d, "
                                                 + "%11.6f, %11.6f, %11.6f, %11.6f")


class URF(object):

    def __init__(self,
                 urf_file=None,
                 get_abmn_locations=False,
                 get_geometric_factor=False,
                 survey_type='dipole-dipole',
                 dimension=2,
                 space_type='half-space'):
        """
        Construct an instance from a urf file.

        Parameters
        ----------
        urf_file : str or pathlib.Path
            urf file path.
        get_abmn_locations : bool
            Whether to get A, B, M and N locations.
        get_geometric_factor : bool
            Whether to calculate geometric factor.
        survey_type : str
            Accepted parameters are "dipole-dipole", "pole-dipole", "dipole-pole" and "pole-pole".
        dimension : int
            Accepted parameters are "2" and "3".
        space_type : str
            Accepted parameters are "whole-space" and "half-space".
        """

        self.urf_info = None
        self.id2coord_map = None
        self.a_locations = None
        self.b_locations = None
        self.m_locations = None
        self.n_locations = None
        self.abmn_locations = None
        self.geometric_factor = None
        self.survey_type = survey_type
        self.dimension = dimension
        self.space_type = space_type

        if urf_file is not None:
            # read urf
            self.read(urf_file)
            # get information from data
            if get_abmn_locations:
                self.get_abmn_locations()
            # get geometry factor
            if get_geometric_factor:
                self.get_geometric_factor()

    @property
    def Tx_id(self):
        """Tx part in urf."""
        return self.urf_info['Tx_id']

    @Tx_id.setter
    def Tx_id(self, value):
        """Set Tx part in urf."""
        self.urf_info['Tx_id'] = value

    @property
    def Rx_id(self):
        """Rx part in urf."""
        return self.urf_info['Rx_id']

    @Rx_id.setter
    def Rx_id(self, value):
        """Set Rx part in urf."""
        self.urf_info['Rx_id'] = value

    @property
    def RxP2_id(self):
        """RxP2 part in urf."""
        return self.urf_info['RxP2_id']

    @RxP2_id.setter
    def RxP2_id(self, value):
        """Set RxP2 part in urf."""
        self.urf_info['RxP2_id'] = value

    @property
    def coord(self):
        """Geometry part in urf."""
        return self.urf_info['coord']

    @coord.setter
    def coord(self, value):
        """Set Geometry part in urf."""
        self.urf_info['coord'] = value

    @property
    def data(self):
        """Measurements part in urf."""
        return self.urf_info['data']

    @data.setter
    def data(self, value):
        """Set Measurements part in urf."""
        value = np.asanyarray(value)
        if value.ndim == 2:
            self.urf_info['data'] = value
        else:
            warnings.warn(
                "The ndim of assigned value is not equal to 2", UserWarning)
            self.urf_info['data'] = value

    @property
    def a_id(self):
        """Id of A electrode"""
        if self.data.ndim == 2 and np.all(np.greater(self.data.shape, (0, 0))):
            return self.data[:, 0].astype(np.int64)
        else:
            warnings.warn("URF.data has a ndim that is not equal to 2, or does not contain A column.",
                          UserWarning)
            return None

    @property
    def b_id(self):
        """Id of B electrode"""
        if self.data.ndim == 2 and np.all(np.greater(self.data.shape, (0, 1))):
            return self.data[:, 1].astype(np.int64)
        else:
            warnings.warn("URF.data has a ndim that is not equal to 2, or does not contain B column.",
                          UserWarning)
            return None

    @property
    def m_id(self):
        """Id of M electrode"""
        if self.data.ndim == 2 and np.all(np.greater(self.data.shape, (0, 2))):
            return self.data[:, 2].astype(np.int64)
        else:
            warnings.warn("URF.data has a ndim that is not equal to 2, or does not contain M column.",
                          UserWarning)
            return None

    @property
    def n_id(self):
        """Id of N electrode"""
        if self.data.ndim == 2 and np.all(np.greater(self.data.shape, (0, 3))):
            return self.data[:, 3].astype(np.int64)
        else:
            warnings.warn("URF.data has a ndim that is not equal to 2, or does not contain N column.",
                          UserWarning)
            return None

    @property
    def abmn_id(self):
        """Id of ABMN electrode"""
        if self.data.ndim == 2 and np.all(np.greater(self.data.shape, (0, 3))):
            return self.data[:, :4].astype(np.int64)
        else:
            warnings.warn("URF.data has a ndim that is not equal to 2, or does not contain A, B, M and N columns.",
                          UserWarning)
            return None

    @property
    def resistance(self):
        """V/I (ohm)"""
        if self.data.ndim == 2 and np.all(np.greater(self.data.shape, (0, 4))):
            return self.data[:, 4]
        else:
            warnings.warn("URF.data has a ndim that is not equal to 2, or does not contain V/I column.",
                          UserWarning)
            return None

    @property
    def I(self):
        """Electric current (mA)"""
        if self.data.ndim == 2 and np.all(np.greater(self.data.shape, (0, 5))):
            return self.data[:, 5]
        else:
            warnings.warn("URF.data has a ndim that is not equal to 2, or does not contain I column.",
                          UserWarning)
            return None

    @property
    def error(self):
        """Error (%)"""
        if self.data.ndim == 2 and np.all(np.greater(self.data.shape, (0, 6))):
            return self.data[:, 6]
        else:
            warnings.warn("URF.data has a ndim that is not equal to 2, or does not contain Error column.",
                          UserWarning)
            return None

    @property
    def num_data(self):
        """Number of data"""
        if self.data.ndim == 2:
            return self.data.shape[0]
        else:
            warnings.warn(
                "The ndim of URF.data is not equal to 2.", UserWarning)
            return None

    def get_abmn_locations(self):
        """
        Get electrode locations (Unit: m).

        Returns
        -------
        None
        """

        # form an dict that map id to coord
        if self.dimension == 2:
            self.id2coord_map = dict(
                zip(self.coord[:, 0], self.coord[:, 1::2]))
        elif self.dimension == 3:
            self.id2coord_map = dict(zip(self.coord[:, 0], self.coord[:, 1:]))
        else:
            raise NotImplementedError()

        if not np.any(np.array_equal(self.data, np.array([[np.nan]]))):
            try:
                self.a_locations = np.vstack([self.id2coord_map[i]
                                              for i in self.a_id])
                self.b_locations = np.vstack([self.id2coord_map[i]
                                              for i in self.b_id])
                self.m_locations = np.vstack([self.id2coord_map[i]
                                              for i in self.m_id])
                self.n_locations = np.vstack([self.id2coord_map[i]
                                              for i in self.n_id])
            except TypeError as e:
                e.args = (
                    e.args[0] + '. Because the id of a, b, m or n is None',)
                raise e
            self.abmn_locations = np.hstack((self.a_locations, self.b_locations,
                                             self.m_locations, self.n_locations))
        else:
            raise Exception('Nan in urf file!')

    def get_geometric_factor(self):
        """
        Get geometric factor.
        geometric_factor = (space_fact * pi) / g

        For 'space_fact':
            `whole-space` : 4
            `half-space` : 2
        For 'g':
            `dipole-dipole` : 1 / AM - 1 / BM - 1 / AN + 1 / BN
            `pole-dipole` : 1 / am - 1 / an
            `dipole-pole` : 1 / am - 1 / bm
            `pole-pole` : 1 / am

        Returns
        -------
        None
        """

        # Set factor for whole-space or half-space assumption
        if self.space_type == 'whole-space':
            space_fact = 4.
        elif self.space_type == 'half-space':
            space_fact = 2.
        else:
            raise Exception(
                """'space_type must be 'whole-space' | 'half-space'""")

        elec_sep_dict = self.electrode_separations(self.a_locations, self.b_locations,
                                                   self.m_locations, self.n_locations,
                                                   electrode_pair=['AM', 'BM',
                                                                   'AN', 'BN'])
        try:
            am = elec_sep_dict['AM']
            bm = elec_sep_dict['BM']
            an = elec_sep_dict['AN']
            bn = elec_sep_dict['BN']
        except KeyError as e:
            e.args = ('`' + e.args[0] + '` Because ABMN locations is nan!',)
            raise e

        # Determine geometric factor G based on electrode separation distances
        if self.survey_type == 'dipole-dipole':
            g = 1 / am - 1 / bm - 1 / an + 1 / bn
        elif self.survey_type == 'pole-dipole':
            g = 1 / am - 1 / an
        elif self.survey_type == 'dipole-pole':
            g = 1 / am - 1 / bm
        elif self.survey_type == 'pole-pole':
            g = 1 / am
        else:
            raise Exception("""survey_type must be 'dipole-dipole' | 'pole-dipole' |
            'dipole-pole' | 'pole-pole'"""
                            " not {}".format(self.survey_type))

        self.geometric_factor = (space_fact * np.pi) / g

    @staticmethod
    def electrode_separations(a_locations=np.nan, b_locations=np.nan,
                              m_locations=np.nan, n_locations=np.nan,
                              electrode_pair='All'):
        """
        Get electrode separations in specific electrode pair.

        Parameters
        ----------
        a_locations : np.ndarray
            Locations of A electrodes.
        b_locations : np.ndarray
            Locations of B electrodes.
        m_locations : np.ndarray
            Locations of M electrodes.
        n_locations : np.ndarray
            Locations of N electrodes.
        electrode_pair : str or list of str
            The electrode pair for which the distance should be calculated.
            Accepted parameters are "AB", "MN", "AM", "AN", "BM", "BN" and "ALL".

        Returns
        -------
        sep_dict : dict
            Electrodes separation distances.
        """

        if electrode_pair == 'All':
            electrode_pair = np.r_[['AB', 'MN', 'AM', 'AN', 'BM', 'BN']]
        elif isinstance(electrode_pair, list) or isinstance(electrode_pair, str):
            electrode_pair = np.r_[electrode_pair]
        else:
            raise Exception(
                """electrode_pair must be either a string,
                list of strings""" " not {}".format(type(electrode_pair))
            )

        sep_dict = {}
        if np.any(electrode_pair == 'AB'):
            ab = np.sqrt(np.sum((a_locations - b_locations) ** 2, axis=1))
            if not np.all(np.isnan(ab)):
                sep_dict['AB'] = ab

        if np.any(electrode_pair == 'MN'):
            mn = np.sqrt(np.sum((m_locations - n_locations) ** 2, axis=1))
            if not np.all(np.isnan(mn)):
                sep_dict['MN'] = mn

        if np.any(electrode_pair == 'AM'):
            am = np.sqrt(np.sum((a_locations - m_locations) ** 2, axis=1))
            if not np.all(np.isnan(am)):
                sep_dict['AM'] = am

        if np.any(electrode_pair == 'AN'):
            an = np.sqrt(np.sum((a_locations - n_locations) ** 2, axis=1))
            if not np.all(np.isnan(an)):
                sep_dict['AN'] = an

        if np.any(electrode_pair == 'BM'):
            bm = np.sqrt(np.sum((b_locations - m_locations) ** 2, axis=1))
            if not np.all(np.isnan(bm)):
                sep_dict['BM'] = bm

        if np.any(electrode_pair == 'BN'):
            bn = np.sqrt(np.sum((b_locations - n_locations) ** 2, axis=1))
            if not np.all(np.isnan(bn)):
                sep_dict['BN'] = bn

        return sep_dict

    def read(self, urf_file):
        """
        Read urf file.

        Parameters
        ----------
        urf_file : str or pathlib.Path
            urf file path.

        Returns
        -------
        None
        """
        self.urf_info = read_urf(urf_file)

    def write(self, urf_file):
        """
        Write urf file.

        Parameters
        ----------
        urf_file : str or pathlib.Path
            urf file path.
        Returns
        -------
        None
        """
        write_urf(urf_file, self.urf_info)


def get_pkl_list(dir_path, limit=None, sort=True, sort_key=None):
    if not os.path.exists(dir_path):
        warnings.warn(f'No such file or directory: {dir_path}, '
                      'return an empty list', UserWarning)
        return []

    if limit:
        # enumerate is really just a fancy generator:
        enum_generator = enumerate(_list_generator(dir_path, ext='.pkl'))
        file_list = [os.path.join(dir_path, file) for i, file in enum_generator
                     if i < limit]
    else:
        file_list = [os.path.join(dir_path, file)
                     for file in _list_generator(dir_path, ext='.pkl')]

    if sort:
        return sorted(file_list, key=sort_key)
    else:
        return file_list


def _list_generator(dir_path, ext='.npz'):
    # return a generator
    # return (file for file in os.listdir(dir_path)
    # if os.path.isfile(os.path.join(dir_path, file)) and file.endswith(ext))

    # function is a generator?
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file)) and file.endswith(ext):
            yield file
