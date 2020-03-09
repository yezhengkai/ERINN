"""Utilities related to disk I/O."""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import gc
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

from erinn.preprocessing import log_transform
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
                urf_info['Tx_id'] = np.array(line.split(sep=','), dtype=np.int64, ndmin=2)
            elif line == 'Rx':
                line = f.readline().strip()
                urf_info['Rx_id'] = np.array(line.split(sep=','), dtype=np.int64, ndmin=2)
            elif line == 'RxP2':
                line = f.readline().strip()
                if line != '':
                    urf_info['RxP2_id'] = np.array(line.split(sep=','),
                                                   dtype=np.int64, ndmin=2)
            elif line.startswith(':Geometry'):
                if urf_info['RxP2_id'] is None:
                    num_line = int(np.nanmax(np.concatenate((urf_info['Tx_id'],
                                                             urf_info['Rx_id']), axis=1)))
                else:
                    num_line = int(np.nanmax(np.concatenate((urf_info['Tx_id'],
                                                             urf_info['Rx_id'],
                                                             urf_info['RxP2_id']), axis=1)))
                line = [f.readline().strip().split(',') for _ in range(num_line)]
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
            warnings.warn("The ndim of assigned value is not equal to 2", UserWarning)
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
        """Error(%)"""
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
            warnings.warn("The ndim of URF.data is not equal to 2.", UserWarning)
            return None

    def get_abmn_locations(self):
        """
        Get electrode locations.

        Returns
        -------
        None
        """

        # form an dict that map id to coord
        if self.dimension == 2:
            self.id2coord_map = dict(zip(self.coord[:, 0], self.coord[:, 1::2]))
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
                e.args = (e.args[0] + '. Because the id of a, b, m or n is None',)
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
            raise Exception("""'space_type must be 'whole-space' | 'half-space'""")

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


def read_raw_data(data_list):
    """Load Inputs/Targets data from hdf5 file.

    Parameters
    ----------
    data_list : str or list of string
        Path of hdf5 file, which file contains datasets
        named 'Inputs' and Targets'.
        You can load multiple files at once by
        entering a path list.

    Returns
    -------
    x, y : numpy.ndarray
        Input and output data for NN.
    """
    if (not isinstance(data_list, list) and
            isinstance(data_list, str)):
        data_list = [data_list]
    elif not isinstance(data_list[0], str):
        raise ValueError(
            '{} not a string or a list of string'.format(data_list))

    x = np.array([])
    y = np.array([])
    for path in data_list:
        print('Loading', path)
        data = h5py.File(path)
        x_tmp = data['Inputs'][:, :].T.astype('float32')
        y_tmp = np.log10(1 / data['Targets'][:, :].T).astype('float32')
        x = np.vstack([x, x_tmp]) if x.size else x_tmp
        y = np.vstack([y, y_tmp]) if y.size else y_tmp
        print('Load', len(y_tmp), 'Samples.', 'Total Samples:', len(y))
        del x_tmp, y_tmp
        data.close()
        gc.collect()
        print('Done.')
    # data.close()

    return x, y


def write_training_npz(glob_para_h5, h5_list, npz_dir, shape='1d',
                       transform_inputs=False, remove_dir=False):
    """Create processed data for training and save as npz file.

    Parameters
    ----------
    glob_para_h5 : str
        Path of the hdf5 file that contain a group `/glob_para`.
    h5_list : list
        List of string that contain the paths of the hdf5 files.
    npz_dir : str

    shape : str

    transform_inputs : bool

    remove_dir : bool

    Returns
    -------
    None

    References
    ----------
    https://github.com/jimmy60504/SeisNN/blob/master/seisnn/io.py
    https://stackoverflow.com/questions/17223301/python-multiprocessing-is-it-possible-to-have-a-pool-inside-of-a-pool
    """
    if (not isinstance(h5_list, list) and
            isinstance(h5_list, str)):
        h5_list = [h5_list]
    elif not isinstance(h5_list[0], str):
        raise ValueError(
            '{} not a string or a list of string'.format(h5_list))
    if remove_dir:
        shutil.rmtree(npz_dir, ignore_errors=True)
    os.makedirs(npz_dir, exist_ok=True)

    config = h5py.File(glob_para_h5)
    # num_features = config['glob_para']['recloc'].shape[1]
    num_Tx_id = config['glob_para']['Tx_id'].shape[0]
    num_Rx_id = config['glob_para']['Rx_id'].shape[0]
    num_Tx = int(0.5 * (num_Tx_id * (num_Tx_id - 1)))
    num_Rx = int(0.5 * (num_Rx_id * (num_Rx_id - 1)))
    nx = int(config['glob_para']['nx'][()])
    nz = int(config['glob_para']['nz'][()])
    config.close()

    for h5 in h5_list:
        print(f'Convert {h5} data to npz.')
        with h5py.File(h5) as data:
            num_samples = data['Inputs'].shape[1]
        with Pool() as pool:
            par = partial(_write_training_npz, h5=h5, npz_dir=npz_dir, shape=shape,
                          num_Tx=num_Tx, num_Rx=num_Rx, nz=nz, nx=nx, transform_inputs=transform_inputs)
            pool.map_async(par, range(0, num_samples))
            pool.close()
            pool.join()
        gc.collect()

    # Serial version
    # for h5 in h5_list:
    #     data = h5py.File(h5)
    #     num = 0
    #     num_samples = data['Inputs'].shape[1]
    #     print(f'Convert {h5} data to npz.')
    #     for i in range(num_samples):
    #         if shape == '1d':
    #             # all array
    #             # reshape
    #             Inputs = data['Inputs'][:, i].reshape(-1).astype('float32')
    #             # take reciprocal, convert to log10 scale and reshape
    #             Targets = np.log10(1 / data['Targets'][:, i]).reshape(-1).astype('float32')
    #         elif shape == '2d':
    #             # for CPP array
    #             Inputs = data['Inputs'][:, i].reshape(num_Tx, num_Rx).astype('float32')
    #             Targets = np.log10(1 / data['Targets'][:, i]).reshape(nz, nx).astype('float32')
    #         elif shape == '3d':
    #             # for CPP array
    #             Inputs = data['Inputs'][:, i].reshape(num_Tx, num_Rx, 1).astype('float32')
    #             Targets = np.log10(1 / data['Targets'][:, i]).reshape(nz, nx, 1).astype('float32')
    #         else:
    #             raise ValueError(
    #                 "The value shape must be '1d', '2d' or '3d'.")
    #         filename = os.path.splitext(os.path.basename(path))[0] + '_' + str(num) + '.npz'
    #         outfile = os.path.join(npz_dir, filename)
    #         np.savez_compressed(outfile, Inputs=Inputs, Targets=Targets)
    #         num = num + 1
    #     data.close()
    #     gc.collect()


def _write_training_npz(idx, h5, npz_dir, shape, num_Tx, num_Rx, nz, nx, transform_inputs):
    with h5py.File(h5) as data:
        Inputs = data['Inputs'][:, idx]
        Targets = data['Targets'][:, idx]
        if transform_inputs:
            log_transform(Inputs, inplace=True)

        if shape == '1d':
            # all array
            # reshape
            Inputs = Inputs.reshape(-1).astype('float32')
            # Inputs = data['Inputs'][:, idx].reshape(-1).astype('float32')
            # take reciprocal(sigma to rho), convert to log10 scale and reshape
            Targets = np.log10(1 / Targets).reshape(-1).astype('float32')
            # Targets = np.log10(1 / data['Targets'][:, idx]).reshape(-1).astype('float32')
        elif shape == '2d':
            # for CPP array
            Inputs = Inputs.reshape(num_Tx, num_Rx).astype('float32')
            # Inputs = data['Inputs'][:, idx].reshape(num_Tx, num_Rx).astype('float32')
            Targets = np.log10(1 / Targets).reshape(nz, nx).astype('float32')
            # Targets = np.log10(1 / data['Targets'][:, idx]).reshape(nz, nx).astype('float32')
        elif shape == '3d':
            # for CPP array
            Inputs = Inputs.reshape(num_Tx, num_Rx, 1).astype('float32')
            # Inputs = data['Inputs'][:, idx].reshape(num_Tx, num_Rx, 1).astype('float32')
            Targets = np.log10(1 / Targets).reshape(nz, nx, 1).astype('float32')
            # Targets = np.log10(1 / data['Targets'][:, idx]).reshape(nz, nx, 1).astype('float32')
        else:
            raise ValueError(
                "The value shape must be '1d', '2d' or '3d'.")
        filename = os.path.splitext(os.path.basename(data.filename))[0] + '_' + str(idx) + '.npz'
        outfile = os.path.join(npz_dir, filename)
        print(f'Write {outfile}')
        np.savez_compressed(outfile, Inputs=Inputs, Targets=Targets)


def get_npz_list(dir_path, limit=None):
    """

    References
    ----------
    https://github.com/jimmy60504/SeisNN/blob/master/seisnn/io.py
    https://stackoverflow.com/questions/3396279/enumerate-ing-a-generator-in-python
    https://stackoverflow.com/questions/311775/python-create-a-list-with-initial-capacity
    http://blog.cdleary.com/2010/04/efficiency-of-list-comprehensions/
    https://stackoverflow.com/questions/22225666/pre-allocating-a-list-of-none
    """

    if limit:
        # enumerate is really just a fancy generator:
        enum_generator = enumerate(_list_generator(dir_path))
        file_list = [os.path.join(dir_path, file) for i, file in enum_generator
                     if i < limit]
        # file_list = limit * [None]
        # enumerate is really just a fancy generator:
        # enum_generator = enumerate(_list_generator(dir_path))
        # for i, file in enum_generator:
        # file_list[i] = os.path.join(dir_path, file)
    else:
        file_list = [os.path.join(dir_path, file) for file in _list_generator(dir_path)]
        # for file in _list_generator(dir_path):
        # file_list.append(os.path.join(dir_path, file))

    return file_list


def get_pkl_list(dir_path, limit=None):
    if limit:
        # enumerate is really just a fancy generator:
        enum_generator = enumerate(_list_generator(dir_path, ext='.pkl'))
        file_list = [os.path.join(dir_path, file) for i, file in enum_generator
                     if i < limit]
    else:
        file_list = [os.path.join(dir_path, file) for file in _list_generator(dir_path, ext='.pkl')]

    return file_list


def _list_generator(dir_path, ext='.npz'):
    # return a generator
    # return (file for file in os.listdir(dir_path)
    # if os.path.isfile(os.path.join(dir_path, file)) and file.endswith(ext))

    # function is a generator?
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file)) and file.endswith(ext):
            yield file


# TODO: gen_glob_para_h5修改或刪除
def gen_glob_para_h5(config_file, output_h5):
    """This function is for FW2_5D
    """
    [[srcloc, dx, dz, recloc, srcnum], [Tx_id, Rx_id, RxP2_id, coord, _]
     ] = prepare_for_get_2_5Dpara(config_file, return_urf=True)

    array_len = max(coord[:, 1]) - min(coord[:, 1])
    coord_in_model = (coord - [0, array_len / 2, 0, 0])

    with h5py.File(name=output_h5, mode='a') as f:
        # save Tx_id
        try:
            dset = f.create_dataset(
                '/glob_para/Tx_id', data=Tx_id.T, chunks=True)
            print('write {}: {}'.format(output_h5, dset.name))
        except:
            dset = f['/glob_para/Tx_id']
            dset[:] = Tx_id.T
            print('write {}: {}'.format(output_h5, dset.name))
        # save Rx_id
        try:
            dset = f.create_dataset(
                '/glob_para/Rx_id', data=Rx_id.T, chunks=True)
            print('write {}: {}'.format(output_h5, dset.name))
        except:
            dset = f['/glob_para/Rx_id']
            dset[:] = Rx_id.T
            print('write {}: {}'.format(output_h5, dset.name))
        # save RxP2_id
        try:
            dset = f.create_dataset(
                '/glob_para/RxP2_id', data=RxP2_id.T, chunks=True)
            print('write {}: {}'.format(output_h5, dset.name))
        except:
            dset = f['/glob_para/RxP2_id']
            dset[:] = RxP2_id.T
            print('write {}: {}'.format(output_h5, dset.name))
        # save electrode coord (id, x, y, z)
        try:
            dset = f.create_dataset(
                '/glob_para/coord', data=coord.T, chunks=True)
            print('write {}: {}'.format(output_h5, dset.name))
        except:
            dset = f['/glob_para/coord']
            dset[:] = coord.T
            print('write {}: {}'.format(output_h5, dset.name))
        # save electrode coord in forward model(id, x, y, z)
        try:
            dset = f.create_dataset(
                '/glob_para/coord_in_model', data=coord_in_model.T, chunks=True)
            print('write {}: {}'.format(output_h5, dset.name))
        except:
            dset = f['/glob_para/coord_in_model']
            dset[:] = coord_in_model.T
            print('write {}: {}'.format(output_h5, dset.name))
        # save srcloc
        try:
            dset = f.create_dataset(
                '/glob_para/srcloc', data=srcloc.T, chunks=True)
            print('write {}: {}'.format(output_h5, dset.name))
        except:
            dset = f['/glob_para/srcloc']
            dset[:] = srcloc.T
            print('write {}: {}'.format(output_h5, dset.name))
        # save dx
        try:
            dset = f.create_dataset(
                '/glob_para/dx', data=dx.T, chunks=True)
            print('write {}: {}'.format(output_h5, dset.name))
        except:
            dset = f['/glob_para/dx']
            dset[:] = dx.T
            print('write {}: {}'.format(output_h5, dset.name))
        # save dz
        try:
            dset = f.create_dataset(
                '/glob_para/dz', data=dz.T, chunks=True)
            print('write {}: {}'.format(output_h5, dset.name))
        except:
            dset = f['/glob_para/dz']
            dset[:] = dz.T
            print('write {}: {}'.format(output_h5, dset.name))
        # save recloc
        try:
            dset = f.create_dataset(
                '/glob_para/recloc', data=recloc.T, chunks=True)
            print('write {}: {}'.format(output_h5, dset.name))
        except:
            dset = f['/glob_para/recloc']
            dset[:] = recloc.T
            print('write {}: {}'.format(output_h5, dset.name))
        # save srcnum
        try:
            dset = f.create_dataset(
                '/glob_para/srcnum', data=srcnum.T, chunks=True)
            print('write {}: {}'.format(output_h5, dset.name))
        except:
            dset = f['/glob_para/srcnum']
            dset[:] = srcnum.T
            print('write {}: {}'.format(output_h5, dset.name))


# TODO: 要不要保留 fit_generator?
def save_synth_data(model, src_h5, npz_list, data_generator=None, dest_h5=None, inverse_trans=False):
    """
    Save synthetic data.

    The synthetic data to be checked is saved to the hdf5 file, and then the crossplot of
    the synthetic V/I and predictive V/I can be used to check the robustness of the NN model.

    Parameters
    ----------
    model : Instance of `keras Model`
        The neural network model for prediction.
    src_h5 : str
        hdf5 file contain forward modeling parameters in the /glob_para group.
    npz_list : list
        List contains the path of npz file that have `Inputs` and `Targets` data.
    dest_h5 : str, default None
        Copy hdf5 file to new destination. The default is to use src_h5 as dest_h5.
    inverse_trans : bool, default False

    Returns
    -------
    None

    References
    ----------
    https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py
    https://stackoverflow.com/questions/25655588/incremental-writes-to-hdf5-with-h5py?rq=1
    http://download.nexusformat.org/sphinx/examples/h5py/index.html
    """

    dest_h5 = src_h5 if dest_h5 is None else dest_h5
    os.makedirs(os.path.dirname(dest_h5), exist_ok=True)

    tmp = np.load(npz_list[0])
    num_samples = len(npz_list)
    num_features = tmp['Inputs'].size
    num_output_elem = tmp['Targets'].size
    input_shape = model.layers[1].input_shape[1:]
    synth_V = np.empty((num_samples, num_features))
    synth_log_rho = np.empty((num_samples, num_output_elem))
    pred_log_rho = np.empty((num_samples, num_output_elem))

    for i in range(num_samples):
        data = np.load(npz_list[i])
        synth_V[i,] = data['Inputs'].reshape(-1).copy()
        synth_log_rho[i,] = data['Targets'].reshape(-1).copy()
        tmp = model.predict(synth_V[i,].reshape(1, *input_shape), batch_size=1)
        pred_log_rho[i,] = tmp.reshape(tmp.shape[0], -1).copy()
        if inverse_trans:
            log_transform(synth_V[i,], inverse=True, inplace=True)

    # if data_generator is None:
    #     synth_V = np.empty(num_samples, num_features)
    #     synth_log_rho = np.empty(num_samples, num_output_elem)
    #     pred_log_rho = np.empty(num_samples, num_output_elem)
    #     for i in range(num_samples):
    #         synth_V[i, ] = np.load(npz_list[i])['Inputs']
    #         synth_log_rho[i, ] = np.load(npz_list[i])['Targets']
    #         tmp = model.predict(synth_V[i, ].reshape(input_shape), batch_size=32)
    #         pred_log_rho[i, ] = tmp.reshape(tmp.shape[0], -1).copy()
    # elif hasattr(data_generator, '__getitem__')\
    #         and hasattr(data_generator, '__len__')\
    #         and hasattr(data_generator, '__iter__'):
    #     pred_log_rho = model.predict_generator(data_generator, workers=4,
    #                               use_multiprocessing=False, verbose=True)
    #     pred_log_rho = pred_log_rho.reshape(pred_log_rho.shape[0], -1)  # new view
    # else:
    #     raise ValueError(f'{data_generator} id not a valid generator')

    if not os.path.isfile(dest_h5):
        shutil.copy(src_h5, dest_h5)
        print(f'copy {src_h5} to {dest_h5}')

    # Because use Fortran index, data matrix should be transposed.
    synth_V = synth_V.T.astype('float64')
    synth_log_rho = synth_log_rho.T.astype('float64')
    pred_log_rho = pred_log_rho.T.astype('float64')

    with h5py.File(name=dest_h5, mode='a') as f:
        # save synth_V
        try:
            dset = f.create_dataset('/synth_data/synth_V', data=synth_V,
                                    chunks=True, maxshape=(num_features, None))
            print(f'save {dest_h5}: {dset.name}')
        except:
            dset = f['/synth_data/synth_V']
            dset.resize((dset.shape[1] + num_samples), axis=1)
            dset[:, -num_samples:] = synth_V
            print(f'*append {dest_h5}: {dset.name}')
        # save synth_log_rho
        try:
            dset = f.create_dataset('/synth_data/synth_log_rho', data=synth_log_rho,
                                    chunks=True, maxshape=(num_output_elem, None))
            print(f'save {dest_h5}: {dset.name}')
        except:
            dset = f['/synth_data/synth_log_rho']
            dset.resize((dset.shape[1] + num_samples), axis=1)
            dset[:, -num_samples:] = synth_log_rho
            print(f'*append {dest_h5}: {dset.name}')
        # save pred_log_rho
        try:
            dset = f.create_dataset('/synth_data/pred_log_rho', data=pred_log_rho,
                                    chunks=True, maxshape=(num_output_elem, None))
            print(f'save {dest_h5}: {dset.name}')
        except:
            dset = f['/synth_data/pred_log_rho']
            dset.resize((dset.shape[1] + num_samples), axis=1)
            dset[:, -num_samples:] = pred_log_rho
            print(f'*append {dest_h5}: {dset.name}')


def save_daily_data(model, src_h5, urf_dir, dest_h5=None, preprocess=False,
                    start=None, end=None, fmt='%Y%m%d'):
    """
    Save daily data.

    The synthetic data to be checked is saved to the hdf5 file, and then the crossplot of
    the synthetic V/I and predictive V/I can be used to check the robustness of the NN model.

    Parameters
    ----------
    model : Instance of `keras Model`
        The neural network model for prediction.
    src_h5 : str
        hdf5 file contain forward modeling parameters in the /glob_para group.
    urf_dir : str
        Directory containing urf files.
    dest_h5 : str, default None
        Copy hdf5 file to new destination. The default is to use src_h5 as dest_h5.
    preprocess : bool, default False
        Do log transform?
    start : str, default None
        time string. e.g. '20180612'.
        If 'start' is None, 'start' will use'00010101'.
    end : str, default None
        time string. e.g. '20180630'.
        If 'end' is None, 'end' will use '99991231'.
    fmt : str, default '%Y%m%d'
        Format that parse 'start' and 'end' string.
        It is strongly recommended not to change this format!

    Returns
    -------
    None

    References
    ----------
    http://download.nexusformat.org/sphinx/examples/h5py/index.html
    """

    dest_h5 = src_h5 if dest_h5 is None else dest_h5
    os.makedirs(os.path.dirname(dest_h5), exist_ok=True)

    if not os.path.isfile(dest_h5):
        shutil.copy(src_h5, dest_h5)
        print(f'copy {src_h5} to {dest_h5}')

    # convert start and end to datetime.datetime instance
    start, end = datetime_range(start, end, fmt=fmt)
    # get receive_date list
    dlist = list(''.join(dlist.split('.')[
                         0:-1]) for dlist in os.listdir(urf_dir) if dlist.endswith(".urf"))
    # filter dates in datetime range and return an iterable filter instance
    it = filter(lambda t: datetime_in_range(t, start, end), dlist)
    input_shape = model.layers[1].input_shape[1:]

    for receive_date in it:

        urf = os.path.join(urf_dir, f'{receive_date}.urf')
        _, _, _, _, data = read_urf(urf)
        obs_V = data[:, 4].reshape(1, -1)  # get resistance and reshape
        V = np.nan_to_num(obs_V)  # replace nan with 0 !!!
        missing_value = u'replace nan in obs_V with 0 for prediction'  # UTF-8 encoding
        if preprocess:
            log_transform(V, inverse=False, inplace=True)
        pred_log_rho = model.predict(V.reshape(1, *input_shape), batch_size=1)
        pred_log_rho = pred_log_rho.reshape(1, -1)

        # Because use Fortran index, data matrix should be transposed.
        obs_V = obs_V.T.astype('float64')
        pred_log_rho = pred_log_rho.T.astype('float64')

        with h5py.File(name=dest_h5, mode='a') as f:
            # save pred_log_rho
            try:
                dset = f.create_dataset(
                    f'/daily_data/{receive_date}/pred_log_rho',
                    data=pred_log_rho, chunks=True)
                dset.attrs['missing_value'] = missing_value
                print(f'save {dest_h5}: {dset.name}')
            except:
                dset = f[f'/daily_data/{receive_date}/pred_log_rho']
                dset[:] = pred_log_rho
                dset.attrs['missing_value'] = missing_value
                print(f'*resave {dest_h5}: {dset.name}')
            # save obs_V
            try:
                dset = f.create_dataset(
                    f'/daily_data/{receive_date}/obs_V',
                    data=obs_V, chunks=True)
                print(f'save {dest_h5}: {dset.name}')
            except:
                dset = f[f'/daily_data/{receive_date}/obs_V']
                dset[:] = obs_V
                print(f'*resave {dest_h5}: {dset.name}')


def save_nn_model(model, output_dir='.', model_name='model.h5'):
    """Save keras model's object, weights, architecture and graph image.

    Parameters
    ----------
    model : Instance of `keras Model`
        The neural network model you want to save.
    output_dir : str
        The directory where the model is saved.
    model_name : str, default 'model.h5'
        Model name with a h5(h5df) extension.

    Returns
    -------
    None
    """

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # split model_name and extension name
    name, extension = os.path.splitext(model_name)
    # check extension is .h5 or .hdf5
    pattern = re.compile('.(h5|hdf5)')
    if not re.match(pattern, extension):
        extension = '.h5'
    # extension = '.h5' if not extension else extension

    # save complete model
    fullname = os.path.join(output_dir, model_name)
    model.save(fullname)
    print(model.name + ' saved as ' + fullname)

    # save model architecture
    fullname = os.path.join(output_dir, name + '_architecture.json')
    model_json = model.to_json()
    with open(fullname, "w") as json_file:
        json_file.write(model_json)
    print(model.name + ' saved as ' + fullname)

    # save model weights
    fullname = os.path.join(output_dir, name + '_weights' + extension)
    model.save_weights(fullname)
    print(model.name + ' saved as ' + fullname)

    # save simple graph image
    plot_model(model, to_file=os.path.join(
        output_dir, 'model_graph.png'), show_shapes=True)


def save_used_data(src_h5, used_data, dest_h5=None):
    """
    Save used data.

    Save used training and testing data to hdf5 file, this function can also save validation data.

    Parameters
    ----------
    model : Instance of `keras Model`
        The neural network model for prediction.
    src_h5 : str
        hdf5 file contain forward modeling parameters in the /glob_para group.
    used_data : tuple
        Tuple contains training and testing data or additional validation data.
        In each matrix, the number of rows must be the number of samples.
        ((x_train, y_train), (x_test, y_test))
        ((x_train, y_train), (x_test, y_test), (x_valid, y_valid))
    dest_h5 : str, default None
        Copy hdf5 file to new destination. The default is to use src_h5 as dest_h5.

    Returns
    -------
    None

    References
    ----------
    https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py
    https://stackoverflow.com/questions/25655588/incremental-writes-to-hdf5-with-h5py?rq=1
    http://download.nexusformat.org/sphinx/examples/h5py/index.html
    """

    try:
        (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = used_data
        with_valid = True
        print('Have validation data')
    except:
        (x_train, y_train), (x_test, y_test) = used_data
        with_valid = False

    dest_h5 = src_h5 if dest_h5 is None else dest_h5

    if not os.path.isfile(dest_h5):
        shutil.copy(src_h5, dest_h5)
        print('copy {} to {}'.format(src_h5, dest_h5))

    num_train_samples = x_train.shape[0]
    num_test_samples = x_test.shape[0]
    num_features = x_train.shape[1]
    num_output_elem = y_train.shape[1]
    # Because use Fortran-order indexing, data matrix should be transposed.
    x_train = x_train.T.astype('float64')
    y_train = y_train.T.astype('float64')
    x_test = x_test.T.astype('float64')
    y_test = y_test.T.astype('float64')
    if with_valid:
        num_valid_samples = x_valid.shape[0]
        x_valid = x_valid.T.astype('float64')
        y_valid = y_valid.T.astype('float64')

    with h5py.File(name=dest_h5, mode='a') as f:
        # save x_train
        try:
            dset = f.create_dataset('/used_data/x_train', data=x_train,
                                    chunks=True, maxshape=(num_features, None))
            print(f'save {dest_h5}: {dset.name}')
        except:
            dset = f['/used_data/x_train']
            dset.resize((dset.shape[1] + num_train_samples), axis=1)
            dset[:, -num_train_samples:] = x_train
            print(f'*append {dest_h5}: {dset.name}')
        # save y_train
        try:
            dset = f.create_dataset('/used_data/y_train', data=y_train,
                                    chunks=True, maxshape=(num_output_elem, None))
            print(f'save {dest_h5}: {dset.name}')
        except:
            dset = f['/used_data/y_train']
            dset.resize((dset.shape[1] + num_train_samples), axis=1)
            dset[:, -num_train_samples:] = y_train
            print(f'*append {dest_h5}: {dset.name}')
        # save x_test
        try:
            dset = f.create_dataset('/used_data/x_test', data=x_test,
                                    chunks=True, maxshape=(num_features, None))
            print(f'save {dest_h5}: {dset.name}')
        except:
            dset = f['/used_data/x_test']
            dset.resize((dset.shape[1] + num_test_samples), axis=1)
            dset[:, -num_test_samples:] = x_test
            print(f'*append {dest_h5}: {dset.name}')
        # save y_test
        try:
            dset = f.create_dataset('/used_data/y_test', data=y_test,
                                    chunks=True, maxshape=(num_output_elem, None))
            print(f'save {dest_h5}: {dset.name}')
        except:
            dset = f['/used_data/y_test']
            dset.resize((dset.shape[1] + num_test_samples), axis=1)
            dset[:, -num_test_samples:] = y_test
            print(f'*append {dest_h5}: {dset.name}')
        if with_valid:
            # save x_valid
            try:
                dset = f.create_dataset('/used_data/x_valid', data=x_valid,
                                        chunks=True, maxshape=(num_features, None))
                print(f'save {dest_h5}: {dset.name}')
            except:
                dset = f['/used_data/x_valid']
                dset.resize((dset.shape[1] + num_valid_samples), axis=1)
                dset[:, -num_valid_samples:] = x_valid
                print(f'*append {dest_h5}: {dset.name}')
            # save y_valid
            try:
                dset = f.create_dataset('/used_data/y_valid', data=y_valid,
                                        chunks=True, maxshape=(num_output_elem, None))
                print(f'save {dest_h5}: {dset.name}')
            except:
                dset = f['/used_data/y_valid']
                dset.resize((dset.shape[1] + num_valid_samples), axis=1)
                dset[:, -num_valid_samples:] = y_valid
                print(f'*append {dest_h5}: {dset.name}')
