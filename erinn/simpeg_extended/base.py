from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import contextlib
import multiprocessing as mp
import os
import warnings
from functools import partial
from itertools import combinations

import numpy as np
from SimPEG import Maps
from SimPEG.EM import Static
from tqdm import tqdm

from erinn.simpeg_extended.random_model import get_random_model
from erinn.utils.io_utils import URF
from erinn.utils.io_utils import read_config_file
from erinn.utils.io_utils import write_pkl
from erinn.utils.os_utils import next_path

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


class Simulator(object):

    def __init__(self, config_file):

        self._LINE_LENGTH_RATIO = 0.2  # for determining core_z_length
        self._DELTA_TRN_RATIO = 1.75  # for determining core_z_length
        self._survey = None
        self._topo = None
        self._mesh = None
        self._active_idx = None
        self._problem = None
        self.random_resistivity_generator = None
        self.config = read_config_file(config_file)

        # suitable for 2D surface survey now...
        # read urf file
        self.urf = URF(self.config['geometry_urf'],
                       survey_type='dipole-dipole',
                       dimension=2,
                       space_type='half-space')

        # generate an instance of IO
        self._IO = Static.DC.IO()

        self._prepare()
        self._get_unpaired_survey()
        self._get_mapping()
        self._get_problem()

    @property
    def mesh(self):
        return self._mesh

    @property
    def survey(self):
        return self._survey

    @property
    def problem(self):
        return self._problem

    @property
    def IO(self):
        return self._IO

    def _prepare(self):

        # get the ids and resistances
        if self.urf.data is None:
            c_pair = [set(i)
                      for i in combinations(self.urf.Tx_id.flatten().tolist(), 2)]  # list of c_pair set
            p_pair = [set(i)
                      for i in combinations(self.urf.Rx_id.flatten().tolist(), 2)]  # list of p_pair set

            # TODO: list comprehension is better?
            abmn_id = []
            for i in range(len(c_pair)):
                for j in range(len(p_pair)):
                    if c_pair[i].isdisjoint(p_pair[j]):  # Return True if two sets have a null intersection.
                        abmn_id.append(sorted(c_pair[i]) + sorted(p_pair[j]))  # use sorted to convert set to list

            # construct essential quantity for urf.data
            abmn_id = np.array(abmn_id)
            num_data = abmn_id.shape[0]
            resistance = np.ones((num_data, 1))
            i = 1000 * np.ones((num_data, 1))
            error = np.zeros((num_data, 1))
            self.urf.data = np.hstack((abmn_id, resistance, i, error))
            self.urf.get_abmn_locations()  # In urf, z is positive up. In SimPEG, z is positive up.
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.urf.resistance is None:
                    raise ValueError("URF.data has a ndim that is not equal to 2, or does not contain V/I column.\n"
                                     "Check your urf file.")
                self.urf.get_abmn_locations()  # In urf, z is positive up. In SimPEG, z is positive up.

        # Filter out electrode pairs that do not adhere to a specific array type
        if self.config['array_type'] != 'all_combination':

            # Calculate the separation distance between the electrodes in the x direction
            am = self.urf.m_locations[:, 0] - self.urf.a_locations[:, 0]
            mn = self.urf.m_locations[:, 0] - self.urf.n_locations[:, 0]
            nb = self.urf.b_locations[:, 0] - self.urf.n_locations[:, 0]

            # Check if the electrode is on the ground
            at_ground = np.logical_and(np.logical_and(self.urf.a_locations[:, 1] == 0,
                                                      self.urf.b_locations[:, 1] == 0),
                                       np.logical_and(self.urf.m_locations[:, 1] == 0,
                                                      self.urf.n_locations[:, 1] == 0))
            # TODO: the arrangement of AMNB is not important?
            # Check that the electrode arrangement is correct
            positive_idx = np.logical_and(np.logical_and(am > 0, mn > 0), nb > 0)

            # Check specific array arrangement
            if self.config['array_type'] == 'Wenner_Schlumberger':
                # Must be an integer multiple?
                row_idx = np.logical_and(am == nb, am % mn == 0)
                final_idx = np.logical_and(np.logical_and(at_ground, positive_idx), row_idx)
            elif self.config['array_type'] == 'Wenner':
                row_idx = np.logical_and(am == mn, mn == nb)
                final_idx = np.logical_and(np.logical_and(at_ground, positive_idx), row_idx)
            elif self.config['array_type'] == 'Wenner_Schlumberger_NonInt':
                row_idx = np.logical_and(am == nb, am >= mn)
                final_idx = np.logical_and(np.logical_and(at_ground, positive_idx), row_idx)
            else:
                raise NotImplementedError()
            self.urf.data = self.urf.data[final_idx, :]  # update data

            # read 2D terrain file and update abmn locations
            try:
                self._topo = np.loadtxt(self.config['terrain_file'], delimiter=",", skiprows=3)
                _, x_idx, y_idx = np.intersect1d(self.urf.coord[:, 1], self._topo[:, 0],
                                                 assume_unique=True, return_indices=True)
                self.urf.coord[x_idx, 3] = self._topo[y_idx, 1]  # update z coordinate of electrodes
            finally:
                self.urf.get_abmn_locations()
            # except OSError:
            #     self._topo = self._IO.electrode_locations

    def _get_unpaired_survey(self):

        # Generate DC survey using IO object
        # suitable for 2D surface survey now...
        self._survey = self._IO.from_ambn_locations_to_survey(
            self.urf.a_locations, self.urf.b_locations,
            self.urf.m_locations, self.urf.n_locations,
            survey_type='dipole-dipole',
            data_dc=self.urf.resistance,
            data_dc_type='volt'
        )

        # generate mesh and applied topography(terrain) information
        # In addition to 12 padding cells, IO.set_mesh will automatically add 3 cells on each side of the x direction
        !!!!!!
        delta_terrain = np.max(self._topo[:, 1]) - np.min(self._topo[:, 1])
        line_length = abs(self._IO.electrode_locations[:, 0].max()
                          - self._IO.electrode_locations[:, 0].min())

        if line_length * self._LINE_LENGTH_RATIO > delta_terrain:
            core_z_length = line_length * self._LINE_LENGTH_RATIO
        else:
            core_z_length = delta_terrain * self._DELTA_TRN_RATIO

        self._mesh, self._active_idx = self._IO.set_mesh(
            topo=self._topo,
            dx=1, dy=None, dz=1,
            n_spacing=None, corezlength=core_z_length,
            npad_x=12, npad_y=12, npad_z=10,
            pad_rate_x=1, pad_rate_y=1, pad_rate_z=1,
            ncell_per_dipole=2, mesh_type='TensorMesh',
            dimension=2, method='nearest'
        )
        # Manipulating electrode location (Drape location right below [cell center] the topography)
        self._survey.drapeTopo(self._mesh, self._active_idx, option="top")

    def _get_mapping(self):
        # Use Exponential Map: m = log(rho)
        active_map = Maps.InjectActiveCells(self._mesh, indActive=self._active_idx, valInactive=np.log(1e8))
        self._mapping = Maps.ExpMap(self._mesh) * active_map

    def get_random_resistivity_generator(self, num_samples):
        return get_random_model(self.config, self._mesh, num_samples=num_samples)

    def _get_problem(self, simulation_type='N'):
        # "N" means potential is defined at nodes
        # "CC" means potential is defined at cell center
        if simulation_type == 'N':
            self._problem = Static.DC.Problem2D_N(self._mesh,
                                                  rhoMap=self._mapping,
                                                  storeJ=True,
                                                  Solver=Solver)
        elif simulation_type == 'CC':
            self._problem = Static.DC.Problem2D_CC(self._mesh,
                                                   rhoMap=self._mapping,
                                                   storeJ=True,
                                                   Solver=Solver)
        else:
            raise NotImplementedError()

        # Pair problem with survey
        if not self._problem.ispaired:
            self._problem.pair(self._survey)
        else:
            self._problem.unpair()
            self._problem.pair(self._survey)

    def make_synthetic_data(self, resistivity, std=None, f=None, force=False):
        resistivity = np.log10(resistivity[self._active_idx])
        return self._survey.makeSyntheticData(resistivity, std=std, f=f, force=force)


def make_dataset(config_file):
    """Generate raw dataset and save it as pickle.

    Parameters
    ----------
    config_file : str, pathlib.Path or dict
        Path to a yaml file for configuration or a dictionary for configuration.

    Returns
    -------
    None
    """
    # parse config
    config = read_config_file(config_file)
    train_dir = os.path.join(config['dataset_dir'], 'train')
    valid_dir = os.path.join(config['dataset_dir'], 'validation')
    test_dir = os.path.join(config['dataset_dir'], 'test')
    num_samples_train = int(config['num_samples'] * config['train_ratio'])
    num_samples_valid = int(config['num_samples']
                            * (config['train_ratio'] + config['valid_ratio'])
                            - num_samples_train)
    num_samples_test = config['num_samples'] - num_samples_train - num_samples_valid

    simulator = Simulator(config)
    for dir_name, num_samples in ((train_dir, num_samples_train),
                                  (valid_dir, num_samples_valid),
                                  (test_dir, num_samples_test)):
        if num_samples == 0:
            pass
        else:
            os.makedirs(dir_name, exist_ok=True)
            suffix_num = next_path(os.path.join(dir_name, 'raw_data_%s.pkl'), only_num=True)

            par = partial(_make_dataset, simulator=simulator, dir_name=dir_name)
            resistivity_generator = simulator.get_random_resistivity_generator(num_samples=num_samples)
            suffix_generator = iter(range(suffix_num, suffix_num + num_samples))
            pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
            for _ in tqdm(pool.imap_unordered(par, zip(resistivity_generator, suffix_generator)),
                          desc=f'Generate {os.path.basename(dir_name)} data',
                          total=num_samples):
                pass
            pool.close()
            pool.join()


def _make_dataset(zip_item, simulator, dir_name):
    """Protected function for parallel generate dataset.

    Generate noise-free synthetic data and save it as pickle.

    Parameters
    ----------
    zip_item : zip object
    simulator : Simulator
    dir_name : str or pathlib.Path

    Returns
    -------
    None
    """
    resistivity, suffix_num = zip_item
    # stop printing messages
    with contextlib.redirect_stdout(None):
        data_synthetic = simulator.make_synthetic_data(resistivity, std=0)
    # pickle dump/load is faster than numpy savez_compressed(or save)/load
    pkl_name = os.path.join(dir_name, f'raw_data_{suffix_num:0>6}.pkl')
    write_pkl({'inputs': data_synthetic, 'targets': np.log10(resistivity)}, pkl_name)
