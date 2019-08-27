from __future__ import division, absolute_import, print_function

import os
import pickle
from itertools import combinations

import numpy as np

from .fw2_5d import dcfw2_5D
from .fw2_5d import get_2_5Dpara
from .rand_synth_model import get_rand_model
from ..utils.io_utils import read_config_file
from ..utils.io_utils import read_urf


def prepare_for_get_2_5d_para(config_file, return_urf=False):
    config = read_config_file(config_file)

    urf = config['geometry_urf']
    Tx_id, Rx_id, RxP2_id, coord, data = read_urf(urf)
    # Collect pairs id
    if np.all(np.isnan(data)):
        C_pair = [set(i) for i in combinations(Tx_id.flatten().tolist(), 2)]
        P_pair = [set(i) for i in combinations(Rx_id.flatten().tolist(), 2)]
        CP_pair = []
        for i in range(len(C_pair)):
            for j in range(len(P_pair)):
                if C_pair[i].isdisjoint(P_pair[j]):
                    CP_pair.append(sorted(C_pair[i]) + sorted(P_pair[j]))  # use sorted to convert set to list
        CP_pair = np.array(CP_pair, dtype=np.int64)
    else:
        CP_pair = data[:, :4].astype(np.int64)

    # Convert id to coordinate
    recloc = np.hstack((coord[CP_pair[:, 2] - 1, 1:4:2],
                        coord[CP_pair[:, 3] - 1, 1:4:2]))
    recloc[:, 1:4:2] = np.abs(recloc[:, 1:4:2])  # In urf, z is positive up. In fw25d, z is positive down.
    SRCLOC = np.hstack((coord[CP_pair[:, 0] - 1, 1:4:2],
                        coord[CP_pair[:, 1] - 1, 1:4:2]))
    SRCLOC[:, 1:4:2] = np.abs(SRCLOC[:, 1:4:2])  # In urf, z is positive up. In fw25d, z is positive down.

    # Collect pairs that fit the array configuration
    if config['array_type'] != 'all_combination':
        # Check if the electrode is on the ground
        at_ground = np.logical_and(np.logical_and(SRCLOC[:, 1] == 0, SRCLOC[:, 3] == 0),
                                   np.logical_and(recloc[:, 1] == 0, recloc[:, 3] == 0))
        SRCLOC = SRCLOC[at_ground, :]
        recloc = recloc[at_ground, :]
        AM = recloc[:, 0] - SRCLOC[:, 0]
        MN = recloc[:, 2] - recloc[:, 0]
        NB = SRCLOC[:, 2] - recloc[:, 2]
        # Check that the electrode arrangement is correct
        positive_idx = np.logical_and(np.logical_and(AM > 0, MN > 0), NB > 0)
        SRCLOC = SRCLOC[positive_idx, :]
        recloc = recloc[positive_idx, :]
        AM = AM[positive_idx]
        MN = MN[positive_idx]
        NB = NB[positive_idx]
        if config['array_type'] == 'Wenner_Schlumberger':
            # Must be an integer multiple?
            row_idx = np.logical_and(AM == NB, AM % MN == 0)
            SRCLOC = SRCLOC[row_idx, :]
            recloc = recloc[row_idx, :]
        elif config['array_type'] == 'Wenner':
            row_idx = np.logical_and(AM == MN, MN == NB)
            SRCLOC = SRCLOC[row_idx, :]
            recloc = recloc[row_idx, :]
        elif config['array_type'] == 'Wenner_Schlumberger_NonInt':
            row_idx = np.logical_and(AM == NB, AM >= MN)
            SRCLOC = SRCLOC[row_idx, :]
            recloc = recloc[row_idx, :]

    srcloc, srcnum = np.unique(SRCLOC, return_inverse=True, axis=0)
    srcnum = np.reshape(srcnum, (-1, 1))  # matlab index starts from 1, python index starts from 0

    array_len = max(coord[:, 1]) - min(coord[:, 1])
    srcloc[:, [0, 2]] = srcloc[:, [0, 2]] - array_len / 2
    recloc[:, [0, 2]] = recloc[:, [0, 2]] - array_len / 2
    dx = np.ones((config['nx'], 1))
    dz = np.ones((config['nz'], 1))

    if return_urf:
        return [[srcloc, dx, dz, recloc, srcnum],
                [Tx_id, Rx_id, RxP2_id, coord, data]]
    else:
        return srcloc, dx, dz, recloc, srcnum


def get_forward_para(config_file):

    config = read_config_file(config_file)
    srcloc, dx, dz, recloc, srcnum = prepare_for_get_2_5d_para(config)
    para_pkl = config['Para_pkl']
    num_k_g = config['num_k_g']

    if not os.path.isfile(para_pkl):
        print('Create Para for FW2_5D.')
        s = np.ones((config['nx'], config['nz']))
        Para = get_2_5Dpara(srcloc, dx, dz, s, num_k_g, recloc, srcnum)
        with open(para_pkl, 'wb') as f:
            pickle.dump(Para, f)
        config['Para'] = Para
    else:
        print('Load Para pickle file')
        with open(para_pkl, 'rb') as f:
            Para = pickle.load(f)
        # Check if Para is in accordance with current configuration
        if 'Q' not in Para:
            print('No Q matrix in `Para` dictionary, creating a new one.')
            s = np.ones((config['nx'], config['nz']))
            Para = get_2_5Dpara(srcloc, dx, dz, s, num_k_g, recloc, srcnum)
            with open(para_pkl, 'wb') as f:
                pickle.dump(Para, f)
            config['Para'] = Para
        elif Para['Q'].shape[0] != srcnum.shape[0] \
                or Para['Q'].shape[1] != dx.size * dz.size \
                or Para['b'].shape[1] != srcloc.shape[0]:
            print('Size of Q matrix is wrong, creating a new one.')
            s = np.ones((config['nx'], config['nz']))
            Para = get_2_5Dpara(srcloc, dx, dz, s, num_k_g, recloc, srcnum)
            with open(para_pkl, 'wb') as f:
                pickle.dump(Para, f)
            config['Para'] = Para
        else:
            config['Para'] = Para

    config['srcloc'] = srcloc
    config['dx'] = dx
    config['dz'] = dz
    config['recloc'] = recloc
    config['srcnum'] = srcnum

    return config


def forward_simulation(outputs, config):

    Para = config['Para']
    dx = config['dx']
    dz = config['dz']
    recloc = config['recloc']

    # Inputs: delta V
    # num_samples = outputs.shape[1]
    # inputs = np.zeros((num_samples, recloc.shape[0]))
    sigma_size = (dx.size, dz.size)

    # TODO: add multiprocessing
    # for i in range(num_samples):
    s = np.reshape(outputs, sigma_size)
    dobs, _ = dcfw2_5D(s, Para)
    inputs = dobs.flatten()

    yield inputs


def make_dataset(config_file):

    config = read_config_file(config_file)
    os.makedirs(config['data_dir'], exist_ok=True)
    config = get_forward_para(config)

    i = 1
    for sigma in get_rand_model(config):
        for potential in forward_simulation(sigma, config):
            npz_name = os.path.join(config['data_dir'], 'raw_data_' + str(i))
            np.savez_compressed(npz_name, inputs=potential, outputs=1/sigma)
            i += 1
