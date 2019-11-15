import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pyswarms as ps
from pyswarms.utils.search import RandomSearch

from erinn.python.FW2_5D.fw2_5d_ext import get_forward_para, forward_simulation
from erinn.python.utils.io_utils import read_pkl


def _forward_simulation(pred_log_rho_swarm, obs_V, config):
    l1 = np.array([], dtype=np.float64)
    # l2 = np.array([], dtype=np.float64)
    smooth = np.array([], dtype=np.float64)
    for pred_log_rho in pred_log_rho_swarm:
        tmp = np.power(10, pred_log_rho).reshape(config['nz'], config['nx']).T
        sigma = 1 / tmp
        # l2_tmp = np.array([np.sum(np.power(forward_simulation(sigma, config) - obs_V, 2))])
        l1_tmp = np.array([np.sum(np.abs(forward_simulation(sigma, config) - obs_V))])
        smooth_tmp = np.array([np.sum(np.abs(np.gradient(tmp)))])
        # smooth_tmp = np.array([np.sum(np.power(np.gradient(tmp), 2))])
        # smooth_tmp = np.array([np.sum(np.power(np.diff(F, 0).flatten(), 2))
        #                        + np.sum(np.power(np.diff(tmp, 1).flatten(), 2))])
        if smooth.size:
            l1 = np.concatenate((l1, l1_tmp))
            # l2 = np.concatenate((l2, l2_tmp))
            smooth = np.concatenate((smooth, smooth_tmp))
        else:
            l1 = l1_tmp.copy()
            # l2 = l2_tmp.copy()
            smooth = smooth_tmp.copy()
    return l1 + 0.2 * smooth


if __name__ == '__main__':

    config_file = os.path.join('..', 'config', 'config.yml')
    prediction_dir = os.path.join('..', 'models', 'no_preprocess', 'predictions', 'raw_data')
    pkl_file = os.path.join(prediction_dir, 'result_4750.pkl')
    data = read_pkl(pkl_file)

    # # Random search hyper-parameter
    # options = {'c1': [0.1, 5],
    #            'c2': [0.1, 5],
    #            'w': [0.1, 5],
    #            'k': [4, 32],
    #            'p': 1}
    # num_particle = 64
    # iters = 30
    # n_selection_iters = 10
    # dimensions = data['pred_log_rho'].size
    # max_bound = 5 * np.ones((dimensions, ))
    # min_bound = -2 * np.ones((dimensions, ))
    # bounds = (min_bound, max_bound)
    # velocity_clamp = (-0.3, 0.3)
    #
    # obs_V = data['synth_V'].flatten()
    # config = get_forward_para(config_file)
    # par = partial(_forward_simulation, obs_V=obs_V, config=config)
    # g = RandomSearch(ps.single.LocalBestPSO, n_particles=num_particle,
    #                  dimensions=dimensions, options=options, objective_func=par,
    #                  iters=iters, n_selection_iters=n_selection_iters,
    #                  bounds=bounds, velocity_clamp=velocity_clamp)
    #
    # best_score, best_options = g.search()
    # # >>> best_options = {'c1': 3.5322163151126724,
    # #                     'c2': 2.518708436292546,
    # #                     'w': 1.4265179791119178,
    # #                     'k': 21,
    # #                     'p': 1}

    # Set up hyper-parameters
    options = {'c1': 0.5,
               'c2': 0.4,
               'w': 0.9,
               'k': 16,
               'p': 1}
    num_particle = 64
    iters = 50
    num_processes = os.cpu_count()
    dimensions = data['pred_log_rho'].size
    max_bound = 5 * np.ones((dimensions,))
    min_bound = -2 * np.ones((dimensions,))
    bounds = (min_bound, max_bound)
    bh_strategy = 'random'  # {'nearest', 'random', 'shrink', 'reflective', 'intermediate', 'periodic'}
    velocity_clamp = (-0.2, 0.2)
    vh_strategy = 'unmodified'  # {'unmodified', 'adjust', 'invert', 'zero'}
    ftol = -np.inf
    init_pos = np.tile(data['pred_log_rho'].flatten(), (num_particle, 1))
    init_pos[1:, :] = init_pos[1:, :] + ((1 - (-1)) * np.random.random(init_pos[1:, :].shape) + (-1))

    optimizer = ps.single.LocalBestPSO(n_particles=num_particle,
                                       dimensions=dimensions,
                                       options=options,
                                       bounds=bounds,
                                       bh_strategy=bh_strategy,
                                       velocity_clamp=velocity_clamp,
                                       vh_strategy=vh_strategy,
                                       ftol=ftol,
                                       init_pos=init_pos)

    # Perform optimization
    obs_V = data['synth_V'].flatten()
    config = get_forward_para(config_file)
    # par = partial(_forward_simulation, obs_V=obs_V, config=config)
    # cost, pos = optimizer.optimize(par, iters=iters)
    cost, pos = optimizer.optimize(_forward_simulation, iters=iters,
                                   n_processes=num_processes, obs_V=obs_V, config=config)

    plt.imshow(data['synth_log_rho'])
    plt.colorbar()
    plt.show()
    plt.imshow(data['pred_log_rho'])
    plt.colorbar()
    plt.show()
    plt.imshow(pos.reshape(30, 140))
    plt.colorbar()
    plt.show()
