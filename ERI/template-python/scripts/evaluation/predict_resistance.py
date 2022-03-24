"""Physical simulation using the resistivity predicted by the neural network (get the resistance, i.e. V/I)."""
import contextlib
import multiprocessing as mp
import os
from functools import partial

import numpy as np
from tqdm import tqdm

from erinn.preprocessing import to_midpoint
from erinn.preprocessing import to_txrx
from erinn.utils.io_utils import read_config_file
from erinn.utils.io_utils import get_pkl_list, read_pkl, write_pkl

# TODO: Organize reusable code snippets into functions

FILEDIR = os.path.dirname(__file__)

def _forward_simulation(pkl_name, simulator):
    data = read_pkl(pkl_name)
    # shape_V = data['synthetic_resistance'].shape
    resistivity = np.flipud(np.power(10, data['predicted_resistivity_log10'])).flatten()
    # stop printing messages
    with contextlib.redirect_stdout(None):
        data['predicted_resistance'] = simulator.make_synthetic_data(resistivity, std=0, force=True)

    # TODO: REMOVE! Because We use raw synthetic_resistance, we don't reshape predicted_resistance
    # Make the predicted_resistance the same shape as the synthetic_resistance
    # preprocess = simulator.config['training']['preprocess']
    # for k, v in preprocess.items():
    #     if k == 'to_midpoint' and v.get('perform'):
    #         data['predicted_resistance'] = to_midpoint(
    #             data['predicted_resistance'],
    #             simulator.urf.abmn_locations[:, :4],
    #             simulator.urf.abmn_locations[:, 4:],
    #             value=0, dim=2
    #         )
    #     elif k == 'to_txrx' and v.get('perform'):
    #         data['predicted_resistance'] = to_txrx(
    #             data['predicted_resistance'],
    #             simulator.urf.abmn_locations[:, :4],
    #             simulator.urf.abmn_locations[:, 4:],
    #             value=0, dim=2
    #         )
    write_pkl(data, pkl_name)


if __name__ == '__main__':
    # read config file
    config_file = os.path.join(FILEDIR, '..', '..', 'config', 'for_predict_resistance.yml')
    config = read_config_file(config_file)

    # parse config and setting
    model_dir = os.path.join(FILEDIR, config['model_dir'])
    predictions_dir = os.path.join(model_dir, 'predictions')
    simulator_pkl = os.path.join(model_dir, 'simulator.pkl')
    simulator = read_pkl(simulator_pkl)
    pkl_list_result = get_pkl_list(predictions_dir)


    par = partial(_forward_simulation, simulator=simulator)
    pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
    for _ in tqdm(pool.imap_unordered(par, pkl_list_result),
                  total=len(pkl_list_result), desc='predict resistance (V/I)'):
        pass
    pool.close()
    pool.join()

    # Serial version
    # with tqdm(total=len(pkl_list_result), desc='write pkl') as pbar:
    #     for i, pred in enumerate(pkl_list_result):
    #         data = read_pkl(pkl_list_result[i])
    #         shape_V = data['synth_V'].shape
    #         sigma = 1 / np.power(10, data['pred_log_rho']).T
    #         data['pred_V'] = forward_simulation(sigma, config).reshape(shape_V)
    #
    #         write_pkl(data, pkl_list_result[i])
    #         pbar.update()
