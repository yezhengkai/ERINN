"""Physical simulation using the resistivity predicted by the neural network (get the resistance, i.e. V/I)."""
import contextlib
import multiprocessing as mp
import os
from functools import partial

import numpy as np
from tqdm import tqdm

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
