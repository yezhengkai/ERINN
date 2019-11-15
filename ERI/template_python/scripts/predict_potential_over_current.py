import multiprocessing as mp
import os
from functools import partial

import numpy as np
from tqdm import tqdm

from erinn.python.FW2_5D.fw2_5d_ext import get_forward_para, forward_simulation
from erinn.python.utils.io_utils import get_pkl_list, read_pkl, write_pkl


def _forward_simulation(pkl_name, config):
    data = read_pkl(pkl_name)
    shape_V = data['synth_V'].shape
    sigma = 1 / np.power(10, data['pred_log_rho']).T
    data['pred_V'] = forward_simulation(sigma, config).reshape(shape_V)
    write_pkl(data, pkl_name)


if __name__ == '__main__':
    # Setting
    config_file = os.path.join('..', 'config', 'config.yml')
    model_dir = os.path.join('..', 'models', 'add_noise_log_transform')
    predictions_dir = os.path.join(model_dir, 'predictions', 'raw_data')

    pkl_list_result = get_pkl_list(predictions_dir)

    os.makedirs(predictions_dir, exist_ok=True)
    config = get_forward_para(config_file)

    par = partial(_forward_simulation, config=config)
    pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
    for _ in tqdm(pool.imap_unordered(par, pkl_list_result),
                  total=len(pkl_list_result), desc='predict V/I'):
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
