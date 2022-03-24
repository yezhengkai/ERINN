import os

import numpy as np

from erinn.utils.io_utils import read_config_file, read_urf, read_pkl
from erinn.utils.vis_utils import plot_result_synth

# TODO: Organize reusable code snippets into functions

FILEDIR = os.path.dirname(__file__)

# read config
config_file = os.path.join(FILEDIR, '..', '..', 'config', 'for_plot.yml')
config = read_config_file(config_file)

# parse config and setting
model_dir = os.path.join(FILEDIR, config['model_dir'])
save_figs_dir = os.path.join(config['save_figs_dir'])
predictions_dir = os.path.join(model_dir, 'predictions')
simulator_pkl = os.path.join(model_dir, 'simulator.pkl')
simulator = read_pkl(simulator_pkl)
num_figs = config["num_figs"]
if isinstance(num_figs, str):
    if num_figs == 'all':
        num_figs = np.inf  # use np.inf to save all figures
    else:
        raise(ValueError('String input of "num_figs" only accepts "all"'))
elif not isinstance(num_figs, int):
    raise(TypeError('Input of "num_figs" only accepts "str" and "int" types'))

os.makedirs(save_figs_dir, exist_ok=True)
iterator_pred = os.scandir(predictions_dir)

plot_result_synth(iterator_pred, num_figs, simulator, save_dir=save_figs_dir)
