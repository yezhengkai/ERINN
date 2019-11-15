import os

import numpy as np

from erinn.python.utils.io_utils import read_config_file, read_urf
from erinn.python.utils.vis_utils import plot_result_synth

# setting
config_file = os.path.join('..', 'config', 'config.yml')
predictions_dir = os.path.join('..', 'models', 'log_transform', 'predictions', 'raw_data')
figs_dir = os.path.join('..', 'reports', 'log_transform', 'testing_figs_raw')
num_figs = np.inf  # np.inf  # use np.inf to save all figures
config = read_config_file(config_file)

os.makedirs(figs_dir, exist_ok=True)
iterator_pred = os.scandir(predictions_dir)
geo_urf = config['geometry_urf']

# electrode coordinates in the forward model
_, _, _, coord, _ = read_urf(geo_urf)
xz = coord[:, 1:3]
xz[:, 0] += (config['nx'] - coord[:, 1].max()) / 2

plot_result_synth(iterator_pred, num_figs, xz, save_dir=figs_dir)
