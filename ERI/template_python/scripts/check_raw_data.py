import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from erinn.python.utils.io_utils import read_pkl, read_config_file

config_file = os.path.join('..', 'config', 'config.yml')
preprocess_dir = 'noise_10'
train_dir = os.path.join('..', 'data', preprocess_dir, 'train')
valid_dir = os.path.join('..', 'data', preprocess_dir, 'valid')
test_dir = os.path.join('..', 'data', preprocess_dir, 'test')

config = read_config_file(config_file)
iterator_train = os.scandir(train_dir)
iterator_valid = os.scandir(valid_dir)
iterator_test = os.scandir(test_dir)
num = 2

# inspired by https://joseph-long.com/writing/colorbars/
params = {
    'image.origin': 'upper',
    'image.interpolation': 'nearest',
    'image.cmap': 'jet',
    'axes.grid': False,
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    'font.size': 8,  # was 10
    'legend.fontsize': 6,  # was 10
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.figsize': [4, 1],  # [3.39, 2.10],
    'figure.dpi': 150,
    'font.family': 'serif',
}
mpl.rcParams.update(params)


def plot_raw_data(iterator, config_dict, num_figs):

    num_figs = 1 if num_figs < 1 else num_figs
    i = 1
    for file in iterator:
        data = read_pkl(file.path)
        print(data['inputs'].shape, data['targets'].shape)
        delta_v = data['inputs'].reshape((210, 780))
        rho = data['targets'].reshape((config_dict['nz'], config_dict['nx']))
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(delta_v, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        ax.set_xlabel('Rx_pair')
        ax.set_ylabel('Tx_pair')
        cbar.set_label('$\Delta V/I$')

        fig, ax = plt.subplots()
        im = ax.imshow(rho)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        ax.set_xlabel('m')
        ax.set_ylabel('m')
        cbar.set_label('$log_{10}\/\Omega-m$')

        plt.show()
        if i == num_figs:
            break
        else:
            i += 1


plot_raw_data(iterator_train, config, num)
plot_raw_data(iterator_valid, config, num)
plot_raw_data(iterator_test, config, num)
