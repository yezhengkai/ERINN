import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from erinn.utils.io_utils import read_pkl
from erinn.preprocessing import to_txrx
from erinn.preprocessing import to_midpoint
from erinn.utils.vis_utils import get_rcParams

FILEDIR = os.path.dirname(__file__)

# config_file = os.path.join('..', '..', 'config', 'config.yml')
workdir = os.path.join(
    FILEDIR, '..', 'ERI', 'template-python', 'scripts', 'visualization'
)
os.chdir(workdir)
raw_data = 'trial1'
train_dir = os.path.join('..', '..', 'data', raw_data, 'training')
valid_dir = os.path.join('..', '..', 'data', raw_data, 'validation')
test_dir = os.path.join('..', '..', 'data', raw_data, 'testing')
simulator_pkl = os.path.join('..', '..', 'data', raw_data, 'simulator.pkl')
print(os.getcwd())

# config = read_config_file(config_file)
iterator_train = os.scandir(train_dir)
iterator_valid = os.scandir(valid_dir)
iterator_test = os.scandir(test_dir)
simulator = read_pkl(simulator_pkl)
num = 1

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
# mpl.rcParams.update(params)
get_rcParams(params)


def plot_data(iterator, simulator, num_figs):

    SRCLOC = simulator.urf.abmn_locations[:, :4]
    RECLOC = simulator.urf.abmn_locations[:, 4:]
    active_idx = simulator.active_idx
    nCx = simulator.mesh.nCx
    nCy = simulator.mesh.nCy
    vectorCCx = simulator.mesh.vectorCCx
    vectorCCy = simulator.mesh.vectorCCy

    num_figs = 1 if num_figs < 1 else num_figs
    i = 1
    for file in iterator:
        data = read_pkl(file.path)
        print(data['resistance'].shape,
              data['resistivity_log10'].shape)
        resistance = data['resistance']
        resistivity = data['resistivity_log10']

        # plot resistance
        # txrx version
        fig, ax = plt.subplots(figsize=(16, 9))
        im = ax.imshow(
            to_txrx(
                resistance,
                SRCLOC,
                RECLOC,
                value=np.nan
            )[:, :, 0],
            origin='lower'
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        ax.set_xlabel('Rx_pair')
        ax.set_ylabel('Tx_pair')
        cbar.set_label(r'$\Delta V/I$')

        # midpoint version
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(
            to_midpoint(
                resistance,
                SRCLOC,
                RECLOC,
                value=np.nan
            )[:, :, 0]
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        ax.set_xlabel('common midpoint')
        ax.set_ylabel('count')
        cbar.set_label(r'$\Delta V/I$')
        ax.set_aspect('auto', adjustable='box')

        # plot resistivity
        # imshow version
        fig, ax = plt.subplots()
        im = simulator.mesh.plotImage(resistivity[active_idx], ax=ax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im[0], cax=cax)
        ax.set_xlabel('m')
        ax.set_ylabel('m')
        cbar.set_label(r'$\Omega \bullet m (log_{10})$')

        # contourf version
        fig, ax = plt.subplots()
        simulator.mesh.plotImage(resistivity[active_idx], ax=ax)
        im = ax.contourf(vectorCCx, vectorCCy,
                         resistivity[active_idx].reshape((nCy, nCx)))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        ax.set_xlabel('m')
        ax.set_ylabel('m')
        cbar.set_label(r'$\Omega \bullet m (log_{10})$')

        plt.show()
        if i == num_figs:
            break
        else:
            i += 1


plot_data(iterator_train, simulator, num)
# plot_data(iterator_valid, simulator, num)
# plot_data(iterator_test, simulator, num)
