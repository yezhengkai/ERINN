from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import re
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .io_utils import read_pkl
from .np_utils import crop_zeros


def get_rcParams(new_params=None, update=True, figsize='l'):
    """Get the specific rcParams used by this package.

    Parameters
    ----------
    new_params : dict, default None
        Dictionary of matplotlib's rcParams to manipulate plot setting.
    update : bool, dafault True
        The current rcPamams is updated when True,
        and the rcParams dictionary is returned when False.
    figsize : str, tuple or list, default 'l'
        's': squared paper size
            (4.8*4.8) inch * 300 dpi => (1440*1440) pixel
        'p': portrait paper size
            (4.8*6.4) inch * 300 dpi => (1440*1920) pixel
        'l': landscape paper size
            (6.4*4.8) inch * 300 dpi => (1920*1440) pixel
        tuple or list with 2 element:
            (width, height)

    Returns
    -------
    params : dict
        Dictionary of rcParams.
    """

    # 's': square
    # 'p': portrait
    # 'l': landscape
    if figsize == 's':
        w, h = 4.8, 4.8
    elif figsize == 'p':
        w, h = 4.8, 6.4
    elif figsize == 'l':
        w, h = 6.4, 4.8
    elif isinstance(figsize, (list, tuple)):
        if len(figsize) != 2:
            warnings.warn(
                'The length of figsize is wrong. Use default figsize', Warning)
            w, h = 6.4, 4.8
        else:
            w, h = figsize
    else:
        warnings.warn('figsize is wrong. Use default figsize', Warning)
        w, h = 6.4, 4.8

    # https://stackoverflow.com/questions/47782185/increment-matplotlib-string-font-size
    # sizes = ['xx-small','x-small','small','medium',
    #           'large','x-large','xx-large']
    params = {
        'axes.titlesize': 'x-large',
        'axes.labelsize': 'large',
        'figure.dpi': 100.0,
        'font.family': ['sans-serif'],
        'font.size': 12,
        'figure.figsize': [w, h],
        'figure.titlesize': 'x-large',
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'legend.fontsize': 'medium',
        'mathtext.fontset': 'stix',
        'savefig.dpi': 300.0,
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
    }

    if isinstance(new_params, dict):
        # https://stackoverflow.com/questions/8930915/append-dictionary-to-a-dictionary
        params.update(new_params)

    if update is True:
        mpl.rcParams.update(params)
        return
    else:
        return params


# TODO: The following function need to rewrite and improve
def structureplot_obs(rho, xz, nx, nz, receive_date,
                      filepath='.', mode=None, params=None):
    """Polt resistivity strcture of real field data.

    Parameters
    ----------
    rho : np.ndarray
        Predictive resistivity by NN.
    xz : np.ndarray, 2d array with shape (num_electrode, 2)
        1st column represents x direction and
        2nd column represents z direction.
    nz : int
        Depth of forward model in meters.
        Each mesh is 1m * 1m.
    nx : int
        Width of forward model in meters.
        Each mesh is 1m * 1m.
    receive_date : str or int
        Date the data was received from electrode array.
    filepath : str, default is current directory
        The directory where figures are saved.
    mode : str
        Select the mode to manipulate crossplot.
        'save': save crossplot image.
        'show': show crossplot on screen.
        Others: return figure object.
    params : dict
        Dictionary of matplotlib's rcParams to manipulate plot setting.

    Returns
    -------
    fig : matplotlib's figure object
        The fig is returned when mode is neither 'save' nor 'show'.
    """

    Params = {'image.cmap': 'jet'}
    if isinstance(params, dict):
        Params.update(params)
    get_rcParams(Params, figsize='l')
    fig, ax0 = plt.subplots(dpi=300)
    ax0.plot(xz[:, 0], -xz[:, 1], 'k.')

    v = np.linspace(1, 3, 17, endpoint=True)
    im0 = ax0.contourf(np.flipud(rho.reshape(nz, nx)), v,
                       extent=[0, nx, nz, 0], extend='both')
    ax0.invert_yaxis()
    cbar = plt.colorbar(im0, ticks=v)

    cbar.set_label(r'Resistivity $log_{10}(\Omega m)$')
    # cbar.ax.tick_params(labelsize=12)
    ax0.set_title('{}'.format(receive_date))
    ax0.set_xlabel('Width(m)')
    ax0.set_ylabel('Depth(m)')
    ax0.tick_params(axis='both')
    fig.tight_layout()

    if mode == 'save':
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
        filename = 'pred_by_NN_{}.png'.format(receive_date)
        fulname = os.path.join(filepath, filename)
        fig.savefig(fulname)
        plt.close(fig)
        mpl.rcdefaults()
    elif mode == 'show':
        plt.draw()
        plt.show()
        mpl.rcdefaults()
    else:
        plt.draw()
        mpl.rcdefaults()
        return fig


def crosspolt_obs_V(obs_V, pred_V, receive_date,
                    filepath='.', mode=None, params=None):
    """Crossplot of observed V/I versus predictive V/I.

    Parameters
    ----------
    obs_V : np.ndarray
        Measured resistivity.
    pred_V : np.ndarray, 2d array with shape (num_electrode, 2)
        Predictive resistivity by NN.
    receive_date : str or int
        Date the data was received from electrode array.
    filepath : str, default is current directory
        The directory where figures are saved.
    mode : str
        Select the mode to manipulate crossplot.
        'save': save crossplot image.
        'show': show crossplot on screen.
        Others: return figure object.
    params : dict
        Dictionary of matplotlib's rcParams to manipulate plot setting.

    Returns
    -------
    fig : matplotlib's figure object
        The fig is returned when mode is neither 'save' nor 'show'.
    """

    # check nan ratio
    nb_nan = np.sum(np.isnan(obs_V))
    nb_tot = obs_V.size
    nan_ratio = nb_nan/nb_tot
    if nan_ratio == 1:
        warnings.warn(f'nan ratio is 1 at {receive_date}. '
                      f'The crossplot is not drawn.', Warning)
        return
    # residual sum of squares
    SS_res = np.sum(
        np.square(obs_V[~np.isnan(obs_V)] - pred_V[~np.isnan(obs_V)]))
    # total sum of squares
    SS_tot = np.sum(
        np.square(obs_V[~np.isnan(obs_V)] - obs_V[~np.isnan(obs_V)].mean()))
    # R squared
    R2 = (1 - SS_res/(SS_tot + 1e-8))
    # Root Mean Square Error
    RMS = np.sqrt(np.mean(np.power(
        obs_V[~np.isnan(obs_V)] - pred_V[~np.isnan(obs_V)], 2)))
    # RMS = np.sqrt(np.mean(np.power(
    #     (pred_V[~np.isnan(obs_V)] - obs_V[~np.isnan(obs_V)]) / obs_V[~np.isnan(obs_V)], 2)))
    # correlation coefficient
    corrcoef = np.corrcoef(
        obs_V[~np.isnan(obs_V)], pred_V[~np.isnan(obs_V)])[0, 1]

    # get obs_V maximum and minimum
    x_min, x_max = obs_V[~np.isnan(obs_V)].min(), obs_V[~np.isnan(obs_V)].max()

    # plotting
    get_rcParams(params, update=True, figsize='s')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot([x_min, x_max], [x_min, x_max], 'r')
    ax.scatter(obs_V, pred_V, s=2, c='blue', alpha=.5)
    ax.set_title(receive_date)
    ax.set_xlabel(r'Observed $\Delta V/I$')
    ax.set_ylabel(r'Predictive $\Delta V/I$')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    lim = abs(x_max) if abs(x_max) > abs(x_min) else abs(x_min)
    r = 2 * lim
    # use escape sequence
    # \N{name}: Prints a character from the Unicode database
    textstr = 'R\N{SUPERSCRIPT TWO}: {:{width}.{prec}f}\n'\
              'RMS: {:{width}.{prec}f}\n'\
              'corrcoef: {:{width}.{prec}f}\n'\
              'nan ratio: {:.2%}'.format(
                  R2, RMS, corrcoef, nan_ratio, width=6, prec=4)
    props = dict(boxstyle='round', facecolor=(1, 0.5, 0.5), alpha=0.5)
    ax.text(-lim + 0.01 * r, lim - 0.01 * r, textstr, bbox=props,
            fontsize=10, va='top', ha='left')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-lim * 1.1, lim * 1.1)
    ax.set_ylim(-lim * 1.1, lim * 1.1)
    fig.tight_layout()

    if mode == 'save':
        filename = 'crossplot_{}'.format(receive_date)
        fulname = os.path.join(filepath, filename)
        fig.savefig(fulname)
        plt.close(fig)
        mpl.rcdefaults()
    elif mode == 'show':
        plt.draw()
        plt.show()
        mpl.rcdefaults()
    else:
        plt.draw()
        mpl.rcdefaults()
        return fig


def scatter_obs_V(obs_V, pred_V, receive_date,
                  filepath='.', mode=None, params=None):
    """Scatter plot of observed V/I versus predictive V/I.

    This function is similar to pred_crosspolt,
    but it's doesn't adjust display range and calculate metrics.

    Parameters
    ----------
    obs_V : np.ndarray
        Observed V/I.
    pred_V : np.ndarray, 2d array with shape (num_electrode, 2)
        Predictive V/I.
    receive_date : str or int
        Date the data was received from electrode array.
    filepath : str, default is current directory
        The directory where figures are saved.
    mode : str
        Select the mode to manipulate crossplot.
        'save': save crossplot image.
        'show': show crossplot on screen.
        Others: return figure object.
    params : dict
        Dictionary of matplotlib's rcParams to manipulate plot setting.

    Returns
    -------
    fig : matplotlib's figure object
        The fig is returned when mode is neither 'save' nor 'show'.
    """
    # Temporarily change rcParam.
    # Restore the rc params from Matplotlib's internal defaults
    # before exiting this function.
    get_rcParams(params, update=True, figsize='s')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(obs_V, pred_V, s=2, c='blue', alpha=.5)
    ax.set_title(receive_date)
    ax.set_xlabel(r'Observed $\Delta V/I$')
    ax.set_ylabel(r'Predictive $\Delta V/I$')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    fig.tight_layout()

    if mode == 'save':
        filename = 'scatter_{}'.format(receive_date)
        fulname = os.path.join(filepath, filename)
        fig.savefig(fulname)
        plt.close(fig)
        mpl.rcdefaults()
    elif mode == 'show':
        plt.draw()
        plt.show()
        mpl.rcdefaults()
    else:
        plt.draw()
        mpl.rcdefaults()
        return fig


def structureplot_synth(synth_log_rho, pred_log_rho, xz, mode=None, save_dir='.', suffix=None, params=None):
    """
    Plot synthetic resistivity and predictive resistivity to illustrate subsurface structure.

    Parameters
    ----------
    synth_log_rho : numpy.ndarray
        Synthetic resistivity (ground truth).
    pred_log_rho : numpy.ndarray
        Predictive resistivity.
    xz : numpy.ndarray
        Electrode coordinates. (x, z)
    mode : str
        Select the mode to manipulate the drawing.
        'save': save image.
        'show': show on screen.
        Others: return figure object.
    save_dir : str
        The directory where figures are saved.
    suffix : str
        The suffix string for 'structureplot_' when 'Save' mode is selected.
    params : dict
        Dictionary of matplotlib's rcParams to manipulate plot setting.
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The fig is returned when mode is neither 'save' nor 'show'.
    """

    _params = {'image.aspect': 'auto',
               'image.cmap': 'jet',
               'lines.linestyle': 'None',
               'lines.marker': '.',
               'lines.markeredgecolor': 'k',
               'lines.markerfacecolor': 'k',
               'lines.markersize': '4.0'}
    if isinstance(params, dict):
        _params.update(params)
    get_rcParams(_params, figsize='l')
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
    ax0.plot(xz[:, 0], -xz[:, 1], clip_on=False, zorder=100)  # for electrodes
    ax1.plot(xz[:, 0], -xz[:, 1], clip_on=False, zorder=100)  # for electrodes

    # set properties
    levels = np.linspace(1, 3, 17, endpoint=True)
    nz, nx = pred_log_rho.shape
    vmin, vmax = 1, 3
    extent = (0, nx, nz, 0)
    cbar_size, cbar_pad = '3%', 0.1

    # plot synthetic resistivity
    im0 = ax0.imshow(synth_log_rho, vmin=vmin, vmax=vmax, extent=extent)
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size=cbar_size, pad=cbar_pad)
    cbar0 = fig.colorbar(im0, cax=cax, extend='both')
    cbar0.set_label(r'$\Omega-m (log_{10}\/scale)$')
    ax0.set_title('Synthetic resistivity')
    ax0.set_ylabel('Depth (m)')

    # plot predictive resistivity
    im1 = ax1.contourf(np.flipud(pred_log_rho), levels=levels, extent=extent, extend='both')
    ax1.invert_yaxis()
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
    cbar1 = fig.colorbar(im1, cax=cax)
    cbar1.set_label(r'$\Omega-m (log_{10}\/scale)$')
    ax1.set_title('Predictive resistivity')
    ax1.set_xlabel('Width (m)')
    ax1.set_ylabel('Depth (m)')
    fig.tight_layout()

    if mode == 'save':
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        filename = 'structureplot_{}'.format(suffix)
        fullname = os.path.join(save_dir, filename)
        fig.savefig(fullname)
        plt.close(fig)
        mpl.rcdefaults()
    elif mode == 'show':
        plt.draw()
        plt.show()
        mpl.rcdefaults()
    else:
        plt.draw()
        mpl.rcdefaults()
        return fig


def crossplot_synth(synth_V, pred_V, mode=None, save_dir='.', suffix=None, params=None):
    """
    Crossplot of synthetic equivalent resistivity and predictive equivalent resistivity.

    Parameters
    ----------
    synth_V : numpy.ndarray
        Synthetic equivalent resistivity (Potential difference divided by current, ground truth).
    pred_V : numpy.ndarray
        Predictive equivalent resistivity.
    mode : str
        Select the mode to manipulate the drawing.
        'save': save image.
        'show': show on screen.
        Others: return figure object.
    save_dir : str
        The directory where figures are saved.
    suffix : str
        The suffix string for 'crossplot_' when 'Save' mode is selected.
    params : dict
        Dictionary of matplotlib's rcParams to manipulate plot setting.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The fig is returned when mode is neither 'save' nor 'show'.
    """

    # get synth_V maximum and minimum
    x_min, x_max = synth_V[~np.isnan(synth_V)].min(), synth_V[~np.isnan(synth_V)].max()
    error_ratio = 0.1

    # calculate metrics
    # 1. Root mean squared relative error
    RMSRE = np.sqrt(np.mean(np.power((pred_V - synth_V) / synth_V, 2))) * 100
    # 2. Point ratio in the tolerance area
    in_ratio = np.mean((abs(pred_V - synth_V) / synth_V) <= error_ratio) * 100

    # plotting
    get_rcParams(params, figsize='s')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot([x_min, x_max], [x_min, x_max], 'r')
    ax.plot([x_min, x_max],
            [x_min - abs(x_min * error_ratio), x_max + abs(x_max * error_ratio)],
            color='green', linestyle='dashed', linewidth=1)
    ax.plot([x_min, x_max],
            [x_min + abs(x_min * error_ratio), x_max - abs(x_max * error_ratio)],
            color='green', linestyle='dashed', linewidth=1)
    ax.scatter(synth_V, pred_V, s=2, c='blue', alpha=.5)
    ax.set_xlabel(r'Synthetic $\Delta V/I$')
    ax.set_ylabel(r'Predictive $\Delta V/I$')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    lim = abs(x_max) if abs(x_max) > abs(x_min) else abs(x_min)
    r = 2 * lim
    # use escape sequence
    # \N{name}: Prints a character from the Unicode database
    metrics_str = 'RMSRE: {:{width}.{prec}f}%\n' \
                  'Within {:d}% error:\n  {:{width}.{prec}f}%' \
                  .format(RMSRE, int(error_ratio * 100), in_ratio, width=6, prec=4)
    props = dict(boxstyle='round', facecolor=(1, 0.5, 0.5), alpha=0.5)
    ax.text(-lim + 0.01 * r, lim - 0.01 * r, metrics_str,
            bbox=props, va='top', ha='left')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-lim * 1.1, lim * 1.1)
    ax.set_ylim(-lim * 1.1, lim * 1.1)
    fig.tight_layout()

    if mode == 'save':
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        filename = 'crossplot_{}'.format(suffix)
        fullname = os.path.join(save_dir, filename)
        fig.savefig(fullname)
        plt.close(fig)
        mpl.rcdefaults()
    elif mode == 'show':
        plt.draw()
        plt.show()
        mpl.rcdefaults()
    else:
        plt.draw()
        mpl.rcdefaults()
        return fig


def heatmap_synth(iterator, mode=None, save_dir='.', params=None):
    """
    Heatmap of synthetic resistivty and predictive resistivity.

    Parameters
    ----------
    iterator : iterator of os.DirEntry objects
        For iterating all pkl file in certain directory.
    mode : str
        Select the mode to manipulate the drawing.
        'save': save image.
        'show': show on screen.
        Others: return figure object.
    save_dir : str
        The directory where figures are saved.
    params : dict
        Dictionary of matplotlib's rcParams to manipulate plot setting.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The fig is returned when mode is neither 'save' nor 'show'.

    References
    ----------
    https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.histogram2d.html
    https://stackoverflow.com/questions/17237000/create-a-stacked-2d-histogram-using-different-weights
    """

    # setting
    # for calculating 2d histogram
    heatmap = 0  # initialize heatmap
    start, stop, num_edge = -2, 6, 161
    xedges = np.linspace(start, stop, num_edge)
    yedges = np.linspace(start, stop, num_edge)
    remain = 5  # for crop_zeros
    # for metric
    MSE = 0  # mean squared error
    # for plotting
    _params = {'image.cmap': 'jet',
               'lines.linewidth': 2,
               'lines.linestyle': '--'}  # predefined rcParams
    # color bar
    cbar_size, cbar_pad = '3%', 0.1

    # read data in specific directory and calculate 2d histogram
    for tmp, file in enumerate(iterator):
        data = read_pkl(file.path)
        synth_log_rho = data['synth_log_rho'].flatten()
        pred_log_rho = data['pred_log_rho'].flatten()
        hist, xedges, yedges = np.histogram2d(synth_log_rho, pred_log_rho, bins=(xedges, yedges))
        heatmap += hist
        MSE += np.square(np.subtract(synth_log_rho, pred_log_rho)).sum()

    try:
        MSE = MSE / ((tmp + 1) * synth_log_rho.size)  # mean squared error
    except NameError:
        raise ValueError('The iterator reaches the end or is empty.')

    bound = crop_zeros(heatmap, remain=remain, return_bound='only_bound')
    heatmap = heatmap[bound[0]:bound[1], bound[0]:bound[1]]
    xedges = xedges[bound[0]:bound[1] + 1]
    yedges = yedges[bound[0]:bound[1] + 1]

    # update rcParams
    if isinstance(params, dict):
        _params.update(params)
    get_rcParams(_params, figsize='s')

    fig, ax = plt.subplots()
    # heatmap
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, heatmap, norm=mpl.colors.LogNorm())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=cbar_size, pad=cbar_pad)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Count')

    # diagonal line
    ax.plot([xedges[0 + remain], xedges[-1 - remain]],
            [yedges[0 + remain], yedges[-1 - remain]])

    metrics_str = 'MSE: {:{width}.{prec}f}'.format(MSE, width=6, prec=4)
    props = dict(boxstyle='round', facecolor=(1, 0.5, 0.5), alpha=0.5)
    ax.text(xedges[0] + 0.05 * (xedges[-1] - xedges[0]),
            xedges[-1] - 0.05 * (xedges[-1] - xedges[0]),
            metrics_str, bbox=props, va='top', ha='left')

    # adjust property
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'Synthetic resistivity $log_{10}(\Omega-m)$')
    ax.set_ylabel(r'Predictive resistivity $log_{10}(\Omega-m)$')
    fig.tight_layout()

    # different mode
    if mode == 'save':
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fullname = os.path.join(save_dir, 'heatmap')
        fig.savefig(fullname)
        plt.close(fig)
        mpl.rcdefaults()
    elif mode == 'show':
        plt.draw()
        plt.show()
        mpl.rcdefaults()
    else:
        plt.draw()
        mpl.rcdefaults()
        return fig


def txrx_plot(true_V, pred_V, mode=None, save_dir='.', suffix=None, params=None):
    """
    Plot relative error (in log10 scale) of true equivalent resistivity and predictive equivalent resistivity, and
    arrange the result by transmitter pairs and receiver pairs.

    Parameters
    ----------
    true_V : numpy.ndarray
        True equivalent resistivity (Potential difference divided by current, ground truth).
    pred_V : numpy.ndarray
        Predictive equivalent resistivity.
    mode : str
        Select the mode to manipulate the drawing.
        'save': save image.
        'show': show on screen.
        Others: return figure object.
    save_dir : str
        The directory where figures are saved.
    suffix : str
        The suffix string for 'crossplot_' when 'Save' mode is selected.
    params : dict
        Dictionary of matplotlib's rcParams to manipulate plot setting.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The fig is returned when mode is neither 'save' nor 'show'.
    """

    misfit_percent = ((pred_V - true_V) / true_V) * 100

    _params = {'image.aspect': 'auto',
               'image.cmap': 'bwr',
               'image.origin': 'lower'}
    if isinstance(params, dict):
        _params.update(params)
    get_rcParams(_params, figsize='l')

    # set properties
    vmin, vmax = -100, 100
    extent = (0.5, true_V.shape[1] + 0.5, 0.5, true_V.shape[0] + 0.5)
    cbar_size, cbar_pad = '5%', 0.6
    # plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.imshow(misfit_percent, vmin=vmin, vmax=vmax, extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size=cbar_size, pad=cbar_pad)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Misfit of resistance (%)')
    ax.set_title('Data Misfit')
    ax.set_xlabel('Receiver ID')
    ax.set_ylabel('Transmitter ID')

    if mode == 'save':
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        filename = 'txrx_{}'.format(suffix)
        fullname = os.path.join(save_dir, filename)
        fig.savefig(fullname)
        plt.close(fig)
        mpl.rcdefaults()
    elif mode == 'show':
        plt.draw()
        plt.show()
        mpl.rcdefaults()
    else:
        plt.draw()
        mpl.rcdefaults()
        return fig


def plot_result_synth(iterator, num_figs, xz, save_dir='.'):
    """
    Convenient function for saving crossplot, structure plot and heatmap of synthetic data.

    Parameters
    ----------
    iterator : iterator of os.DirEntry objects
        For iterating all pkl file in certain directory.
    num_figs : int
        The number of figures to save.
    save_dir : str
        The directory where figures are saved.

    Returns
    -------
    None
    """

    # create another iterator that is the same as the original iterator
    two_iterator_tuple = itertools.tee(iterator, 2)

    num_figs = 1 if num_figs < 1 else num_figs
    i = 1
    for file in two_iterator_tuple[0]:
        data = read_pkl(file.path)
        synth_V = data['synth_V']
        pred_V = data['pred_V']
        synth_log_rho = data['synth_log_rho']
        pred_log_rho = data['pred_log_rho']

        suffix = re.findall(r'\d+', file.path)[0]
        crossplot_synth(synth_V, pred_V, mode='save', save_dir=save_dir, suffix=suffix)
        txrx_plot(synth_V, pred_V, mode='save', save_dir=save_dir, suffix=suffix)
        structureplot_synth(synth_log_rho, pred_log_rho, xz, mode='save', save_dir=save_dir, suffix=suffix)
        if i == num_figs:
            break
        else:
            i += 1
    heatmap_synth(two_iterator_tuple[1], mode='save', save_dir=save_dir)
