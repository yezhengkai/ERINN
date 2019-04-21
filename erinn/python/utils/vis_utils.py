from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


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
            (width, hight)

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


def structureplot_synth(rho, xz, nx, nz, filepath='.', mode=None, params=None):

    Params = {'image.cmap': 'jet'}
    if isinstance(params, dict):
        Params.update(params)
    get_rcParams(Params, figsize='3')

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 9.6))
    ax0, ax1 = ax
    fig.set_dpi(300)
    v = np.linspace(1, 3, 17, endpoint=True)
    im0 = ax0.contourf(np.flipud(y_test[index, :].reshape(nz, nx)), v,
                       extent=[0, nx, nz, 0], extend='both')
    im1 = ax1.contourf(np.flipud(y_pred[index, :].reshape(nz, nx)), v,
                       extent=[0, nx, nz, 0], extend='both')
    ax0.invert_yaxis(), ax1.invert_yaxis()
    cbar = fig.colorbar(im0, ax=ax.ravel().tolist(),
                        orientation='horizontal', ticks=v)
    cbar.set_label(r'$Resistivity \/\/ log_{10}(\Omega m)$', fontsize=14)
    cbar.ax.tick_params(axis='both', labelsize=12)
    ax0.set_title('y_true', fontsize=18), ax0.set_ylabel(
        'Depth(m)', fontsize=14)
    ax1.set_title('y_pred', fontsize=18), ax1.set_ylabel(
        'Depth(m)', fontsize=14)
    ax0.set_xlabel('Width(m)', fontsize=14), ax1.set_xlabel(
        'Width(m)', fontsize=14)
    ax0.tick_params(axis='both', labelsize=12), ax1.tick_params(
        axis='both', labelsize=12)
    plt.savefig(fulname + '_structure_%d.png' % index)
    plt.close(fig)

    Params = {'image.cmap': 'jet'}
    if isinstance(params, dict):
        Params.update(params)
    get_rcParams(Params, fig_type=3)
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


def crosspolt_synth(y_test, y_pred,
                    filepath='.', mode=None, params=None):
    """Crossplot of ground truth y and predictive y.

    Parameters
    ----------
    y_test : np.ndarray
        ground truth resistivity.
    y_pred : np.ndarray, 2d array with shape (num_electrode, 2)
        Predictive resistivity by NN.
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
    # cross plot
    # residual sum of squares
    SS_res = np.sum(np.square(y_test - y_pred))
    # total sum of squares
    SS_tot = np.sum(np.square(y_test - np.mean(y_test)))
    # R squared
    R2 = (1 - SS_res/(SS_tot + 1e-8))
    # Root Mean Square Error
    RMS = np.sqrt(np.mean(np.power(y_test - y_pred, 2)))
    # correlation coefficient
    corrcoef = np.corrcoef(y_test, y_pred)[0, 1]

    # resistivity crossplot
    fig = plt.figure(dpi=300)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.title(r'$corrcoef={:{width}.{prec}f},\/\/R^2={:{width}.{prec}f}$'.format(
        corrcoef, R2, width=6, prec=4), {'fontsize': 20})
    plt.scatter(y_test[index, :], y_pred[index, :], color='b')
    plt.plot([-0.5, 5], [-0.5, 5], 'k--')
    plt.xlim(-0.5, 5), plt.ylim(-0.5, 5)
    plt.xlabel(r'$True\/\/resistivity\/\/log_{10}(\Omega m)$', fontsize=14)
    plt.ylabel(
        r'$Predictive\/\/resistivity\/\/log_{10}(\Omega m)$', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig(fulname + '_RhoCrossPlot_%d.png' % index)
    plt.close(fig)

    # plotting
    get_rcParams(params, update=True, fig_type=1)
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

    # # check nan ratio
    # nb_nan = np.sum(np.isnan(obs_V))
    # nb_tot = obs_V.size
    # nan_ratio = nb_nan/nb_tot
    # if nan_ratio == 1:
    #     warnings.warn(f'nan ratio is 1 at {receive_date}. '
    #                   f'The crossplot is not drawn.', Warning)
    #     return
    # # residual sum of squares
    # SS_res = np.sum(
    #     np.square(obs_V[~np.isnan(obs_V)] - pred_V[~np.isnan(obs_V)]))
    # # total sum of squares
    # SS_tot = np.sum(
    #     np.square(obs_V[~np.isnan(obs_V)] - obs_V[~np.isnan(obs_V)].mean()))
    # # R squared
    # R2 = (1 - SS_res/(SS_tot + 1e-8))
    # # Root Mean Square Error
    # RMS = np.sqrt(np.mean(np.power(
    #     obs_V[~np.isnan(obs_V)] - pred_V[~np.isnan(obs_V)], 2)))
    # # RMS = np.sqrt(np.mean(np.power(
    # #     (pred_V[~np.isnan(obs_V)] - obs_V[~np.isnan(obs_V)]) / obs_V[~np.isnan(obs_V)], 2)))
    # # correlation coefficient
    # corrcoef = np.corrcoef(
    #     obs_V[~np.isnan(obs_V)], pred_V[~np.isnan(obs_V)])[0, 1]

    # # get obs_V maximum and minimum
    # x_min, x_max = obs_V[~np.isnan(obs_V)].min(), obs_V[~np.isnan(obs_V)].max()

    # # plotting
    # get_rcParams(params, update=True, fig_type=1)
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot([x_min, x_max], [x_min, x_max], 'r')
    # ax.scatter(obs_V, pred_V, s=2, c='blue', alpha=.5)
    # ax.set_title(receive_date)
    # ax.set_xlabel(r'Observed $\Delta V/I$')
    # ax.set_ylabel(r'Predictive $\Delta V/I$')
    # ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    # lim = abs(x_max) if abs(x_max) > abs(x_min) else abs(x_min)
    # r = 2 * lim
    # # use escape sequence
    # # \N{name}: Prints a character from the Unicode database
    # textstr = 'R\N{SUPERSCRIPT TWO}: {:{width}.{prec}f}\n'\
    #           'RMS: {:{width}.{prec}f}\n'\
    #           'corrcoef: {:{width}.{prec}f}\n'\
    #           'nan ratio: {:.2%}'.format(
    #               R2, RMS, corrcoef, nan_ratio, width=6, prec=4)
    # props = dict(boxstyle='round', facecolor=(1, 0.5, 0.5), alpha=0.5)
    # ax.text(-lim + 0.01 * r, lim - 0.01 * r, textstr, bbox=props,
    #         fontsize=10, va='top', ha='left')

    # ax.set_aspect('equal', adjustable='box')
    # ax.set_xlim(-lim * 1.1, lim * 1.1)
    # ax.set_ylim(-lim * 1.1, lim * 1.1)
    # fig.tight_layout()

    # if mode == 'save':
    #     filename = 'crossplot_{}'.format(receive_date)
    #     fulname = os.path.join(filepath, filename)
    #     fig.savefig(fulname)
    #     plt.close(fig)
    #     mpl.rcdefaults()
    # elif mode == 'show':
    #     plt.draw()
    #     plt.show()
    #     mpl.rcdefaults()
    # else:
    #     plt.draw()
    #     mpl.rcdefaults()
    #     return fig
