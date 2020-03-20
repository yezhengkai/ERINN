"""
Generate synthetic resistivity models randomly.

References
----------
https://stackoverflow.com/questions/44865023/circular-masking-an-image-in-python-using-numpy-arrays
https://docs.scipy.org/doc/numpy-1.16.0/reference/routines.random.html
https://numpy.org/doc/1.18/reference/random/index.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm
https://www.sicara.ai/blog/2019-01-28-how-computer-generate-random-numbers
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import numpy as np
from SimPEG.Utils.ModelBuilder import getIndicesBlock
from SimPEG.Utils.ModelBuilder import getIndicesSphere
from scipy.signal import convolve2d
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import uniform

from erinn.utils.io_utils import read_config_file


# TODO: Maybe use Generator instead of RandomState
# TODO: Check random property. Make sure every execution is different
def rand_num_shape(num_shape):
    """Randomly samples an integer from the space defined in the num_shape dictionary.

    Parameters
    ----------
    num_shape : dict
        num_`shape`, the `shape` can be a circle, a rectangle or other shape.

    Returns
    -------
    num : int
        A randomly sampled integer.
    """
    if num_shape['type'] == 'list':
        return int(np.random.RandomState().choice(num_shape['value'], 1))
    elif num_shape['type'] == 'range':
        num = np.arange(*num_shape['value'])
        return int(np.random.RandomState().choice(num, 1))
    else:
        raise ValueError(f"{num_shape['type']} is an invalid type.")


def rand_rect(x_bound, y_bound, w_range, h_range, mesh, num_rect):

    stack_rect = []
    for _ in range(num_rect):
        width = np.random.RandomState().uniform(w_range[0], w_range[1])
        height = np.random.RandomState().uniform(h_range[0], h_range[1])
        x_min = np.random.RandomState().uniform(x_bound[0], x_bound[1] - width)
        y_min = np.random.RandomState().uniform(y_bound[0], y_bound[1] - height)
        x_max = x_min + width
        y_max = y_min + height
        block_idx = getIndicesBlock([x_min, y_min],
                                    [x_max, y_max],
                                    mesh.gridCC)[0]  # return tuple. 1st is 1D integer array
        stack_rect.append((block_idx, 'rect'))
    return stack_rect


def rand_circle(center_x_bound, center_y_bound, radius_bound, mesh, num_circle):

    stack_circle = []
    for _ in range(num_circle):
        center = [np.random.RandomState().uniform(center_x_bound[0], center_x_bound[1]),
                  np.random.RandomState().uniform(center_y_bound[0], center_y_bound[1])]
        radius = np.random.RandomState().uniform(radius_bound[0], radius_bound[1])
        circle_idx = getIndicesSphere(center, radius, mesh.gridCC)  # return 1D bool array
        stack_circle.append((circle_idx, 'circle'))
    return stack_circle


def smooth2d(arr: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    """Smooth 2d array using moving average.

    Parameters
    ----------
    arr : array_like

    kernel_shape : sequence of ints

    Returns
    -------
    arr : numpy.ndarray

    """
    arr = np.asarray(arr)
    if not arr.ndim == 2:
        raise ValueError('The array to be smoothed must be a 2D array.')
    if len(kernel_shape) != 2:
        raise ValueError('The kernel_shape must be an integer sequence of 2 elements.')

    arr = convolve2d(arr, np.ones(kernel_shape), mode='same')
    normalize_matrix = convolve2d(np.ones(arr.shape),
                                  np.ones(kernel_shape), mode='same')
    return arr / normalize_matrix


def get_pd(**kwargs):
    """
    Get probability distribution.

    Returns
    -------
    pd : scipy.stats.rv_frozen
        Desired probability distribution.

    Other Parameters
    ----------------
    use_hidden : bool
    pdf : str
    scale : str
    a : float
    b : float
    hidden_for_a : list or tuple
    hidden_for_b : list or tuple
    hidden_pdf : str
    """
    allowed_kwargs = {'use_hidden', 'pdf', 'scale', 'a', 'b', 'hidden_for_a', 'hidden_for_b', 'hidden_pdf'}
    for key in allowed_kwargs:
        if key not in kwargs:
            raise ValueError('You did not input enough or correct keyword argument.')
    use_hidden = kwargs['use_hidden']
    pdf = kwargs['pdf']
    scale = kwargs['scale']
    a = kwargs['a']
    b = kwargs['b']
    hidden_for_a = kwargs['hidden_for_a']
    hidden_for_b = kwargs['hidden_for_b']
    hidden_pdf = kwargs['hidden_pdf']

    if use_hidden:
        if pdf == 'uniform':
            pd = RandUniform(hidden_for_a, hidden_for_b, scale, hidden_pdf)
        elif pdf == 'normal':
            if scale == 'linear':
                pd = RandTruncnorm(hidden_for_a, hidden_for_b,
                                   0, np.inf, scale, hidden_pdf)
            elif scale == 'log10':
                pd = RandTruncnorm(hidden_for_a, hidden_for_b,
                                   -np.inf, np.inf, scale, hidden_pdf)
            else:
                raise ValueError('You did not input enough or correct keyword argument.')
        else:
            raise ValueError('You did not input enough or correct keyword argument.')
    else:
        if pdf == 'uniform':
            pd = uniform(a, b - a)
        elif pdf == 'normal':
            if scale == 'linear':
                _a = (0 - a) / b
                _b = (np.inf - a) / b
                pd = truncnorm(_a, _b, loc=a, scale=b)
            elif scale == 'log10':
                pd = norm(loc=a, scale=b)
            else:
                raise ValueError('You did not input enough or correct keyword argument.')
        else:
            raise ValueError('You did not input enough or correct keyword argument.')

    return pd


def get_rvs(**kwargs):
    """Get random Variates.

    Returns
    -------
    rand_vars : numpy.ndarray

    Other Parameters
    ----------------
    use_hidden : bool
    scale : str
    pd : str
    size : list or tuple
    """
    allowed_kwargs = ['use_hidden', 'scale', 'pd', 'size']
    for key in allowed_kwargs:
        if key not in kwargs:
            raise ValueError('You did not input enough or correct keyword argument.')
    use_hidden = kwargs['use_hidden']
    scale = kwargs['scale']
    pd = kwargs['pd']
    size = kwargs['size']

    if use_hidden:
        rand_vars = pd.rvs(size=size)
        pd.new_pd()
    else:
        rand_vars = np.ones(size) * pd.rvs()

    if scale == 'log10':
        rand_vars = np.power(10, rand_vars)

    return rand_vars


class RandPd(object):

    def __init__(self, a_range, b_range, scale, hidden_pdf):
        self.a_range = a_range  # lower bound or mu(mean)
        self.b_range = b_range  # upper bound or std(standard deviation)
        self.scale = scale  # linear or log10
        self.hidden_pdf = hidden_pdf  # pdf for hidden variable
        if self.scale == 'linear':
            self.clip_a_for_a = (0 - self.a_range[0]) / self.a_range[1]
            self.clip_b_for_a = (np.inf - self.a_range[0]) / self.a_range[1]
            self.clip_a_for_b = (0 - self.b_range[0]) / self.b_range[1]
            self.clip_b_for_b = (np.inf - self.b_range[0]) / self.b_range[1]
        elif self.scale == 'log10':
            self.clip_a_for_a = (-np.inf - self.a_range[0]) / self.a_range[1]
            self.clip_b_for_a = (np.inf - self.a_range[0]) / self.a_range[1]
            self.clip_a_for_b = (-np.inf - self.b_range[0]) / self.b_range[1]
            self.clip_b_for_b = (np.inf - self.b_range[0]) / self.b_range[1]


class RandUniform(RandPd):
    def __init__(self, a_range, b_range, scale, hidden_pdf):
        super(RandUniform, self).__init__(a_range, b_range, scale, hidden_pdf)
        self._new_para()
        self.pd = uniform(loc=self.loc, scale=self.scale)
        self.pd.random_state.seed()  # re-seed

    def new_pd(self):
        self._new_para()
        self.pd = uniform(loc=self.loc, scale=self.scale)
        self.pd.random_state.seed()  # re-seed

    def rvs(self, *args, **kwargs):
        return self.pd.rvs(*args, **kwargs)

    def _new_para(self):
        if self.hidden_pdf == 'uniform':
            self.loc, self.scale = sorted([np.random.uniform(self.a_range[0], self.a_range[1]),
                                           np.random.uniform(self.b_range[0], self.b_range[1])])
            self.scale -= self.loc
        elif self.hidden_pdf == 'normal':
            self.loc, self.scale = sorted([truncnorm(self.clip_a_for_a, self.clip_b_for_a,
                                                     loc=self.a_range[0],
                                                     scale=self.a_range[1]).rvs(),
                                           truncnorm(self.clip_a_for_b, self.clip_b_for_b,
                                                     loc=self.b_range[0],
                                                     scale=self.b_range[1]).rvs()])
            self.scale -= self.loc

    def __repr__(self):
        return '\n'.join([f'loc: {self.loc}',
                          f'scale: {self.scale}'])


class RandTruncnorm(RandPd):

    def __init__(self, mu_range, std_range, clip_a, clip_b, scale, hidden_pdf):
        super(RandTruncnorm, self).__init__(mu_range, std_range, scale, hidden_pdf)
        self._overwrite_clip()
        self.clip_a = clip_a
        self.clip_b = clip_b
        self._new_para()
        self.pd = truncnorm(self._a, self._b, loc=self._mu, scale=self._std)
        self.pd.random_state.seed()  # re-seed

    def new_pd(self):
        self._new_para()
        self.pd = truncnorm(self._a, self._b, loc=self._mu, scale=self._std)
        self.pd.random_state.seed()  # re-seed

    def rvs(self, *args, **kwargs):
        return self.pd.rvs(*args, **kwargs)

    def _new_para(self):
        if self.hidden_pdf == 'uniform':
            self._mu, self._std = np.random.uniform(self.a_range[0], self.a_range[1]), \
                                  np.random.uniform(self.b_range[0], self.b_range[1])
            self._std = self._std if self._std > 0 else 0
        elif self.hidden_pdf == 'normal':
            self._mu, self._std = truncnorm(self.clip_a_for_a, self.clip_b_for_a,
                                            loc=self.a_range[0],
                                            scale=self.a_range[1]).rvs(), \
                                  truncnorm(self.clip_a_for_b, self.clip_b_for_b,
                                            loc=self.b_range[0],
                                            scale=self.b_range[1]).rvs()
            self._std = self._std if self._std > 0 else 0
        self._a = (self.clip_a - self._mu) / self._std
        self._b = (self.clip_b - self._mu) / self._std

    def _overwrite_clip(self):
        if self.scale == 'log10':
            self.clip_a_for_a = (-np.inf - self.a_range[0]) / self.a_range[1]
            self.clip_b_for_a = (np.inf - self.a_range[0]) / self.a_range[1]
            self.clip_a_for_b = (0 - self.b_range[0]) / self.b_range[1]
            self.clip_b_for_b = (np.inf - self.b_range[0]) / self.b_range[1]

    def __repr__(self):
        return '\n'.join([f'a: {self._a}',
                          f'b: {self._b}',
                          f'mu: {self._mu}',
                          f'std: {self._std}'])


def get_random_model(config_file, mesh, num_examples=None):

    config = read_config_file(config_file)
    x_bound = [np.nanmin(mesh.vectorNx), np.nanmax(mesh.vectorNx)]
    z_bound = [np.nanmin(mesh.vectorNy), np.nanmax(mesh.vectorNy)]
    kernel_shape = (config['z_kernel_size'], config['x_kernel_size'])
    if num_examples is None:
        num_examples = config['num_examples']

    # create the instance of resistivity "value" probability distribution
    # background
    pd_background = get_pd(use_hidden=config['use_hidden_background'],
                           pdf=config['pdf_background'],
                           scale=config['scale_background'],
                           a=config['a_background'],
                           b=config['b_background'],
                           hidden_for_a=(config['hidden_a_for_a_background'], config['hidden_b_for_a_background']),
                           hidden_for_b=(config['hidden_a_for_b_background'], config['hidden_b_for_b_background']),
                           hidden_pdf=config['hidden_pdf_background'])

    # rectangle(block)
    pd_rect = get_pd(use_hidden=config['use_hidden_rect'],
                     pdf=config['pdf_rect'],
                     scale=config['scale_rect'],
                     a=config['a_rect'],
                     b=config['b_rect'],
                     hidden_for_a=(config['hidden_a_for_a_rect'], config['hidden_b_for_a_rect']),
                     hidden_for_b=(config['hidden_a_for_b_rect'], config['hidden_b_for_b_rect']),
                     hidden_pdf=config['hidden_pdf_rect'])

    # circle
    pd_circle = get_pd(use_hidden=config['use_hidden_circle'],
                       pdf=config['pdf_circle'],
                       scale=config['scale_circle'],
                       a=config['a_circle'],
                       b=config['b_circle'],
                       hidden_for_a=(config['hidden_a_for_a_circle'], config['hidden_b_for_a_circle']),
                       hidden_for_b=(config['hidden_a_for_b_circle'], config['hidden_b_for_b_circle']),
                       hidden_pdf=config['hidden_pdf_circle'])

    for _ in range(num_examples):

        size = (mesh.nC,)
        resistivity = get_rvs(use_hidden=config['use_hidden_background'],
                              scale=config['scale_background'],
                              pd=pd_background,
                              size=size)

        # generate parameter for rectangle and circle
        num_rect = rand_num_shape(config['num_rect'])
        num_circle = rand_num_shape(config['num_circle'])

        stack = rand_rect(x_bound, z_bound,
                          config['h_range'],
                          config['w_range'],
                          mesh, num_rect)
        stack.extend(rand_circle(x_bound, z_bound,
                                 config['radius_bound'],
                                 mesh, num_circle))
        np.random.shuffle(stack)

        for _ in range(num_rect + num_circle):
            elem = stack.pop()
            size = resistivity[elem[0]].shape
            if elem[1] == 'rect':
                resistivity[elem[0]] = get_rvs(use_hidden=config['use_hidden_rect'],
                                               scale=config['scale_rect'],
                                               pd=pd_rect,
                                               size=size)
            elif elem[1] == 'circle':
                resistivity[elem[0]] = get_rvs(use_hidden=config['use_hidden_circle'],
                                            scale=config['scale_circle'],
                                            pd=pd_circle,
                                            size=size)
            else:
                raise NotImplementedError()

        resistivity = smooth2d(resistivity.reshape(mesh.nCy, mesh.nCx),
                               kernel_shape)  # shape is (nz, nx)
        # The resistivity starts at the bottom left of the SimPEG 2d mesh.
        yield resistivity.flatten()
