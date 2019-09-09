"""
Generate synthetic conductivity models randomly.

References
----------
https://stackoverflow.com/questions/44865023/circular-masking-an-image-in-python-using-numpy-arrays
https://docs.scipy.org/doc/numpy-1.16.0/reference/routines.random.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from scipy.signal import convolve2d
from scipy.stats import norm, truncnorm, uniform

from ..utils.io_utils import read_config_file


def rand_rect(x_bound, y_bound, w_range, h_range, num_rect, dtype='int'):
    if dtype != 'int' or dtype != 'float':
        dtype = 'int'

    stack_rect = []
    if dtype == 'int':
        for _ in range(num_rect):
            width = np.random.randint(w_range[0], w_range[1] + 1)
            height = np.random.randint(h_range[0], h_range[1] + 1)
            x_min = np.random.randint(x_bound[0], x_bound[1] - width + 1)
            y_min = np.random.randint(x_bound[0], y_bound[1] - height + 1)
            stack_rect.append([x_min, y_min, width, height])
    elif dtype == 'float':
        for _ in range(num_rect):
            width = np.random.uniform(w_range[0], w_range[1])
            height = np.random.uniform(h_range[0], h_range[1])
            x_min = np.random.uniform(x_bound[0], x_bound[1] - width)
            y_min = np.random.uniform(x_bound[0], y_bound[1] - height)
            stack_rect.append([x_min, y_min, width, height])
    return stack_rect


def rand_circle(w, h, center_x_bound, center_y_bound, radius_bound, num_circle):
    """

    """

    stack_mask = []
    for _ in range(num_circle):
        center = [np.random.uniform(center_x_bound[0], center_x_bound[1]),
                  np.random.uniform(center_y_bound[0], center_y_bound[1])]
        radius = np.random.uniform(radius_bound[0], radius_bound[1])
        mask = create_circular_mask(h, w, center=center, radius=radius)
        stack_mask.append(mask)
    return stack_mask


def create_circular_mask(h, w, center=None, radius=None):
    """

    References
    ----------
    .. [1] Stackoverflow, "Circular masking an image in python using numpy arrays",
        https://stackoverflow.com/questions/44865023/circular-masking-an-image-in-python-using-numpy-arrays

    """
    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def smooth2d(arr: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    """
    Smooth 2d array using moving average.

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

    essential_keys = ['use_hidden', 'pdf', 'scale', 'a', 'b', 'hidden_for_a', 'hidden_for_b', 'hidden_pdf']

    for key in essential_keys:
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

    def new_pd(self):
        self._new_para()
        self.pd = uniform(loc=self.loc, scale=self.scale)

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

    def new_pd(self):
        self._new_para()
        self.pd = truncnorm(self._a, self._b, loc=self._mu, scale=self._std)

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


def get_rand_model(config_file, num_samples=None):

    config = read_config_file(config_file)
    x_bound = [0, config['nx']]
    z_bound = [0, config['nz']]
    kernel_shape = (config['x_kernel_size'], config['z_kernel_size'])
    if num_samples is None:
        num_samples = config['num_samples']

    # create the instance of probability instance
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

    for _ in range(num_samples):

        size = (x_bound[1], z_bound[1])
        resistivity = get_rvs(use_hidden=config['use_hidden_background'],
                              scale=config['scale_background'],
                              pd=pd_background,
                              size=size)

        # generate parameter for rectangle and circle
        stack = rand_rect(z_bound, x_bound, config['h_range'], config['w_range'], config['num_rect'])
        stack.extend(rand_circle(z_bound[1], x_bound[1],
                                 z_bound, x_bound,
                                 config['radius_bound'], config['num_circle']))
        np.random.shuffle(stack)

        for _ in range(config['num_rect'] + config['num_circle']):
            elem = stack.pop()
            if len(elem) == 4:
                size = (elem[3], elem[2])
                resistivity[elem[1]:elem[1] + elem[3],
                            elem[0]:elem[0] + elem[2]] = get_rvs(use_hidden=config['use_hidden_rect'],
                                                                 scale=config['scale_rect'],
                                                                 pd=pd_rect,
                                                                 size=size)
            else:
                size = resistivity[elem].shape
                resistivity[elem] = get_rvs(use_hidden=config['use_hidden_circle'],
                                            scale=config['scale_circle'],
                                            pd=pd_circle,
                                            size=size)

        resistivity = smooth2d(resistivity, kernel_shape)  # shape is (nx, nz)
        sigma = 1 / resistivity.T  # shape is (nz, nx)

        yield sigma.flatten()
