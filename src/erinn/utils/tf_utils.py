"""TensorFlow-related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import numpy as np
import tensorflow as tf


def tf_log(tensor, base=np.exp(1.)):
    r"""TensorFlow logarithmic operation.

    Parameters
    ----------
    tensor : tensorflow.Tensor
        Tensor for logarithmic operation.
    base : int or float, default np.exp(1.)
        Base of logarithmic operation.

    Returns
    -------
    new_tensor : tensorflow.Tensor
        Tensor after logarithmic operation.

    Notes
    -----
    Logarithmic identity
    .. math:: \log_a{b}=\frac{\log_c{b}}{\log_c{a}}

    References
    ----------
    .. [1] https://github.com/tensorflow/tensorflow/issues/1666#issuecomment-202453841
    .. [2] https://en.wikipedia.org/wiki/List_of_logarithmic_identities
    """
    numerator = tf.math.log(tensor)
    denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
    new_tensor = numerator / denominator
    return new_tensor
