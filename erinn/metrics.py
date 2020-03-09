"""Custom metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import numpy as np
from tensorflow.keras import backend as K


def r_squared(y_true, y_pred):
    """Coefficient of determination(R-squared)

    This function is defined for Keras custom metrics.

    Parameters
    ----------
    y_true : tf.Tensor or np.ndarray
        True value
    y_pred : tf.Tensor or np.ndarray
        Predictive value

    Note
    ----
    The definition is referenced from the wikipedia.
    You can check it again at the following URL.
    https://en.wikipedia.org/wiki/Coefficient_of_determination

    References
    ----------
    https://stackoverflow.com/questions/42351184/how-to-calculate-r2-in-tensorflow
    https://www.kaggle.com/rohumca/linear-regression-in-tensorflow
    """

    try:
        # residual sum of squares
        ss_res = K.sum(K.square(y_true - y_pred))
        # ss_res = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
        # total sum of squares
        ss_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)))
        # ss_tot = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true, axis=0))))
        # R squared
        r2 = (1 - ss_res/(ss_tot + K.epsilon()))
        # r2 = tf.subtract(1.0, tf.div(ss_res, ss_tot))
    except:
        # residual sum of squares
        ss_res = np.sum(np.square(y_true - y_pred))
        # total sum of squares
        ss_tot = np.sum(np.square(y_true - np.mean(y_true, axis=0)))
        r2 = (1 - ss_res/(ss_tot + 1e-7))
    return r2
