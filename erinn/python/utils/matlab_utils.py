"""Utilities calls matlab functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import subprocess
import warnings


# TODO:
#  Matlab function(pred_V_2) should be improved
#  Add a Matlab function(maybe fwd_simu.m) for fundemental prediction


def pred_V(src, dest=None, start=None, end=None):
    """Sub-processing calls the matlab function to predict V/I

    Parameters
    ----------
    src : str
        hdf5 file contain forward modeling parameters
    dest : str, default None
        Copy hdf5 file to new destination. The default is not to copy
    start : datetime.datetime object, default None
        Starting datetime. The default
    end : datetime.datetime object, default None
        Ending datetime.
    """

    dest = src if dest is None else dest
    if isinstance(start, datetime.datetime):
        start = start.strftime('%Y%m%d')
    else:
        warnings.warn(
            'start argument is not a datetime.datetime object.'
            ' Replace it with default.', Warning)
        start = None
    if isinstance(end, datetime.datetime):
        end = end.strftime('%Y%m%d')
    else:
        warnings.warn(
            'end argument is not a datetime.datetime object.'
            ' Replace it with default.', Warning)
        end = None

    matlab = ['matlab']
    options = ['-nodesktop', '-nosplash', '-wait', '-r']
    command = ['predict_V_2 {} {} Start {} End {}; quit;'.format(
        src, dest, start, end)]
    print(matlab + options + command)
    p = subprocess.Popen(matlab + options + command, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    print('stdout: {}'.format(stdout))
    print('stderr: {}'.format(stderr))
