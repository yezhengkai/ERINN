"""OS-related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import os
import platform


class OSPlatform(object):
    """Check what operating system I am running.

    Parameters
    ----------
    None

    References
    ----------
    .. [1] https://stackoverflow.com/questions/1854/python-what-os-am-i-running-on

    Examples
    --------
    >>> _os = OSPlatform()
    >>> _os.is_WINDOWS  # If you run it on Windows, it will return True
    True
    >>> _os.is_LINUX  # If you run it on Windows, it will return False
    False
    >>> _os.is_MAC  # If you run it on Windows, it will return False
    False
    """

    def __init__(self):
        self.is_MAC = self.check_platform("darwin")
        self.is_LINUX = self.check_platform("linux")
        self.is_WINDOWS = self.check_platform("windows")

    @staticmethod
    def check_platform(os_name):
        return os_name.lower() == platform.system().lower()


def next_path(path_pattern, only_num=False):
    """Finds the next free path.

    Finds the next free path in an sequentially named list of files.
    Runs in log(n) time where n is the number of existing files in sequence.

    Parameters
    ----------
    path_pattern : str
        The pattern contains numbers.
    only_num : bool
        Whether to return only numbers.

    Returns
    -------
    str or int

    Examples
    --------
    >>> import shutil
    >>> from pathlib import Path
    >>> Path('./tmp').mkdir()
    >>> Path('./tmp/file-1.txt').touch()
    >>> Path('./tmp/file-2.txt').touch()
    >>> Path('./tmp/file-3.txt').touch()
    >>> path_pattern = './tmp/file-%s.txt'
    >>> print(next_path(path_pattern))
    ./tmp/file-4.txt
    >>> shutil.rmtree('./tmp', ignore_errors=True)

    References
    ----------
    .. [1] https://stackoverflow.com/questions/17984809/how-do-i-create-a-incrementing-filename-in-python
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    if only_num:
        return b
    else:
        return path_pattern % b
