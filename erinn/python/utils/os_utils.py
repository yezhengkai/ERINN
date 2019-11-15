from __future__ import division, absolute_import, print_function

import platform


class OSPlatform(object):
    """
    Check what operating system I am running.

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
