"""Time-related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import datetime


def datetime_range(start, end, fmt='%Y%m%d'):
    """Change `start` and `end` string to datetime object.

    Parameters
    ----------
    start : str
        Starting datetime string.
    end : str
        Ending datetime string.
    fmt : str, default '%Y%m%d'
        Format for parsing start and end string.

    Returns
    ------
    start : datetime.datetime
        Desired `start` datetime.datetime object.
    end : datetime.datetime
        Desired `end` datetime.datetime object.
    """

    if start is not None and end is not None:
        start = datetime.datetime.strptime(start, fmt)
        end = datetime.datetime.strptime(end, fmt)
    elif start is not None and end is None:
        start = datetime.datetime.strptime(start, fmt)
        end = datetime.datetime.max
    elif start is None and end is not None:
        start = datetime.datetime.min
        end = datetime.datetime.strptime(end, fmt)
    elif start is None and end is None:
        start = datetime.datetime.min
        end = datetime.datetime.max

    return start, end


def datetime_in_range(t, start, end, fmt='%Y%m%d'):
    """Check if the date is within the specified datetime range.

    Parameters
    ----------
    t : datetime.datetime object
        Datetime that you want to check.
    start : datetime.datetime object
        Starting datetime.
    end : datetime.datetime object
        Ending datetime.
    fmt : str, default '%Y%m%d'
        Format for parsing start and end string.

    Returns
    -------
    bool
        Return True if t is within the time range
    """
    # check all arguments are datetime.datetime object
    if not isinstance(start, datetime.datetime):
        start = datetime.datetime.strptime(str(start), fmt)
    if not isinstance(end, datetime.datetime):
        end = datetime.datetime.strptime(str(end), fmt)
    if not isinstance(t, datetime.datetime):
        t = datetime.datetime.strptime(str(t), fmt)
    # check start <= end
    if start > end:
        start, end = end, start
    return start <= t <= end
