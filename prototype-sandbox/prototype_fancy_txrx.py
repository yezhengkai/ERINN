# %%
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from numba import njit

from erinn.utils.io_utils import read_pkl

FILEDIR = os.path.dirname(__file__)

# %%
workdir = os.path.join(
    FILEDIR, '..', 'ERI', 'template-python', 'scripts', 'preprocessing'
)
os.chdir(workdir)

resistance_pkl =  os.path.join(
    '..', '..', 'data', 'trial1',
    'training', 'resistance', 'raw', '000001.pkl'
)
simulator_pkl = os.path.join(
    '..', '..', 'data', 'trial1', 'simulator.pkl'
)

resistance = read_pkl(resistance_pkl)
simulator = read_pkl(simulator_pkl)
abmn_id = simulator.urf.abmn_id
num_electrode = len(
    np.unique(
        np.hstack(
            (simulator.urf.Tx_id.flatten(), simulator.urf.Rx_id.flatten())
        )
    )
)

# %%
# Index column is coresponding to A/M, B/N, space
Index = np.array(
    sorted(list(combinations(np.arange(1, num_electrode + 1), 2)),
           key=lambda ab: ab[1] - ab[0])
)
Index = np.hstack((Index, np.expand_dims(Index[:, 1] - Index[:, 0], axis=1)))
# np.expand_dims(Index[:, 1] - Index[:, 0], axis=1)  # Equivalent to x[:, np.newaxis]

# %%
# References:
# https://stackoverflow.com/questions/21800169/python-pandas-get-index-of-rows-which-column-matches-certain-value

VoverI = np.empty((len(Index), len(Index)))
VoverI.fill(np.nan)

@njit
def assign_VoverI_inplace(VoverI, Index, abmn_id, resistance):
    for i, res in enumerate(resistance):
        index_AB = np.flatnonzero(
            np.logical_and(
                abmn_id[i, 0] == Index[:, 1],
                abmn_id[i, 1] == Index[:, 0]
            )
        )
        index_MN = np.flatnonzero(
            np.logical_and(
                abmn_id[i, 2] == Index[:, 0],
                abmn_id[i, 3] == Index[:, 1]
            )
        )
        VoverI[index_MN[0], index_AB[0]] = res

assign_VoverI_inplace(VoverI, Index, abmn_id, resistance)

# %%
def ezplot(VoverI, x_range, y_range):
    fig, ax = plt.subplots(dpi=600)
    # im = ax.pcolormesh(VoverI[np.ix_(y_range, x_range)])
    im = ax.imshow(VoverI[np.ix_(y_range, x_range)], origin='lower', aspect='auto')
    ax.set_title('TxRx')
    ax.set_xlabel('Tx')
    ax.set_ylabel('Rx')
    fig.colorbar(im, ax=ax)
    plt.show()
ezplot(VoverI, np.arange(1000), np.arange(1000))

# %%
# References:
# https://stackoverflow.com/questions/20528328/numpy-logical-or-for-more-than-two-arguments
# https://stackoverflow.com/questions/42309460/boolean-masking-on-multiple-axes-with-numpy
uni_Tx_delta = np.unique(np.abs(abmn_id[:, 0] - abmn_id[:, 1]))
uni_Rx_delta = np.unique(np.abs(abmn_id[:, 2] - abmn_id[:, 3]))

used_Tx_delta_mask = np.logical_or.reduce(list(Index[:, 2] == Tx_delta for Tx_delta in uni_Tx_delta))
used_Rx_delta_mask = np.logical_or.reduce(list(Index[:, 2] == Rx_delta for Rx_delta in uni_Rx_delta))
# np.any(list(Index[:, 2] == Tx_delta for Tx_delta in uni_Tx_delta), axis=0)
tmp = VoverI[np.ix_(used_Rx_delta_mask, used_Tx_delta_mask)]
ezplot(tmp, np.arange(480), np.arange(2016))

# %%
# remove all nan row
tmp = VoverI[~np.isnan(VoverI).all(axis=1)]
ezplot(tmp, np.arange(480), np.arange(2012))
# remove all nan column
tmp = tmp[:, ~np.isnan(tmp).all(axis=0)]
ezplot(tmp, np.arange(120), np.arange(2012))


# %%
# Organize the above code snippets into a function

@njit
def assign_resistance_inplace(resistance, new_resistance, Index, abmn_id):
    for i, res in enumerate(resistance):
        if abmn_id[i, 0] < abmn_id[i, 1]:
            index_AB = np.flatnonzero(
                np.logical_and(
                    abmn_id[i, 0] == Index[:, 0],
                    abmn_id[i, 1] == Index[:, 1]
                )
            )
        else:
            index_AB = np.flatnonzero(
                np.logical_and(
                    abmn_id[i, 0] == Index[:, 1],
                    abmn_id[i, 1] == Index[:, 0]
                )
            )
        if abmn_id[i, 2] < abmn_id[i, 3]:
            index_MN = np.flatnonzero(
                np.logical_and(
                    abmn_id[i, 2] == Index[:, 0],
                    abmn_id[i, 3] == Index[:, 1]
                )
            )
        else:
            index_MN = np.flatnonzero(
                np.logical_and(
                    abmn_id[i, 2] == Index[:, 1],
                    abmn_id[i, 3] == Index[:, 0]
                )
            )
        new_resistance[index_AB[0], index_MN[0]] = res

def to_txrx_sort_by_spacing(array, abmn_id, num_electrode, value=0.0, dim=3):
    Index = np.array(
        sorted(list(combinations(np.arange(1, num_electrode + 1), 2)),
               key=lambda ab: ab[1] - ab[0])
    )
    Index = np.hstack((Index, np.expand_dims(Index[:, 1] - Index[:, 0], axis=1)))
    new_array = np.ones((len(Index), len(Index)), dtype=np.float) * value
    assign_resistance_inplace(array, new_array, Index, abmn_id)
    
    if remove_blank:
        if conpact:
            # remove all `value` row
            tmp = VoverI[~np.isnan(VoverI).all(axis=1)]
            # remove all `value` column
            tmp = tmp[:, ~np.isnan(tmp).all(axis=0)]

    if dim == 3:
        new_array = np.expand_dims(new_array, axis=2)  # reshape to 3D array
    elif dim == 2:
        new_array = new_array
    else:
        raise ValueError('`dim` accepts only 2 or 3')

    return new_array


resistance = read_pkl(resistance_pkl)
simulator = read_pkl(simulator_pkl)
abmn_id = simulator.urf.abmn_id
num_electrode = len(
    np.unique(
        np.hstack(
            (simulator.urf.Tx_id, simulator.urf.Rx_id)
        )
    )
)

fancy_txrx = to_txrx_sort_by_spacing(resistance, abmn_id, num_electrode, value=np.nan, dim=3)

# %%
