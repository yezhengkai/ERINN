# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import CNN
from . import CNN1D
from . import CNN1D_Rx
from . import CNN1D_Tx
from . import DFN

# Globally-importable models.
from .CNN import get_cnn_relu
from .CNN1D import get_cnn1d_relu
from .CNN1D_Rx import get_cnn1d_rx_relu
from .CNN1D_Tx import get_cnn1d_tx
from .CNN1D_Tx import get_cnn1d_tx_relu
from .DFN import get_dfn_relu
