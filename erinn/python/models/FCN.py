from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, Reshape, Dropout
from tensorflow.python.keras.layers.advanced_activations import PReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model


def get_fcn_relu(output_size, height, width, channel, show=True):
    nb_filter = [8, 16, 32, 64, 128]
    # nb_filter = [32, 64, 128, 256, 512]
    pool_size = (1, 2)
    padding_size = ((0, 0), (3, 4))

    model_input = Input(shape=(height, width, channel), name='main_input')
    conv1_1 = Conv2D()(model_input)



