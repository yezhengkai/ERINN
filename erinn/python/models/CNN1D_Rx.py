from __future__ import division, absolute_import, print_function

from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Reshape
from tensorflow.keras.models import Model

# 第五種架構


def get_cnn1d_rx_relu(output_size, img_height, img_width, show=True):

    # value with same Rx would use in same channel
    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = Reshape((img_height, img_width), name='Reshape_1')(model_input)
    # x = Permute((2, 1), name='Permute_1')(x)
    x = Conv1D(filters=512, kernel_size=11, strides=1,
               padding='same', activation='relu',
               name='Conv1D_relu_1')(x)
    x = Conv1D(filters=256, kernel_size=5, strides=1,
               padding='same', activation='relu',
               name='Conv1D_relu_2')(x)
    x = Conv1D(filters=128, kernel_size=3, strides=1,
               padding='same', activation='relu',
               name='Conv1D_relu_3')(x)
    x = Conv1D(filters=64, kernel_size=3, strides=1,
               padding='same', activation='relu',
               name='Conv1D_relu_4')(x)
    x = Conv1D(filters=32, kernel_size=3, strides=1,
               padding='same', activation='relu',
               name='Conv1D_relu_5')(x)
    # x = Conv1D(filters=8, kernel_size=48, strides=1,
    #            padding='same', activation='relu',
    #            name='Conv1D_relu_6')(x)
    x = Flatten(name='Flatten_1')(x)
    x = Dense(512, activation='relu', name='Dense_relu_1')(x)
    x = Dense(512, activation='relu', name='Dense_relu_2')(x)
    # x = Dense(512, activation='relu', name='Dense_relu_3')(x)
    cnn1d_output = Dense(output_size, activation='linear',
                         name='Output_Dense_linear')(x)
    cnn1d = Model(inputs=model_input, outputs=cnn1d_output,
                  name='CNN1D_Rx')

    if show:
        print('CNN1D Rx summary:')
        cnn1d.summary()
        print()

    return cnn1d
