from __future__ import absolute_import, division, print_function

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model


# 第一種架構: 深度前饋網路(deep feedforward network)
# 也叫做前饋神經網路(feedforward neural network)或多層感知機(multilayer perceptron, MLP)
def get_dfn(output_size, img_height, img_width, show=True):

    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = Dense(256, activation='selu', name='Dense_selu_1')(model_input)
    x = BatchNormalization(name='BN_1')(x)
    x = Dense(256, activation='tanh', name='Dense_tanh_1')(x)
    x = BatchNormalization(name='BN_2')(x)
    x = Dense(256, activation='tanh', name='Dense_tanh_2')(x)
    dfn_output = Dense(output_size, activation='linear',
                       name='Output_Dense_linear')(x)
    dfn = Model(inputs=model_input, outputs=dfn_output, name='DFN')

    if show:
        print('DFN summary:')
        dfn.summary()
        print()

    return dfn


def get_dfn_relu(output_size, img_height, img_width, show=True):

    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = BatchNormalization(name='BN_1')(model_input)
    x = Dense(256, activation='relu', name='Dense_relu_1')(x)
    # x = BatchNormalization()(x)
    x = Dense(256, activation='relu', name='Dense_relu_2')(x)
    # x = BatchNormalization()(x)
    x = Dense(256, activation='relu', name='Dense_relu_3')(x)
    dfn_output = Dense(output_size, activation='linear',
                       name='Output_Dense_linear')(x)
    dfn = Model(inputs=model_input, outputs=dfn_output, name='DFN_relu')

    if show:
        print('DFN_relu summary:')
        dfn.summary()
        print()

    return dfn


def get_dfn_selu(output_size, img_height, img_width, show=True):

    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = BatchNormalization()(model_input)
    x = Dense(256, activation='selu', name='Dense_selu_1')(x)
    # x = BatchNormalization()(x)
    x = Dense(256, activation='selu', name='Dense_selu_2')(x)
    # x = BatchNormalization()(x)
    x = Dense(256, activation='selu', name='Dense_selu_3')(x)
    dfn_output = Dense(output_size, activation='linear',
                       name='Output_Dense_linear')(x)
    dfn = Model(inputs=model_input, outputs=dfn_output, name='DFN_selu')

    if show:
        print('DFN_selu summary:')
        dfn.summary()
        print()

    return dfn
