from __future__ import division, absolute_import, print_function

from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Reshape
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model

# 第三種架構


def get_cnn1d_prelu(output_size, img_height, img_width, show=True):
    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = Reshape((img_height * img_width, 1))(model_input)
    x = Conv1D(filters=16, kernel_size=48, strides=2,
               padding='same', name='Conv1D_1')(x)
    x = Conv1D(filters=8, kernel_size=48, strides=2,
               padding='same', name='Conv1D_2')(x)
    x = Conv1D(filters=4, kernel_size=48, strides=2,
               padding='same', name='Conv1D_3')(x)
    x = Conv1D(filters=2, kernel_size=48, strides=2,
               padding='same', name='Conv1D_4')(x)
    x = Conv1D(filters=1, kernel_size=48, strides=2,
               padding='same', name='Conv1D_5')(x)
    x = Flatten(name='Flatten')(x)
    x = PReLU(name='PReLU_1')(x)
    x = BatchNormalization(name='BN_1')(x)
    x = Dense(512, activation='tanh', name='Dense_tanh_1')(x)
    x = BatchNormalization(name='BN_2')(x)
    x = Dense(512, activation='tanh', name='Dense_tanh_2')(x)
    cnn1d_output = Dense(output_size, activation='linear',
                         name='Output_Dense_linear')(x)
    cnn1d = Model(inputs=model_input, outputs=cnn1d_output, name='CNN1D_PReLU')

    if show:
        print('CNN1D_PReLU summary:')
        cnn1d.summary()
        print()

    return cnn1d


def get_cnn1d_relu(output_size, img_height, img_width, show=True):
    """ Use AlexNet architecture
    """
    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = Reshape((img_height * img_width, 1), name='Reshape_1')(model_input)
    x = Conv1D(filters=3, kernel_size=11,
               strides=2, activation='relu',
               padding='same', name='Conv1D_relu_1')(x)
    x = Conv1D(filters=8, kernel_size=5,
               strides=2, activation='relu',
               padding='same', name='Conv1D_relu_2')(x)
    x = Conv1D(filters=12, kernel_size=3,
               strides=2, activation='relu',
               padding='same', name='Conv1D_relu_3')(x)
    x = Conv1D(filters=12, kernel_size=3,
               strides=2, activation='relu',
               padding='same', name='Conv1D_relu_4')(x)
    x = Conv1D(filters=8, kernel_size=3,
               strides=2, activation='relu',
               padding='same', name='Conv1D_relu_5')(x)
    x = Flatten(name='Flatten_1')(x)
    x = Dense(512, activation='relu', name='Dense_relu_1')(x)
    x = Dense(512, activation='relu', name='Dense_relu_2')(x)
    # x = Dense(512, activation='relu', name='Dense_relu_3')(x)
    cnn1d_output = Dense(output_size, activation='linear',
                         name='Output_Dense_linear')(x)
    cnn1d = Model(inputs=model_input, outputs=cnn1d_output, name='CNN1D_relu')

    if show:
        print('CNN1D_relu summary:')
        cnn1d.summary()
        print()

    return cnn1d


def get_cnn1d_selu(output_size, img_height, img_width, show=True):
    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = Reshape((img_height * img_width, 1), name='Reshape_1')(model_input)
    x = Conv1D(filters=16, kernel_size=48,
               strides=2, activation='selu',
               padding='same', name='Conv1D_selu_1')(x)
    x = Conv1D(filters=8, kernel_size=48,
               strides=2, activation='selu',
               padding='same', name='Conv1D_selu_2')(x)
    x = Conv1D(filters=4, kernel_size=48,
               strides=2, activation='selu',
               padding='same', name='Conv1D_selu_3')(x)
    x = Conv1D(filters=2, kernel_size=48,
               strides=2, activation='selu',
               padding='same', name='Conv1D_selu_4')(x)
    x = Conv1D(filters=1, kernel_size=48,
               strides=2, activation='selu',
               padding='same', name='Conv1D_selu_5')(x)
    x = Flatten(name='Flatten_1')(x)
    x = Dense(512, activation='selu', name='Dense_selu_1')(x)
    x = Dense(512, activation='selu', name='Dense_selu_2')(x)
    cnn1d_output = Dense(output_size, activation='linear',
                         name='Output_Dense_linear')(x)
    cnn1d = Model(inputs=model_input, outputs=cnn1d_output, name='CNN1D_selu')

    if show:
        print('CNN1D_selu summary:')
        cnn1d.summary()
        print()

    return cnn1d
