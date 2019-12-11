from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Reshape, Dropout
from tensorflow.keras.layers.advanced_activations import PReLU
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model


# 第二種架構
# convolutional neural network


def get_cnn_prelu(output_size, img_height, img_width, show=True):
    """"""
    img_channels = 1
    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = Reshape((img_height, img_width, img_channels))(model_input)
    x = Conv2D(filters=6, kernel_size=(16, 12),
               # activation='relu',
               strides=(2, 2), padding='same',
               kernel_initializer="he_normal",
               # kernel_regularizer=l2(0.00001),
               data_format="channels_last",
               name='Conv2D_1')(x)
    x = PReLU(name='PReLU_1')(x)
    c1_output = Dropout(0.4, noise_shape=None, seed=None, name='Dropout_1')(x)
    # c1_output = MaxPooling2D(pool_size=(2, 2)) (x)
    x = BatchNormalization(name='BN_1')(c1_output)
    x = Conv2D(filters=3, kernel_size=(16, 12),
               # activation='relu',
               strides=(2, 2), padding='same',
               kernel_initializer="he_normal",
               # kernel_regularizer=l2(0.00001),
               data_format="channels_last",
               name='Conv2D_2')(x)
    x = PReLU(name='PReLU_2')(x)
    c2_output = Dropout(0.4, noise_shape=None, seed=None)(x)
    # c2_output = MaxPooling2D(pool_size=(2, 2)) (x)
    x = BatchNormalization(name='BN_2')(c2_output)
    x = Conv2D(filters=1, kernel_size=(8, 12),
               # activation='relu',
               strides=(2, 2), padding='same',
               kernel_initializer="he_normal",
               # kernel_regularizer=l2(0.00001),
               data_format="channels_last",
               name='Conv2D_3')(x)
    x = PReLU(name='PReLU_3')(x)
    c3_output = Dropout(0.4, noise_shape=None, seed=None)(x)
    # c3_output = MaxPooling2D(pool_size=(2, 2)) (x)
    # x = BatchNormalization()(c3_output)
    # x = Conv2D(filters=1, kernel_size=(2, 3),
    # activation='relu',
    # strides=(2, 3), padding='same',
    # kernel_initializer="he_normal",
    # kernel_regularizer=l2(0.00001),
    # data_format="channels_last") (x)
    # C4_output = Dropout(0.4, noise_shape=None, seed=None)(x)
    # C4_output = MaxPooling2D(pool_size=(2, 2)) (x)
    x = Flatten(name='Flatten')(c3_output)
    # x = BatchNormalization()(x)
    x = Dense(128, activation='selu', name='Dense_selu_1')(x)
    x = Dropout(0.4, noise_shape=None, seed=None)(x)
    x = BatchNormalization(name='BN_3')(x)
    x = Dense(512, activation='selu', name='Dense_tanh_1')(x)
    x = Dropout(0.4, noise_shape=None, seed=None)(x)
    # x = BatchNormalization()(x)
    x = Dense(512, activation='selu', name='Dense_selu_2')(x)
    x = Dropout(0.4, noise_shape=None, seed=None)(x)
    x = BatchNormalization(name='BN_4')(x)
    x = Dense(512, activation='selu', name='Dense_tanh_2')(x)
    x = Dropout(0.4, noise_shape=None, seed=None)(x)
    x = BatchNormalization(name='BN_5')(x)
    # x = Dense(512, activation='tanh')(x)
    # x = Dropout(0.4, noise_shape=None, seed=None)(x)
    # x = BatchNormalization()(x)
    cnn_output = Dense(output_size, activation='linear',
                       name='Output_Dense_linear')(x)
    cnn = Model(inputs=model_input, outputs=cnn_output, name='CNN_PReLU')

    if show:
        print('CNN_PReLU summary:')
        cnn.summary()
        print()

    return cnn


def get_cnn_relu(output_size, img_height, img_width, show=True):
    """ Use AlexNet architecture
    """
    img_channels = 1
    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = Reshape((img_height, img_width, img_channels),
                name='Reshape_1')(model_input)
    x = Conv2D(filters=12, kernel_size=(11, 11),
               activation='relu',
               strides=(2, 2), padding='same',
               data_format="channels_last",
               name='Conv2D_relu_1')(x)
    x = Conv2D(filters=32, kernel_size=(5, 5),
               activation='relu',
               strides=(2, 2), padding='same',
               data_format="channels_last",
               name='Conv2D_relu_2')(x)
    x = Conv2D(filters=48, kernel_size=(3, 3),
               activation='relu',
               strides=(2, 2), padding='same',
               data_format="channels_last",
               name='Conv2D_relu_3')(x)
    x = Conv2D(filters=48, kernel_size=(3, 3),
               activation='relu',
               strides=(2, 2), padding='same',
               data_format="channels_last",
               name='Conv2D_relu_4')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3),
               activation='relu',
               strides=(2, 2), padding='same',
               data_format="channels_last",
               name='Conv2D_relu_5')(x)
    x = Flatten(name='Flatten_1')(x)
    x = Dense(512, activation='relu', name='Dense_relu_1')(x)
    x = Dense(512, activation='relu', name='Dense_relu_2')(x)
    # x = Dense(512, activation='relu', name='Dense_relu_3')(x)
    # x = Dense(512, activation='relu', name='Dense_relu_4')(x)
    cnn_output = Dense(output_size, activation='linear',
                       name='Output_Dense_linear')(x)
    cnn = Model(inputs=model_input, outputs=cnn_output, name='CNN_relu')

    if show:
        print('CNN_relu summary:')
        cnn.summary()
        print()

    return cnn


def get_cnn_selu(output_size, img_height, img_width, show=True):
    """"""
    img_channels = 1
    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = Reshape((img_height, img_width, img_channels),
                name='Reshape_1')(model_input)
    x = Conv2D(filters=6, kernel_size=(16, 12),
               activation='selu',
               strides=(2, 2), padding='same',
               data_format="channels_last",
               name='Conv2D_selu_1')(x)
    x = Conv2D(filters=3, kernel_size=(16, 12),
               activation='selu',
               strides=(2, 2), padding='same',
               data_format="channels_last",
               name='Conv2D_selu_2')(x)
    x = Conv2D(filters=1, kernel_size=(8, 12),
               activation='selu',
               strides=(2, 2), padding='same',
               data_format="channels_last",
               name='Conv2D_selu_3')(x)
    x = Flatten(name='Flatten_1')(x)
    x = Dense(128, activation='selu', name='Dense_selu_1')(x)
    x = Dense(512, activation='selu', name='Dense_selu_2')(x)
    x = Dense(512, activation='selu', name='Dense_selu_3')(x)
    x = Dense(512, activation='selu', name='Dense_selu_4')(x)
    cnn_output = Dense(output_size, activation='linear',
                       name='Output_Dense_linear')(x)
    cnn = Model(inputs=model_input, outputs=cnn_output, name='CNN_selu')

    if show:
        print('CNN_selu summary:')
        cnn.summary()
        print()

    return cnn
