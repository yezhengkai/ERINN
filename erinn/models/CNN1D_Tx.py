from __future__ import division, absolute_import, print_function

from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Reshape, Permute
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

# 第四種架構


def get_cnn1d_tx(output_size, img_height, img_width,
                 weight_decay=1e-4, show=True):

    # value with same Tx would use in same channel
    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = Reshape((img_height, img_width), name='Reshape_1')(model_input)
    x = Permute((2, 1), name='Permute_1')(x)
    x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same',
               activation='selu', kernel_initializer='lecun_normal',
               bias_initializer='zeros', kernel_regularizer=l2(weight_decay),
               name='Conv1D_selu_1')(x)
    x = Conv1D(filters=128, kernel_size=3, strides=1, padding='same',
               activation='selu', kernel_initializer='lecun_normal',
               bias_initializer='zeros', kernel_regularizer=l2(weight_decay),
               name='Conv1D_selu_2')(x)
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same',
               activation='selu', kernel_initializer='lecun_normal',
               bias_initializer='zeros', kernel_regularizer=l2(weight_decay),
               name='Conv1D_selu_3')(x)
    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same',
               activation='selu', kernel_initializer='lecun_normal',
               bias_initializer='zeros', kernel_regularizer=l2(weight_decay),
               name='Conv1D_selu_4')(x)
    x = Conv1D(filters=16, kernel_size=3, strides=1, padding='same',
               activation='selu', kernel_initializer='lecun_normal',
               bias_initializer='zeros', kernel_regularizer=l2(weight_decay),
               name='Conv1D_selu_5')(x)
    x = Flatten(name='Flatten_1')(x)
    x = Dense(512, activation='selu', kernel_initializer='lecun_normal',
              bias_initializer='zeros', kernel_regularizer=l2(weight_decay),
              name='Dense_selu_1')(x)
    x = Dense(512, activation='selu', kernel_initializer='lecun_normal',
              bias_initializer='zeros', kernel_regularizer=l2(weight_decay),
              name='Dense_selu_2')(x)
    cnn1d_output = Dense(output_size, activation='linear',
                         name='Output_Dense_linear')(x)
    cnn1d = Model(inputs=model_input, outputs=cnn1d_output,
                  name='CNN1D_Tx')

    if show:
        print('CNN1D_Tx summary:')
        cnn1d.summary()
        print()

    return cnn1d


def get_cnn1d_tx_relu(output_size, img_height, img_width, show=True):

    # value with same Tx would use in same channel
    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = Reshape((img_height, img_width), name='Reshape_1')(model_input)
    x = Permute((2, 1), name='Permute_1')(x)
    x = Conv1D(filters=256, kernel_size=11, strides=1,
               padding='same', activation='relu',
               name='Conv1D_relu_1')(x)
    x = Conv1D(filters=128, kernel_size=5, strides=1,
               padding='same', activation='relu',
               name='Conv1D_relu_2')(x)
    x = Conv1D(filters=64, kernel_size=3, strides=1,
               padding='same', activation='relu',
               name='Conv1D_relu_3')(x)
    x = Conv1D(filters=32, kernel_size=3, strides=1,
               padding='same', activation='relu',
               name='Conv1D_relu_4')(x)
    x = Conv1D(filters=16, kernel_size=3, strides=1,
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
                  name='CNN1D_Tx_relu')

    if show:
        print('CNN1D Tx summary:')
        cnn1d.summary()
        print()

    return cnn1d


def get_cnn1d_tx_selu(output_size, img_height, img_width, show=True):
    model_input = Input(shape=(img_height * img_width,), name='Main_input')
    x = Reshape((img_height, img_width), name='Reshape_1')(model_input)
    x = Permute((2, 1), name='Permute_1')(x)
    x = Conv1D(filters=256, kernel_size=48, strides=1,
               padding='same', activation='selu',
               name='Conv1D_selu_1')(x)
    x = Conv1D(filters=128, kernel_size=48, strides=1,
               padding='same', activation='selu',
               name='Conv1D_selu_2')(x)
    x = Conv1D(filters=64, kernel_size=48, strides=1,
               padding='same', activation='selu',
               name='Conv1D_selu_3')(x)
    x = Conv1D(filters=32, kernel_size=48, strides=1,
               padding='same', activation='selu',
               name='Conv1D_selu_4')(x)
    x = Conv1D(filters=16, kernel_size=48, strides=1,
               padding='same', activation='selu',
               name='Conv1D_selu_5')(x)
    x = Conv1D(filters=8, kernel_size=48, strides=1,
               padding='same', activation='selu',
               name='Conv1D_selu_6')(x)
    x = Flatten(name='Flatten_1')(x)
    x = Dense(512, activation='selu', name='Dense_selu_1')(x)
    x = Dense(512, activation='selu', name='Dense_selu_2')(x)
    x = Dense(512, activation='selu', name='Dense_selu_3')(x)
    cnn1d_output = Dense(output_size, activation='linear',
                         name='Output_Dense_linear')(x)
    cnn1d = Model(inputs=model_input, outputs=cnn1d_output,
                  name='CNN1D_Tx_selu')

    if show:
        print('CNN1D Tx summary:')
        cnn1d.summary()
        print()

    return cnn1d
