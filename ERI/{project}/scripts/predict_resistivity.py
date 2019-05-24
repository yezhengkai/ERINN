import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Reshape
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2

# from erinn.python.generator import PredictGenerator
from erinn.python.metrics import r_squared
from erinn.python.utils.io_utils import get_npz_list, save_synth_data, save_daily_data


# Allowing GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)


# setting
glob_para_h5 = os.path.join('..', 'config', 'glob_para.h5')
npz_dir = os.path.join('..', 'data', 'processed_data', 'testing')
weights_dir = os.path.join('..', 'models', 'weights')
dest1_h5 = os.path.join('..', 'models', 'predictions', 'testing.h5')
dest2_h5 = os.path.join('..', 'models', 'predictions', 'daily.h5')
urf_dir = os.path.join('..', 'data', 'daily_data')

npz_list = get_npz_list(npz_dir)
input_shape = np.load(npz_list[0])['Inputs'].shape  # use tuple
output_shape = (np.load(npz_list[0])['Targets'].size, )  # use tuple


# create model (Model modified from original Alexnet)
def standard_unit(input_tensor, stage, num_filter, kernel_size=3, strides=(1, 1)):
    dropout_rate = 0.2
    act = LeakyReLU()

    x = Conv2D(num_filter, kernel_size, activation=act, name='conv' + stage + '_1',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
    x = Conv2D(num_filter, (kernel_size, kernel_size), strides=strides,
               activation=act, name='conv' + stage + '_2', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)
    return x


num_filter = [12, 32, 48, 48, 32]
down_strides = (2, 2)
with tf.device('/cpu:0'):
    inputs = Input(input_shape, name='main_input')
    reduced = Dropout(0.2, name='dp_0')(inputs)

    conv1_1 = standard_unit(reduced, stage='11', num_filter=num_filter[0], strides=down_strides)
    conv2_1 = standard_unit(conv1_1, stage='21', num_filter=num_filter[1], strides=down_strides)
    conv3_1 = standard_unit(conv2_1, stage='31', num_filter=num_filter[2], strides=down_strides)
    conv4_1 = standard_unit(conv3_1, stage='41', num_filter=num_filter[3], strides=down_strides)
    conv5_1 = standard_unit(conv4_1, stage='51', num_filter=num_filter[4])

    flat = Flatten(name='flatten_1')(conv5_1)
    dense_1 = Dense(512, activation=LeakyReLU(), name='dense_1', kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4))(flat)
    dense_2 = Dense(512, activation=LeakyReLU(), name='dense_2', kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4))(dense_1)
    outputs = Dense(output_shape[0], name='main_output', kernel_initializer='he_normal', kernel_regularizer=l2(3e-4))(dense_2)
    model = Model(inputs=inputs, outputs=outputs, name='CNN')
    model.summary()


model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=[r_squared])
model.load_weights(os.path.join(weights_dir, 'trained_weight.h5'))

# predict and save
save_synth_data(model, glob_para_h5, npz_list, dest_h5=dest1_h5)
save_daily_data(model, glob_para_h5, urf_dir, dest_h5=dest2_h5)
