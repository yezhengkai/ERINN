import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils import multi_gpu_model

from erinn.python.generator import DataGenerator
from erinn.python.metrics import r_squared
from erinn.python.utils.io_utils import get_npz_list


# Allowing GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)


# setting
npz_dir = os.path.join('..', 'data', 'processed_data', 'training')
weights_dir = os.path.join('..', 'models', 'weights')
tb_log_dir = os.path.join('..', 'models', 'logs')
gpus = 2
epochs = 250

npz_list = get_npz_list(npz_dir)
input_shape = np.load(npz_list[0])['Inputs'].shape  # use tuple
output_shape = (np.load(npz_list[0])['Targets'].size, )  # use tuple


# data generator
split_point = -1000
training_generator = DataGenerator(npz_list[:split_point], input_shape, output_shape,
                                   batch_size=64, shuffle=True)
validation_generator = DataGenerator(npz_list[split_point:], input_shape, output_shape,
                                     batch_size=32)


# callbacks
tensorboard = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                                          write_graph=True, write_images=False)
callbacks = [tensorboard]


# create model
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


# training
if gpus <= 1:
    # 1 gpu
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=[r_squared])
    model.load_weights(os.path.join(weights_dir, 'pretrained_weight.h5'))
    original_weights = keras.backend.batch_get_value(model.weights)
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs, use_multiprocessing=False,
                                  callbacks=callbacks, workers=4)
    # check weights
    weights = keras.backend.batch_get_value(model.weights)
    if all([np.all(w == ow) for w, ow in zip(weights, original_weights)]):
        print('Weights in the template model have not changed')
    else:
        print('Weights in the template model have changed')
else:
    # 2 gpus
    model.load_weights(os.path.join(weights_dir, 'pretrained_weight.h5'))
    original_weights = keras.backend.batch_get_value(model.weights)
    parallel_model = multi_gpu_model(model, gpus=gpus, cpu_relocation=False, cpu_merge=True)
    parallel_model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=[r_squared])
    history = parallel_model.fit_generator(generator=training_generator,
                                           validation_data=validation_generator,
                                           epochs=epochs, use_multiprocessing=False,
                                           callbacks=callbacks, workers=4)
    # check weights
    # https://github.com/keras-team/keras/issues/11313
    weights = keras.backend.batch_get_value(model.weights)
    parallel_weights = keras.backend.batch_get_value(parallel_model.weights)

    if all([np.all(w == ow) for w, ow in zip(weights, original_weights)]):
        print('Weights in the template model have not changed')
    else:
        print('Weights in the template model have changed')

    if all([np.all(w == pw) for w, pw in zip(weights, parallel_weights)]):
        print('Weights in the template and parallel model are equal')
    else:
        print('Weights in the template and parallel model are different')


# save weights
os.makedirs(weights_dir, exist_ok=True)
model.save_weights(os.path.join(weights_dir, 'trained_weight.h5'))
