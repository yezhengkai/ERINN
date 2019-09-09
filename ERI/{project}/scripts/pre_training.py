import os

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Conv2DTranspose, Cropping2D
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
gpus = 0
epochs = 1

npz_list = get_npz_list(npz_dir)
npz_list = npz_list[0:int(len(npz_list) * 0.4)]  # pre-training with small samples
input_shape = np.load(npz_list[0])['Inputs'].shape  # use tuple
output_shape = (np.load(npz_list[0])['Targets'].size, )  # use tuple


# data generator
split_point = -10
training_generator = DataGenerator(npz_list[:split_point], input_shape, output_shape,
                                   batch_size=64, shuffle=True)
validation_generator = DataGenerator(npz_list[split_point:], input_shape, output_shape,
                                     batch_size=32)


# callbacks
tensorboard = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                                          write_graph=True, write_images=False)
callbacks = [tensorboard]


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


dropout_rate = 0.2
num_filter = [16, 32, 64, 128, 256]
up_strides = (2, 2)
down_strides = (2, 2)
crop = [((1, 1), (2, 2)), ((2, 2), (4, 4)), ((4, 4), (4, 4)), ((1, 1), (4, 4)), ((0, 0), (4, 4))]
with tf.device('/cpu:0'):
    inputs = Input(input_shape, name='main_input')
    x = Dropout(dropout_rate, name='dp_0')(inputs)

    conv1_1 = standard_unit(x, stage='11', num_filter=num_filter[0], strides=down_strides)
    conv2_1 = standard_unit(conv1_1, stage='21', num_filter=num_filter[1], strides=down_strides)
    conv3_1 = standard_unit(conv2_1, stage='31', num_filter=num_filter[2], strides=down_strides)
    conv4_1 = standard_unit(conv3_1, stage='41', num_filter=num_filter[3], strides=down_strides)

    conv5_1 = standard_unit(conv4_1, stage='51', num_filter=num_filter[4]) #此行以上是縮減尺寸
    conv5_1 = Cropping2D(crop[0])(conv5_1) #此行之後是放大尺寸再裁切

    up4_2 = Conv2DTranspose(num_filter[3], (3, 3), strides=up_strides, name='up42', padding='same')(conv5_1)
    conv4_2 = standard_unit(up4_2, stage='42', num_filter=num_filter[3])
    conv4_2 = Cropping2D(crop[1])(conv4_2)

    up3_3 = Conv2DTranspose(num_filter[2], (3, 3), strides=up_strides, name='up33', padding='same')(conv4_2)
    conv3_3 = standard_unit(up3_3, stage='33', num_filter=num_filter[2])
    conv3_3 = Cropping2D(crop[2])(conv3_3)

    up2_4 = Conv2DTranspose(num_filter[1], (3, 3), strides=(1, 1), name='up24', padding='same')(conv3_3)
    conv2_4 = standard_unit(up2_4, stage='24', num_filter=num_filter[1])
    conv2_4 = Cropping2D(crop[3])(conv2_4)

    up1_5 = Conv2DTranspose(num_filter[0], (3, 3), strides=(1, 1), name='up15', padding='same')(conv2_4)
    conv1_5 = standard_unit(up1_5, stage='15', num_filter=num_filter[0])
    conv1_5 = Cropping2D(crop[4])(conv1_5)

    x = standard_unit(conv1_5, stage='_out', num_filter=8)
    outputs = Conv2D(1, (1, 1), name='main_output', kernel_initializer='he_normal',
                     padding='same', kernel_regularizer=l2(3e-4))(x)
    model = Model(inputs=inputs, outputs=outputs, name='FCN')
    model.summary()


# training
# https://github.com/keras-team/keras/issues/10842
# https://stackoverflow.com/questions/52932406/is-the-class-generator-inheriting-sequence-thread-safe-in-keras-tensorflow
# use_multiprocessing not working on Windows
if gpus <= 1:
    # 1 gpu
    model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error', metrics=[r_squared])
    original_weights = keras.backend.batch_get_value(model.weights)
    model.fit_generator(generator=training_generator, validation_data=validation_generator,
                        epochs=epochs, use_multiprocessing=False, callbacks=callbacks,
                        workers=4)
    # check weights
    weights = keras.backend.batch_get_value(model.weights)
    if all([np.all(w == ow) for w, ow in zip(weights, original_weights)]):
        print('Weights in the template model have not changed')
    else:
        print('Weights in the template model have changed')
else:
    # 2 gpus
    original_weights = keras.backend.batch_get_value(model.weights)
    parallel_model = multi_gpu_model(model, gpus=gpus, cpu_relocation=False, cpu_merge=True)
    parallel_model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error', metrics=[r_squared])
    history = parallel_model.fit_generator(generator=training_generator,
                                           validation_data=validation_generator,
                                           epochs=epochs, use_multiprocessing=False,
                                           callbacks=[tensorboard], workers=4)
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
model.save_weights(os.path.join(weights_dir, 'pretrained_weight.h5'))
