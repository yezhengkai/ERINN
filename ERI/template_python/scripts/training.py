import importlib
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import multi_gpu_model

from erinn.python.generator import DataGenerator
from erinn.python.metrics import r_squared
from erinn.python.utils.io_utils import get_pkl_list, read_config_file
from erinn.python.utils.os_utils import OSPlatform

# Allowing GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

# setting
config_file = os.path.join('..', 'config', 'config.yml')
config = read_config_file(config_file)
pkl_dir_train = config['train_dir']
pkl_dir_valid = config['valid_dir']
model_dir = config['model_dir']
os.makedirs(model_dir, exist_ok=True)
weights_dir = os.path.join(model_dir, 'weights')
tb_log_dir = os.path.join(model_dir, 'logs')
pre_trained_weight_h5 = config['pre_trained_weights']  # training from this weights.
trained_weight_h5 = os.path.join(weights_dir, 'trained_weight.h5')  # save trained weights to this file.
gpus = config['num_gpu']
batch_size = config['batch_size']
epochs = config['num_epochs']
optimizer = config['optimizer']
learning_rate = config['learning_rate']
optimizer = getattr(importlib.import_module('tensorflow.python.keras.optimizers'), optimizer)(lr=learning_rate)
preprocess_generator = config['preprocess_generator']
loss = config['loss']
use_multiprocessing = False
# when use_multiprocessing is True, training would be slow. Why?
# _os = OSPlatform()  # for fit_generator's keyword arguments `use_multiprocessing`
# if _os.is_WINDOWS:
#     use_multiprocessing = False
# else:
#     use_multiprocessing = True

# load custom keras model
# reference: https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3
pattern = re.compile(r'\'([^\']+)\'')
module_name, py_file = re.findall(pattern, config['custom_NN'])
loader = importlib.machinery.SourceFileLoader(module_name, py_file)
spec = importlib.util.spec_from_loader(module_name, loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)
model = getattr(module, module_name)()

# use custom keras model to define shape
pkl_list_train = get_pkl_list(pkl_dir_train)
pkl_list_valid = get_pkl_list(pkl_dir_valid)
input_shape = model.input_shape[1:]
output_shape = model.output_shape[1:]

# data generator
training_generator = DataGenerator(pkl_list_train, input_shape, output_shape,
                                   batch_size=batch_size, shuffle=True, **preprocess_generator)
validation_generator = DataGenerator(pkl_list_valid, input_shape, output_shape,
                                     batch_size=batch_size, **preprocess_generator)

# TODO: custom callbacks
tensorboard = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                                          write_graph=True, write_images=False)
callbacks = [tensorboard]


# training
if gpus <= 1:
    # 1 gpu
    if not model._is_compiled:
        model.compile(optimizer=optimizer, loss=loss, metrics=[r_squared])
    if os.path.isfile(pre_trained_weight_h5):
        model.load_weights(pre_trained_weight_h5)
    original_weights = keras.backend.batch_get_value(model.weights)
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs, use_multiprocessing=use_multiprocessing,
                                  callbacks=callbacks, workers=os.cpu_count())
    # check weights
    weights = keras.backend.batch_get_value(model.weights)
    if all([np.all(w == ow) for w, ow in zip(weights, original_weights)]):
        print('Weights in the template model have not changed')
    else:
        print('Weights in the template model have changed')
else:
    # 2 gpu or more
    if os.path.isfile(pre_trained_weight_h5):
        model.load_weights(pre_trained_weight_h5)
    original_weights = keras.backend.batch_get_value(model.weights)
    parallel_model = multi_gpu_model(model, gpus=gpus, cpu_relocation=False, cpu_merge=True)
    if not model._is_compiled:
        parallel_model.compile(optimizer=optimizer, loss=loss, metrics=[r_squared])
    else:
        parallel_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics,
                               loss_weights=model.load_weights, sample_weight_mode=model.sample_weight_mode,
                               weighted_metrics=model._compile_weighted_metrics)
    history = parallel_model.fit_generator(generator=training_generator,
                                           validation_data=validation_generator,
                                           epochs=epochs, use_multiprocessing=use_multiprocessing,
                                           callbacks=callbacks, workers=os.cpu_count())
    # check weights
    # references: https://github.com/keras-team/keras/issues/11313
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
model.save_weights(trained_weight_h5)


# pkl_dir_train = os.path.join('..', 'data', 'train')
# pkl_dir_valid = os.path.join('..', 'data', 'valid')
# model_dir = os.path.join('..', 'models')
# weights_dir = os.path.join(model_dir, 'weights')
# tb_log_dir = os.path.join(model_dir, 'logs')
# pre_trained_weight_h5 = ''  # training from this weights.
# trained_weight_h5 = os.path.join(weights_dir, 'trained_weight.h5')  # save trained weights to this file.
# gpus = 2
# epochs = 250

# create model
# def standard_unit(input_tensor, stage, num_filter, kernel_size=3, strides=(1, 1)):
#     dropout_rate = 0.2
#     act = LeakyReLU()
#
#     x = Conv2D(num_filter, kernel_size, activation=act, name='conv' + stage + '_1',
#                kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
#     x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
#     x = Conv2D(num_filter, (kernel_size, kernel_size), strides=strides,
#                activation=act, name='conv' + stage + '_2', padding='same',
#                kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
#     x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)
#     return x
#
#
# dropout_rate = 0.2
# num_filter = [16, 32, 64, 128, 256]
# up_strides = (2, 2)
# down_strides = (2, 2)
# crop = [((1, 1), (2, 2)), ((2, 2), (4, 4)), ((4, 4), (4, 4)), ((1, 1), (4, 4)), ((0, 0), (4, 4))]
# with tf.device('/cpu:0'):
#     inputs = Input(input_shape, name='main_input')
#     x = Dropout(dropout_rate, name='dp_0')(inputs)
#
#     conv1_1 = standard_unit(x, stage='11', num_filter=num_filter[0], strides=down_strides)
#     conv2_1 = standard_unit(conv1_1, stage='21', num_filter=num_filter[1], strides=down_strides)
#     conv3_1 = standard_unit(conv2_1, stage='31', num_filter=num_filter[2], strides=down_strides)
#     conv4_1 = standard_unit(conv3_1, stage='41', num_filter=num_filter[3], strides=down_strides)
#
#     conv5_1 = standard_unit(conv4_1, stage='51', num_filter=num_filter[4])
#     conv5_1 = Cropping2D(crop[0])(conv5_1)
#
#     up4_2 = Conv2DTranspose(num_filter[3], (3, 3), strides=up_strides, name='up42', padding='same')(conv5_1)
#     conv4_2 = standard_unit(up4_2, stage='42', num_filter=num_filter[3])
#     conv4_2 = Cropping2D(crop[1])(conv4_2)
#
#     up3_3 = Conv2DTranspose(num_filter[2], (3, 3), strides=up_strides, name='up33', padding='same')(conv4_2)
#     conv3_3 = standard_unit(up3_3, stage='33', num_filter=num_filter[2])
#     conv3_3 = Cropping2D(crop[2])(conv3_3)
#
#     up2_4 = Conv2DTranspose(num_filter[1], (3, 3), strides=(1, 1), name='up24', padding='same')(conv3_3)
#     conv2_4 = standard_unit(up2_4, stage='24', num_filter=num_filter[1])
#     conv2_4 = Cropping2D(crop[3])(conv2_4)
#
#     up1_5 = Conv2DTranspose(num_filter[0], (3, 3), strides=(1, 1), name='up15', padding='same')(conv2_4)
#     conv1_5 = standard_unit(up1_5, stage='15', num_filter=num_filter[0])
#     conv1_5 = Cropping2D(crop[4])(conv1_5)
#
#     x = standard_unit(conv1_5, stage='_out', num_filter=8)
#     outputs = Conv2D(1, (1, 1), name='main_output', kernel_initializer='he_normal',
#                      padding='same', kernel_regularizer=l2(3e-4))(x)
#     model = Model(inputs=inputs, outputs=outputs, name='FCN')
#     model.summary()


# # training
# if gpus <= 1:
#     # 1 gpu
#     model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=[r_squared])
#     if os.path.isfile(pre_trained_weight_h5):
#         model.load_weights(pre_trained_weight_h5)
#     original_weights = keras.backend.batch_get_value(model.weights)
#     history = model.fit_generator(generator=training_generator,
#                                   validation_data=validation_generator,
#                                   epochs=epochs, use_multiprocessing=False,
#                                   callbacks=callbacks, workers=4)
#     # check weights
#     weights = keras.backend.batch_get_value(model.weights)
#     if all([np.all(w == ow) for w, ow in zip(weights, original_weights)]):
#         print('Weights in the template model have not changed')
#     else:
#         print('Weights in the template model have changed')
# else:
#     # 2 gpus
#     if os.path.isfile(pre_trained_weight_h5):
#         model.load_weights(pre_trained_weight_h5)
#     original_weights = keras.backend.batch_get_value(model.weights)
#     parallel_model = multi_gpu_model(model, gpus=gpus, cpu_relocation=False, cpu_merge=True)
#     parallel_model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=[r_squared])
#     history = parallel_model.fit_generator(generator=training_generator,
#                                            validation_data=validation_generator,
#                                            epochs=epochs, use_multiprocessing=False,
#                                            callbacks=callbacks, workers=4)
#     # check weights
#     # https://github.com/keras-team/keras/issues/11313
#     weights = keras.backend.batch_get_value(model.weights)
#     parallel_weights = keras.backend.batch_get_value(parallel_model.weights)
#
#     if all([np.all(w == ow) for w, ow in zip(weights, original_weights)]):
#         print('Weights in the template model have not changed')
#     else:
#         print('Weights in the template model have changed')
#
#     if all([np.all(w == pw) for w, pw in zip(weights, parallel_weights)]):
#         print('Weights in the template and parallel model are equal')
#     else:
#         print('Weights in the template and parallel model are different')
#
#
# # save weights
# os.makedirs(weights_dir, exist_ok=True)
# model.save_weights(trained_weight_h5)
