import importlib
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import multi_gpu_model

from erinn.python.generator import DataGenerator
from erinn.python.metrics import r_squared
from erinn.python.utils.io_utils import get_pkl_list, read_config_file


# TODO: use tf.distribute.MirroredStrategy instead multi_gpu_model,
#       BUT tf.distribute.Strategy NOT support fit_generator in tf 2.0.0
# Deal with the issue: `fit_generator` is not supported for models compiled with tf.distribute.Strategy.
# https://github.com/tensorflow/tensorflow/issues/30321
tf.compat.v1.disable_eager_execution()


# setting
config_file = os.path.join('..', 'config', 'config.yml')
config = read_config_file(config_file)
glob_para_pkl = config['glob_para_pkl']
pkl_dir_train = config['train_dir']
pkl_dir_valid = config['valid_dir']
model_dir = config['model_dir']
os.makedirs(model_dir, exist_ok=True)
weights_dir = os.path.join(model_dir, 'weights')
tb_log_dir = Path(model_dir).joinpath('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(tb_log_dir, exist_ok=True)
pre_trained_weight_h5 = config['pre_trained_weights']  # training from this weights.
trained_weight_h5 = os.path.join(weights_dir, 'trained_weight.h5')  # save trained weights to this file.
gpus = config['num_gpu']
batch_size = config['batch_size']
epochs = config['num_epochs']
optimizer = config['optimizer']
learning_rate = config['learning_rate']
preprocess_generator = config['preprocess_generator']
loss = config['loss']
use_multiprocessing = False
# when use_multiprocessing is True, training would be slow. Why?
# _os = OSPlatform()  # for fit_generator's keyword arguments `use_multiprocessing`
# if _os.is_WINDOWS:
#     use_multiprocessing = False
# else:
#     use_multiprocessing = True


# Allowing GPU memory growth and set visible GPU
# References:
# https://www.tensorflow.org/guide/gpu
# https://qiita.com/studio_haneya/items/4dfaf2fb2ac44818e7e0
if tf.__version__.startswith('1.'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)
else:
    physical_gpus = tf.config.experimental.list_physical_devices('GPU')
    if physical_gpus:
        try:
            # Set visible gpus in TensorFlow
            tf.config.experimental.set_visible_devices(physical_gpus[0:gpus], 'GPU')
            visible_gpus = tf.config.experimental.get_visible_devices('GPU')
            # Currently, memory growth needs to be the same across GPUs
            for visible_gpu in visible_gpus:
                tf.config.experimental.set_memory_growth(visible_gpu, True)
                print('The memory growth of', visible_gpu, ':', tf.config.experimental.get_memory_growth(visible_gpu))
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(physical_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Visible devices/Memory growth/Virtual devices must be set before GPUs have been initialized
            print(e)
    else:
        print("Not enough GPU hardware devices available")


# TODO: mixed_float16 have some problem in tf 2.0.0, please refer to first URL
# Mixed precision
# References:
# https://github.com/tensorflow/tensorflow/issues/33484
# https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/experimental
# https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s91029-automated-mixed-precision-tools-for-tensorflow-training-v2.pdf
# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')


# load custom keras model
# References:
# https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3
pattern = re.compile(r'\'([^\']+)\'')
module_name, py_file = re.findall(pattern, config['custom_NN'])
loader = importlib.machinery.SourceFileLoader(module_name, py_file)
spec = importlib.util.spec_from_loader(module_name, loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)
model = getattr(module, module_name)()


# TODO: use tf.data.Dataset instead keras Sequence generator
# data generator
# use custom keras model to define shape of inputs and outputs
pkl_list_train = get_pkl_list(pkl_dir_train)
pkl_list_valid = get_pkl_list(pkl_dir_valid)
input_shape = model.input_shape[1:]
output_shape = model.output_shape[1:]

training_generator = DataGenerator(pkl_list_train, input_shape, output_shape,
                                   batch_size=batch_size, shuffle=True,
                                   glob_para_pkl=glob_para_pkl, **preprocess_generator)
validation_generator = DataGenerator(pkl_list_valid, input_shape, output_shape,
                                     batch_size=batch_size, glob_para_pkl=glob_para_pkl,
                                     **preprocess_generator)


# TODO: custom callbacks. Maybe callbacks = [tensorboard, custom_callbacks]
tensorboard = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                                          write_graph=True, write_images=False)
not_nan = keras.callbacks.TerminateOnNaN()
callbacks = [tensorboard, not_nan]


# TODO: use tf.distribute.MirroredStrategy instead multi_gpu_model,
#       BUT tf.distribute.Strategy NOT support fit_generator in tf 2.0.0
# compile keras model within MirroredStrategy.scope()
# References:
# https://www.tensorflow.org/guide/distributed_training
# https://www.tensorflow.org/tutorials/distribute/keras
# https://github.com/tensorflow/tensorflow/blob/919dfc3d066e72ee02baa11fbf7b035d9944daa9/tensorflow/python/distribute/mirrored_strategy.py#L339
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#
#     # load custom keras model
#     # References:
#     # https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3
#     pattern = re.compile(r'\'([^\']+)\'')
#     module_name, py_file = re.findall(pattern, config['custom_NN'])
#     loader = importlib.machinery.SourceFileLoader(module_name, py_file)
#     spec = importlib.util.spec_from_loader(module_name, loader)
#     module = importlib.util.module_from_spec(spec)
#     loader.exec_module(module)
#     model = getattr(module, module_name)()
#
#     # compiled if model didn't compile in user defined function
#     if not model._is_compiled:
#         optimizer = getattr(importlib.import_module('tensorflow.keras.optimizers'), optimizer)(lr=learning_rate)
#         model.compile(optimizer=optimizer, loss=loss, metrics=[r_squared])
#
#
# # data generator
# # use custom keras model to define shape of inputs and outputs
# pkl_list_train = get_pkl_list(pkl_dir_train)
# pkl_list_valid = get_pkl_list(pkl_dir_valid)
# input_shape = model.input_shape[1:]
# output_shape = model.output_shape[1:]
#
# training_generator = DataGenerator(pkl_list_train, input_shape, output_shape,
#                                    batch_size=batch_size, shuffle=True,
#                                    glob_para_pkl=glob_para_pkl, **preprocess_generator)
# validation_generator = DataGenerator(pkl_list_valid, input_shape, output_shape,
#                                      batch_size=batch_size, glob_para_pkl=glob_para_pkl,
#                                      **preprocess_generator)
#
#
# # training
# if os.path.isfile(pre_trained_weight_h5):
#     model.load_weights(pre_trained_weight_h5)
# original_weights = model.get_weights()
# history = model.fit_generator(generator=training_generator,
#                               validation_data=validation_generator,
#                               epochs=epochs, use_multiprocessing=use_multiprocessing,
#                               callbacks=callbacks, workers=os.cpu_count())
# # check weights
# weights = model.get_weights()
# if all([np.all(w == ow) for w, ow in zip(weights, original_weights)]):
#     print('Weights in the template model have not changed')
# else:
#     print('Weights in the template model have changed')


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
