"""Training neural network."""
import importlib
import os
import re
from datetime import datetime
from functools import partial

import numpy as np
import tensorflow as tf

from erinn.tf_dataset import tf_read_dataset
from erinn.metrics import r_squared
from erinn.utils.io_utils import get_pkl_list
from erinn.utils.io_utils import read_config_file
from erinn.utils.io_utils import read_pkl
from erinn.utils.io_utils import write_pkl

FILEDIR = os.path.dirname(__file__)

# read config
config_file = os.path.join(FILEDIR, '..', '..', 'config', 'for_training.yml')
config = read_config_file(config_file)

# parse config and setting
custom_NN = config['custom_NN']
dataset_rootdir = config['dataset_rootdir']
training_dir = os.path.join(dataset_rootdir, 'training')
validation_dir = os.path.join(dataset_rootdir, 'validation')
training_resistance_dir = os.path.join(training_dir, 'resistance', config['resistance_dirname'])
training_resistivity_dir = os.path.join(training_dir, 'resistivity', config['resistivity_dirname'])
validation_resistance_dir = os.path.join(validation_dir, 'resistance', config['resistance_dirname'])
validation_resistivity_dir = os.path.join(validation_dir, 'resistivity', config['resistivity_dirname'])
simulator_pkl = os.path.join(dataset_rootdir, 'simulator.pkl')
simulator = read_pkl(simulator_pkl)  # for physical simulation
save_model_dir = config['save_model_dir']
os.makedirs(save_model_dir, exist_ok=True)
save_weights_dir = os.path.join(save_model_dir, 'weights')
tb_log_dir = os.path.join(save_model_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(tb_log_dir, exist_ok=True)
pre_trained_weight_h5 = config['pre_trained_weights']  # training from this weights.
trained_weight_h5 = os.path.join(save_weights_dir, 'trained_weight.h5')  # save trained weights to this file.
# accelerate
enable_XLA = config['enable_XLA']
enable_mixed_float16 = config['enable_mixed_float16']
# hyper parameters
gpus = config['num_gpu']
batch_size = config['batch_size']
epochs = config['num_epochs']
optimizer_cls = config['optimizer']['class_name']
optimizer_config = config['optimizer']['config']
preprocess = config['preprocess']
loss = config['loss']

# TODO: Organize reusable code snippets into functions


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


# Accelerate
# NOTE:
# 1. mixed_float16 have some problem in tf 2.0.0, please refer to first URL
# 2. If your model ends in softmax, make sure it is float32. 
#    And regardless of what your model ends in, make sure the output is float32.
# References:
# https://github.com/tensorflow/tensorflow/issues/33484
# https://www.tensorflow.org/guide/keras/mixed_precision
# https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/experimental
# https://github.com/sayakpaul/Mixed-Precision-Training-in-tf.keras-2.0/blob/master/With_Loss_Scale_Optimizer/Mixed_Precision_Data_Augmentation.ipynb
# https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s91029-automated-mixed-precision-tools-for-tensorflow-training-v2.pdf
if enable_XLA:
    # Enable XLA (Accelerated Linear Algebra)
    tf.config.optimizer.set_jit(True)
if enable_mixed_float16:
    # Enable AMP (Automatic Mixed Precision)
    # tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
    # Mixed precision
    # tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)


# TODO: custom callbacks. Maybe callbacks = [tensorboard, custom_callbacks]
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=tb_log_dir, histogram_freq=0,
    write_graph=True, write_images=False
)
not_nan = tf.keras.callbacks.TerminateOnNaN()
callbacks = [tensorboard, not_nan]


# Multi-GPU
# NOTE: compile keras model within MirroredStrategy.scope()
# References:
# https://www.tensorflow.org/guide/distributed_training
# https://www.tensorflow.org/tutorials/distribute/keras
# https://github.com/tensorflow/tensorflow/blob/919dfc3d066e72ee02baa11fbf7b035d9944daa9/tensorflow/python/distribute/mirrored_strategy.py#L339
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # load custom keras model
    # References:
    # https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3
    pattern = re.compile(r'\'([^\']+)\'')
    module_name, py_file = re.findall(pattern, custom_NN)
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    model = getattr(module, module_name)()

    # compiled if model didn't compile in user defined function
    if not model._is_compiled:
        optimizer = getattr(
            importlib.import_module('tensorflow.keras.optimizers'),
            optimizer_cls
        ).from_config(optimizer_config)
        model.compile(optimizer=optimizer, loss=loss, metrics=[r_squared])


# Use tf.data.Dataset
# References:
# https://www.tensorflow.org/guide/data
# https://stackoverflow.com/questions/60496435/how-to-convert-tensor-to-string
# https://github.com/tensorflow/tensorflow/issues/24520#issuecomment-532958834
training_resistance_pkl_list = get_pkl_list(training_resistance_dir)
training_resistivity_pkl_list = get_pkl_list(training_resistivity_dir)
validation_resistance_pkl_list = get_pkl_list(validation_resistance_dir)
validation_resistivity_pkl_list = get_pkl_list(validation_resistivity_dir)

list_dataset_training_inputs = tf.data.Dataset.from_tensor_slices(training_resistance_pkl_list)
list_dataset_training_targets = tf.data.Dataset.from_tensor_slices(training_resistivity_pkl_list)
list_dataset_training = tf.data.Dataset.zip((list_dataset_training_inputs, list_dataset_training_targets))

list_dataset_validation_inputs = tf.data.Dataset.from_tensor_slices(validation_resistance_pkl_list)
list_dataset_validation_targets = tf.data.Dataset.from_tensor_slices(validation_resistivity_pkl_list)
list_dataset_validation = tf.data.Dataset.zip((list_dataset_validation_inputs, list_dataset_validation_targets))

# use custom keras model to define shape of inputs and outputs
input_shape = model.input_shape[1:]
output_shape = model.output_shape[1:]
# read transmittor/receiver locations
Tx_locations = simulator.urf.abmn_locations[:, :4]
Rx_locations = simulator.urf.abmn_locations[:, 4:]
# read number of cell centor mesh in the x/z direction
nCx = simulator.mesh.nCx
nCy = simulator.mesh.nCy

read_dataset_info = {
    'preprocess': preprocess,
    'Tx_locations': Tx_locations,
    'Rx_locations': Rx_locations,
    'nCx': nCx, 'nCy': nCy,
    'input_shape': input_shape,
    'output_shape': output_shape
}

# use partial to assign read_dataset_info
par = partial(tf_read_dataset, read_dataset_info=read_dataset_info)
# randomly shuffle file_path => read data and preprocess => take mini-batch => prefetch
dataset_train = list_dataset_training.shuffle(buffer_size=len(training_resistance_pkl_list)).map(par, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(2)
# read data => take mini-batch => prefetch
dataset_valid = list_dataset_validation.map(par).batch(batch_size).prefetch(2)


# training
if os.path.isfile(pre_trained_weight_h5):
    model.load_weights(pre_trained_weight_h5)
original_weights = model.get_weights()
history = model.fit(dataset_train,
                    validation_data=dataset_valid, epochs=epochs,
                    callbacks=callbacks, workers=os.cpu_count())

# check weights
weights = model.get_weights()
if all([np.all(w == ow) for w, ow in zip(weights, original_weights)]):
    print('Weights in the template model have not changed')
else:
    print('Weights in the template model have changed')

# save weights
os.makedirs(save_weights_dir, exist_ok=True)
model.save_weights(trained_weight_h5)

# save simulator
save_simulator_pkl = os.path.join(save_model_dir, 'simulator.pkl')
simulator.config['training'] = config
write_pkl(simulator, save_simulator_pkl)