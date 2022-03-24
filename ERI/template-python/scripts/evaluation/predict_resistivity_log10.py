"""Use trained neural network to predict log10 scale resistivity."""
import importlib
import os
import re
from datetime import datetime
from functools import partial
from unicodedata import name

import tensorflow as tf
from tqdm import tqdm

from erinn.tf_dataset import tf_read_dataset
from erinn.utils.io_utils import get_pkl_list
from erinn.utils.io_utils import read_config_file
from erinn.utils.io_utils import read_pkl
from erinn.utils.io_utils import write_pkl

# TODO: Organize reusable code snippets into functions

FILEDIR = os.path.dirname(__file__)

# read config
config_file = os.path.join(FILEDIR, '..', '..', 'config', 'for_predict_resistivity.yml')
config = read_config_file(config_file)

# parse config and setting
custom_NN = config['custom_NN']
dataset_rootdir = os.path.join(FILEDIR, config['dataset_rootdir'])
testing_dir = os.path.join(dataset_rootdir, 'testing')
testing_resistance_dir = os.path.join(testing_dir, 'resistance', config['resistance_dirname'])
testing_resistivity_dir = os.path.join(testing_dir, 'resistivity', config['resistivity_dirname'])
raw_resistance_dir = os.path.join(testing_dir, 'resistance', "raw")

model_dir = config['model_dir']
simulator_pkl = os.path.join(FILEDIR, model_dir, 'simulator.pkl')
simulator = read_pkl(simulator_pkl)
weights_dir = os.path.join(FILEDIR, model_dir, 'weights')
trained_weights = os.path.join(FILEDIR, weights_dir, 'trained_weight.h5')
save_predictions_dir = os.path.join(FILEDIR, model_dir, 'predictions')
preprocess = config['preprocess']
gpus = config['num_gpu']


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
    py_file = os.path.join(FILEDIR, py_file)
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    model = getattr(module, module_name)()
    # load weights
    model.load_weights(trained_weights)


# Use tf.data.Dataset
# References:
# https://www.tensorflow.org/guide/data
# https://stackoverflow.com/questions/60496435/how-to-convert-tensor-to-string
# https://github.com/tensorflow/tensorflow/issues/24520#issuecomment-532958834
testing_resistance_pkl_list = get_pkl_list(testing_resistance_dir)
testing_resistivity_pkl_list = get_pkl_list(testing_resistivity_dir)
raw_resistance_pkl_list = get_pkl_list(raw_resistance_dir)

list_dataset_testing_inputs = tf.data.Dataset.from_tensor_slices(testing_resistance_pkl_list)
list_dataset_testing_targets = tf.data.Dataset.from_tensor_slices(testing_resistivity_pkl_list)
list_dataset_testing = tf.data.Dataset.zip((list_dataset_testing_inputs, list_dataset_testing_targets))


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
# read data => take mini-batch => prefetch
dataset_test = list_dataset_testing.map(par).batch(1).prefetch(8)

# Prediction
print('\nPredict.')
predict = model.predict(dataset_test, verbose=True)

# Save
os.makedirs(save_predictions_dir, exist_ok=True)
for i, dataset_testing_targets in tqdm(enumerate(list_dataset_testing_targets.as_numpy_iterator()), desc="write pkl"):
    raw_resistance = read_pkl(raw_resistance_pkl_list[i])
    resistivity_log10 = read_pkl(dataset_testing_targets)
    data ={
        "synthetic_resistance": raw_resistance,
        "synthetic_resistivity_log10": resistivity_log10.reshape(output_shape[0:2]),
        "predicted_resistivity_log10": predict[i].reshape(output_shape[0:2])
    }    
    filename = re.findall(r'\d+.pkl', testing_resistance_pkl_list[i])[0]
    write_pkl(data, os.path.join(save_predictions_dir, filename))

# save simulator
simulator.config['testing'] = config
write_pkl(simulator, simulator_pkl)

# TODO: Deprecated! remove it
# with tqdm(total=len(pkl_list_test), desc='write pkl') as pbar:
#     for i, pred in enumerate(predict):
#         data = read_pkl(pkl_list_test[i])  # type(data) is dict
#         data['synthetic_resistance'] = (
#             data.pop('resistance').reshape(input_shape[0:2])
#         )
#         data['synthetic_resistivity_log10'] = (
#             data.pop('resistivity_log10').reshape(output_shape[0:2])
#         )
#         data['predicted_resistivity_log10'] = pred.reshape(output_shape[0:2])

#         suffix = re.findall(r'\d+.pkl', pkl_list_test[i])[0]
#         write_pkl(data, os.path.join(save_predictions_dir, f'result_{suffix}'))
#         pbar.update()
