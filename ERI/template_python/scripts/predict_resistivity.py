import os
import re
from tqdm import tqdm

import tensorflow as tf

from erinn.python.generator import PredictGenerator
from erinn.python.utils.io_utils import read_config_file, get_pkl_list, read_pkl, write_pkl


# setting
config_file = os.path.join('..', 'config', 'config.yml')
config = read_config_file(config_file)
glob_para_pkl = config['glob_para_pkl']
pkl_dir_test = os.path.join('..', 'data', 'raw_data', 'test')
model_dir = os.path.join('..', 'models', 'txrx')
weights_dir = os.path.join(model_dir, 'weights')
predictions_dir = os.path.join(model_dir, 'predictions')
preprocess_generator = config['preprocess_generator']
gpus = 1


# Allowing GPU memory growth
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


# load custom keras model and weights
pattern = re.compile(r'\'([^\']+)\'')
module_name, py_file = re.findall(pattern, config['custom_NN'])
loader = importlib.machinery.SourceFileLoader(module_name, py_file)
spec = importlib.util.spec_from_loader(module_name, loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)
model = getattr(module, module_name)()
model.load_weights(os.path.join(weights_dir, 'trained_weight.h5'))

# data generator
pkl_list_test = get_pkl_list(pkl_dir_test)
input_shape = model.input_shape[1:]
output_shape = model.output_shape[1:]
testing_generator = PredictGenerator(pkl_list_test, input_shape, output_shape,
                                     batch_size=64, glob_para_pkl=glob_para_pkl,
                                     **preprocess_generator)

# Prediction
print('\nPredict.')
predict = model.predict_generator(testing_generator, workers=os.cpu_count(), verbose=True)

# Save
os.makedirs(predictions_dir, exist_ok=True)
with tqdm(total=len(pkl_list_test), desc='write pkl') as pbar:
    for i, pred in enumerate(predict):
        data = read_pkl(pkl_list_test[i])
        data['synth_V'] = data.pop('inputs').reshape(input_shape[0:2])
        data['synth_log_rho'] = data.pop('targets').reshape(output_shape[0:2])
        data['pred_log_rho'] = pred.reshape(output_shape[0:2])

        suffix = re.findall(r'\d+.pkl', pkl_list_test[i])[0]
        write_pkl(data, os.path.join(predictions_dir, f'result_{suffix}'))
        pbar.update()
