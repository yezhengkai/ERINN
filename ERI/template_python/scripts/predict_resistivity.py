import os
import re
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Cropping2D
from tensorflow.python.keras.layers import Input, Dropout
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2

from erinn.python.generator import PredictGenerator
from erinn.python.metrics import r_squared
from erinn.python.utils.io_utils import get_pkl_list, read_pkl, write_pkl

# Allowing GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

# setting
# config_file = os.path.join('..', 'config', 'config.yml')
pkl_dir_test = os.path.join('..', 'data', 'raw_data', 'test')
model_dir = os.path.join('..', 'models', 'add_noise_log_transform')
weights_dir = os.path.join(model_dir, 'weights')
predictions_dir = os.path.join(model_dir, 'predictions', 'raw_data')

pkl_list_test = get_pkl_list(pkl_dir_test)
input_shape = (210, 780, 1)
output_shape = (30, 140, 1)

preprocess_generator = {'add_noise':
                            {'perform': False,
                             'kwargs':
                                 {'ratio': 0.1}
                             },
                        'log_transform':
                            {'perform': True,
                             'kwargs':
                                 {'inverse': False,
                                  'inplace': True}
                             }
                        }
# data generator
testing_generator = PredictGenerator(pkl_list_test, input_shape, output_shape,
                                     batch_size=64, **preprocess_generator)


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


dropout_rate = 0.2
num_filter = [16, 32, 64, 128, 256]
up_strides = (2, 2)
down_strides = (2, 2)
crop = [((1, 1), (2, 2)), ((2, 2), (4, 4)), ((4, 4), (4, 4)), ((1, 1), (4, 4)), ((0, 0), (4, 4))]
with tf.device('/gpu:0'):
    inputs = Input(input_shape, name='main_input')
    x = Dropout(dropout_rate, name='dp_0')(inputs)

    conv1_1 = standard_unit(x, stage='11', num_filter=num_filter[0], strides=down_strides)
    conv2_1 = standard_unit(conv1_1, stage='21', num_filter=num_filter[1], strides=down_strides)
    conv3_1 = standard_unit(conv2_1, stage='31', num_filter=num_filter[2], strides=down_strides)
    conv4_1 = standard_unit(conv3_1, stage='41', num_filter=num_filter[3], strides=down_strides)

    conv5_1 = standard_unit(conv4_1, stage='51', num_filter=num_filter[4])
    conv5_1 = Cropping2D(crop[0])(conv5_1)

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

model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=[r_squared])
model.load_weights(os.path.join(weights_dir, 'trained_weight.h5'))

print('\nPredict.')
predict = model.predict_generator(testing_generator, workers=os.cpu_count(), verbose=True)

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
