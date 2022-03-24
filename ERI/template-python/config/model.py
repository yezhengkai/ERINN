import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# create model
def standard_block(input_tensor, stage, num_filter, kernel_size=3, strides=(1, 1)):
    dropout_rate = 0.2
    # act = LeakyReLU()
    act = "relu"

    x = Conv2D(num_filter, kernel_size,
               activation=act, name='conv' + stage + '_1', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
    x = Conv2D(num_filter, kernel_size, strides=strides,
               activation=act, name='conv' + stage + '_2', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)
    return x


# This model is for templates, so performance may be poor
# The shape of resistance (shape of NN input data): (2770, 227, 1)
# The shape of resistivity (shape of NN target data): (36, 150, 1)
def my_model():
    # setting
    input_shape = (2770, 227, 1)
    dropout_rate = 0.2
    num_filter = [16, 32, 64, 128, 256] 
    up_strides = (1, 2)
    down_strides = (3, 2)
    crop = [((2, 2), (0, 0)), ((3, 3), (3, 3)), ((3, 3), (3, 3)), ((3, 3), (3, 3)), ((4, 4), (3, 3))]

    # construct keras model
    # with tf.device('/cpu:0'):
    inputs = Input(input_shape, name='main_input')
    x = Dropout(dropout_rate, name='dp_0')(inputs)

    conv1_1 = standard_block(x, stage='11', num_filter=num_filter[0],
                                kernel_size=(5, 3), strides=down_strides)
    conv2_1 = standard_block(conv1_1, stage='21', num_filter=num_filter[1],
                                kernel_size=(5, 3),  strides=down_strides)
    conv3_1 = standard_block(conv2_1, stage='31', num_filter=num_filter[2],
                                kernel_size=(5, 3),  strides=down_strides)
    conv4_1 = standard_block(conv3_1, stage='41', num_filter=num_filter[3],
                                kernel_size=(5, 3),  strides=down_strides)

    conv5_1 = standard_block(conv4_1, stage='51', num_filter=num_filter[4])
    conv5_1 = Cropping2D(crop[0])(conv5_1)

    up4_2 = Conv2DTranspose(num_filter[3], (3, 3), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = standard_block(up4_2, stage='42', num_filter=num_filter[3])
    conv4_2 = Cropping2D(crop[1])(conv4_2)

    up3_3 = Conv2DTranspose(num_filter[2], (3, 3), strides=up_strides, name='up33', padding='same')(conv4_2)
    conv3_3 = standard_block(up3_3, stage='33', num_filter=num_filter[2])
    conv3_3 = Cropping2D(crop[2])(conv3_3)

    up2_4 = Conv2DTranspose(num_filter[1], (3, 3), strides=up_strides, name='up24', padding='same')(conv3_3)
    conv2_4 = standard_block(up2_4, stage='24', num_filter=num_filter[1])
    conv2_4 = Cropping2D(crop[3])(conv2_4)

    up1_5 = Conv2DTranspose(num_filter[0], (3, 3), strides=up_strides, name='up15', padding='same')(conv2_4)
    conv1_5 = standard_block(up1_5, stage='15', num_filter=num_filter[0])
    conv1_5 = Cropping2D(crop[4])(conv1_5)

    x = standard_block(conv1_5, stage='_out', num_filter=8)
    outputs = Conv2D(1, (1, 1), name='main_output', kernel_initializer='he_normal',
                        padding='same', kernel_regularizer=l2(3e-4))(x)
    
    outputs = Activation('linear', dtype='float32')(outputs)  # for mixed_float16

    model = Model(inputs=inputs, outputs=outputs, name='FCN')
    model.summary()
    return model
