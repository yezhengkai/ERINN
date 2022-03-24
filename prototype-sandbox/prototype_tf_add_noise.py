"""Test behavior of adding random noise to tf.data.Dataset data pipeline"""
# References:
# https://github.com/tensorflow/tensorflow/issues/35682

import numpy as np
from functools import partial
import tensorflow as tf
from erinn.preprocessing import add_noise
# import tensorflow.compat.v2 as tf

# tf.enable_v2_behavior()

def map_fn(seed):
  return seed + tf.random.stateless_uniform([], [seed, seed])

# ds = tf.data.Dataset.range(5)
ds = tf.data.Dataset.from_tensor_slices([0., 1., 2., 3., 4.])
# ds = ds.map(map_fn).shuffle(5, seed=42, reshuffle_each_iteration=True).batch(5)
ds = ds.map(map_fn, num_parallel_calls=4).shuffle(5, seed=42, reshuffle_each_iteration=True).batch(5)

for elem in ds:
  print(elem.numpy())
for elem in ds:
  print(elem.numpy())

# ===================================================================================
seed = tf.Variable(0, dtype=tf.int64)

def get_seed(_):
  seed.assign_add(1)
  return seed

seeds = tf.data.Dataset.range(1).map(get_seed)
# for elem in seeds:
#   print(elem.numpy())
seeds = seeds.flat_map(lambda seed: tf.data.experimental.RandomDataset(seed=seed))
# for elem in seeds:
#   print(elem.numpy())

# ds = tf.data.Dataset.zip((seeds.batch(2), tf.data.Dataset.range(5)))
ds = tf.data.Dataset.zip((seeds.batch(2), tf.data.Dataset.from_tensor_slices([0., 1., 2., 3., 4.])))
ds = ds.map(lambda seed, _: tf.random.stateless_uniform([], seed=seed), num_parallel_calls=4).batch(5)

print("epoch 1:")
for elem in ds:
  print(elem.numpy())
print("epoch 2:")
for elem in ds:
  print(elem.numpy())

# ===================================================================================
# more similar to data augmentation
from functools import partial
import numpy as np
import tensorflow as tf


def add_noise(array, scale=0.05, noise_type='normal', seed=None, inplace=True):

    if type(noise_type) == bytes:
        noise_type = noise_type.decode('utf-8')
    array = np.asarray(array)
    raw_shape = array.shape

    if not inplace:
        new_array = array.copy()
    else:
        new_array = array

    rng = np.random.default_rng(seed)
    if noise_type == 'normal':
        new_array += scale * abs(new_array) * rng.normal(0.0, 1.0, size=raw_shape)
    elif noise_type == 'uniform':
        new_array += scale * abs(new_array) * rng.uniform(0.0, 1.0, size=raw_shape)
    else:
        raise(NotImplementedError('noise_type is not supported.'))

    if not inplace:
        return new_array.reshape(raw_shape)


def tf_add_noise(array, seed):
    # wrapping par function with py_function
    [new_array] = tf.numpy_function(
        add_noise, [array, 0.05, 'normal', seed, False], [tf.float32]
    )

    return new_array

seed = tf.Variable(0, dtype=tf.int64)

def get_seed(_):
    seed.assign_add(1)
    return seed

seeds = tf.data.Dataset.range(1).map(get_seed)
seeds = seeds.flat_map(lambda seed: tf.data.experimental.RandomDataset(seed=seed))

ds = tf.data.Dataset.zip((seeds.batch(1), tf.data.Dataset.from_tensor_slices([[[0., 1.], [2., 3.]]])))
ds = ds.map(lambda seed, array: tf_add_noise(array, seed),
    num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(5)

for elem in ds:
    print(elem.numpy())
for elem in ds:
    print(elem.numpy())
