from __future__ import division, absolute_import, print_function

import numpy as np
from tensorflow.python.keras.utils import Sequence


# Give up this function
# def data_generator(data_path, batch_size):
#     for path in data_path:
#         f = h5py.File(path)
#         inputs, targets = f['Inputs'], f['Targets']
#         # because this .h5 is created by matlab(F index)
#         length = inputs.shape[1]
#         idx = np.arange(length)
#         np.random.shuffle(idx)
#         batches = [idx[range(batch_size*i, min(length, batch_size*(i+1)))]
#                    for i in range(length//batch_size+1)]
#         for batch in batches:
#             x = np.array([])
#             y = np.array([])
#             if batch.size == 0:
#                 break
#             for j in range(len(batch)):
#                 x_tmp = inputs[:, batch[j]].T.astype('float32')
#                 y_tmp = np.log10(1/targets[:, batch[j]].T).astype('float32')
#                 x = np.vstack([x, x_tmp]) if x.size else x_tmp
#                 y = np.vstack([y, y_tmp]) if y.size else y_tmp
#             yield (x, y)
#     return


class DataGenerator(Sequence):
    """
    Custom Sequence object to train a model on out-of-memory data sets.

    References
    ----------
    https://github.com/jimmy60504/SeisNN/blob/master/seisnn/tensorflow/generator.py
    https://medium.com/tensorflow/training-and-serving-ml-models-with-tf-keras-fd975cc0fa27
    """

    def __init__(self, npz_list, input_shape, output_shape=None, batch_size=32, shuffle=True):
        self.npz_list = npz_list
        self.input_shape = input_shape
        if output_shape is None:
            self.output_shape = input_shape
        else:
            self.output_shape = output_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.npz_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.npz_list) / float(self.batch_size)))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        temp_npz_list = [self.npz_list[k] for k in indexes]
        delta_V, log_rho = self.__data_generation(temp_npz_list)

        return delta_V, log_rho

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_npz_list):
        delta_V = np.empty((len(temp_npz_list), *self.input_shape))
        log_rho = np.empty((len(temp_npz_list), *self.output_shape))
        for i, ID in enumerate(temp_npz_list):
            data = np.load(ID)
            delta_V[i, ] = data['Inputs'].reshape(self.input_shape)
            log_rho[i, ] = data['Targets'].reshape(self.output_shape)

        return delta_V, log_rho


class PredictGenerator(DataGenerator):

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_npz_list = [self.npz_list[k] for k in indexes]
        delta_V = self.__data_generation(temp_npz_list)

        return delta_V

    def __data_generation(self, temp_npz_list):
        delta_V = np.empty((len(temp_npz_list), *self.input_shape))
        for i, ID in enumerate(temp_npz_list):
            data = np.load(ID)
            delta_V[i, ] = data['Inputs'].reshape(self.input_shape)

        return delta_V
