"""
Data generator for keras model.


References
----------
.. [1] https://github.com/jimmy60504/SeisNN/blob/master/seisnn/tensorflow/generator.py
.. [2] https://medium.com/tensorflow/training-and-serving-ml-models-with-tf-keras-fd975cc0fa27
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import generator_stop
from __future__ import print_function

import warnings
from abc import abstractmethod

import numpy as np
from tensorflow.keras.utils import Sequence

from erinn.preprocessing import add_noise
from erinn.preprocessing import log_transform
from erinn.preprocessing import to_midpoint
from erinn.preprocessing import to_txrx
from erinn.utils.io_utils import read_pkl


class BaseGenerator(Sequence):
    """
    Parent class of DataGenerator and PredictGenerator.
    """
    def __init__(self, file_list, input_shape,
                 output_shape=None, batch_size=32, shuffle=False,
                 Tx_locations=None, Rx_locations=None, **preprocess):
        """
        Parameters
        ----------
        file_list : list
            A list of files containing input and target data.
        input_shape : tuple
            The shape of the input layer in the neural network.
        output_shape : tuple, optional
            The shape of the output layer in the neural network.
        batch_size : int, optional
            Size for mini-batch gradient descent.
        shuffle : bool, optional
            Whether to shuffle on the epoch end.
        preprocess : dict, optional

        """
        self.file_list = file_list
        self.input_shape = input_shape
        if output_shape is None:
            self.output_shape = input_shape
        else:
            self.output_shape = output_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_list))
        self.preprocess = preprocess
        if self.preprocess['to_midpoint']['perform']\
                or self.preprocess['to_txrx']['perform']:
            self.Tx_locations = Tx_locations
            self.Rx_locations = Rx_locations
            warnings.warn('Since `to_midpoint` or `to_txrx` is used,'
                          ' the final shape of the input tensor will not be the input_shape you assigned.',
                          UserWarning)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_list) / float(self.batch_size)))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_file_list = [self.file_list[k] for k in indexes]
        data = self.get_data(temp_file_list)
        return data

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @abstractmethod
    def get_data(self, temp_file_list):
        """
        Extract data from each file.

        Parameters
        ----------
        temp_file_list: list
            List of input data path.

        Returns
        -------
            data

        """
        raise NotImplementedError


class DataGenerator(BaseGenerator):
    """
    Custom Sequence object to train a model on out-of-memory data sets.
    """
    def get_data(self, temp_file_list):
        resistance = np.empty((len(temp_file_list), *self.input_shape))
        resistivity_log10 = np.empty((len(temp_file_list), *self.output_shape))
        for i, file in enumerate(temp_file_list):
            data = read_pkl(file)
            if self.preprocess['to_midpoint']['perform']:
                resistance[i, ] = to_midpoint(
                    data['resistance'], self.Tx_locations, self.Rx_locations
                )
            elif self.preprocess['to_txrx']['perform']:
                resistance[i, ] = to_txrx(
                    data['resistance'], self.Tx_locations, self.Rx_locations
                )
            else:
                resistance[i, ] = data['resistance'].reshape(self.input_shape)
            resistivity_log10[i, ] = data['resistivity_log10'].reshape(self.output_shape)

        for k, v in self.preprocess.items():
            if k == 'add_noise' and v.get('perform'):
                add_noise(resistance, **v.get('kwargs'))
            elif k == 'log_transform' and v.get('perform'):
                log_transform(resistance, **v.get('kwargs'))

        return resistance, resistivity_log10


class PredictGenerator(DataGenerator):

    def get_data(self, temp_file_list):
        resistance = np.empty((len(temp_file_list), *self.input_shape))
        for i, file in enumerate(temp_file_list):
            data = read_pkl(file)
            if self.preprocess['to_midpoint']['perform']:
                resistance[i, ] = to_midpoint(
                    data['resistance'], self.Tx_locations, self.Rx_locations
                )
            elif self.preprocess['to_txrx']['perform']:
                resistance[i, ] = to_txrx(
                    data['resistance'], self.Tx_locations, self.Rx_locations
                )
            else:
                resistance[i, ] = data['resistance'].reshape(self.input_shape)

        for k, v in self.preprocess.items():
            if k == 'add_noise' and v.get('perform'):
                add_noise(resistance, **v.get('kwargs'))
            elif k == 'log_transform' and v.get('perform'):
                log_transform(resistance, **v.get('kwargs'))

        return resistance
