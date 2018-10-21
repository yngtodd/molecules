import gc
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Convolution2D


class HyperparamsEncoder:

    def __init__(self):
        self.num_conv_layers = 3
        self.filters = [64, 64, 64]
        self.kernels = [3, 3, 3]
        self.strides = [1, 2, 1]
        self.activation = 'relu'
        self.num_affine_layers = 1
        self.affine_width = [128]
        self.dropout = [0]
        self.latent_dim = 3


class EncoderConvolution2D:

    def __init__(self, input_shape, hyperparameters=HyperparamsEncoder()):
        self.input_shape = input_shape
        self.input = Input(shape=input_shape)
        self.hparams = hyperparameters
        self.graph = self.create_graph(self.input)
        self.embedder = Model(self.input, self.z_mean)

    def __repr__(self):
        return '2D Convolutional Encoder.'

    def summary(self):
        print('Convolutional Encoder:')
        self.embedder.summary()

    def embed(self, data):
        """Embed a datapoint into the latent space.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        np.ndarray of embeddings.
        """
        return self.embedder.predict(data)

    def _conv_layers(self, x):
        """Compose convolution layers.

        Parameters
        ----------
        x : keras.layers.Input
            Shape of the image input.

        Returns
        -------
        conv2d_layers : list
            Convolution layers
        """
        if len(self.hparams.filters)!=self.hparams.num_conv_layers:
            raise Exception("number of filters must equal number of convolutional layers.")
        if len(self.hparams.kernels)!=self.hparams.num_conv_layers:
            raise Exception("number of kernels must equal number of convolutional layers.")
        if len(self.hparams.filters)!=self.hparams.num_conv_layers:
            raise Exception("number of strides must equal length of convolutional layers.")

        conv2d_layers = []
        for i in range(self.hparams.num_conv_layers):
            x = Convolution2D(self.hparams.filters[i],
                              self.hparams.kernels[i],
                              strides=self.hparams.strides[i],
                              activation=self.hparams.activation,
                              padding='same')(x)
            conv2d_layers.append(x)

        del x
        gc.collect()

        return conv2d_layers

    def _affine_layers(self, x):
        """Compose fully connected layers.

        Parameters
        ----------
        x : tensorflow Tensor
            Flattened tensor from convolution layers.

        Returns
        -------
        fc_layers : list
            Fully connected layers for embedding.
        """
        if len(self.hparams.affine_width)!=self.hparams.num_affine_layers:
            raise Exception("number of affine width parameters must equal the number of affine layers")
        if len(self.hparams.dropout)!=self.hparams.num_affine_layers:
            raise Exception("number of dropout parameters must equal the number of affine layers")

        fc_layers = []
        for i in range(self.hparams.num_affine_layers):
            x = Dense(self.hparams.affine_width[i],
                      activation=self.hparams.activation)(Dropout(self.hparams.dropout[i])(x))
            fc_layers.append(x);

        del x
        gc.collect()

        self.z_mean = Dense(self.hparams.latent_dim)(fc_layers[-1])
        self.z_log_var = Dense(self.hparams.latent_dim)(fc_layers[-1])
        self.z = Lambda(self.sampling, output_shape=(self.hparams.latent_dim,))([self.z_mean, self.z_log_var])

        embed = self.z
        return embed

    def sampling(self, args):
        """
        Reparameterization trick by sampling fr an isotropic unit Gaussian.

        Parameters
        ----------
        encoder_output : tensor
            Mean and log of variance of Q(z|X)

        Returns
        -------
        z : tensor
            Sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def create_graph(self, input_):
        """Create keras model outside of class"""
        self.conv_layers = self._conv_layers(input_)
        self.flattened = Flatten()(self.conv_layers[-1])
        z = self._affine_layers(self.flattened)
        return z

    def _get_final_conv_params(self):
        """Get the number of flattened parameters from final convolution layer."""
        input_ = np.ones((1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        dummy = Model(self.input, self.conv_layers[-1])
        conv_shape = dummy.predict(input_).shape
        self.final_conv_shape = conv_shape[1:]
        self.total_conv_params = 1
        for x in conv_shape: self.total_conv_params *= x
