import gc

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
        self.input = Input(shape=input_shape)
        self.hparams = hyperparameters
        self.graph = self._create_graph()

    def __repr__(self):
        return '2D Convolutional Encoder.'

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

        z_mean = Dense(self.hparams.latent_dim)(fc_layers[-1])
        z_log_var = Dense(self.hparams.latent_dim)(fc_layers[-1])
        z = Lambda(self.sampling, output_shape=(self.hparams.latent_dim,))([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def sampling(self, encoder_output):
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
        z_mean, z_log_var = encoder_output
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def _create_graph(self):
        """Create the keras model."""
        conv_layers = self._conv_layers(self.input)
        flattened = Flatten()(conv_layers[-1])
        z_mean, z_log_var , z = self._affine_layers(flattened)
        graph = Model(self.input, [z_mean, z_log_var, z], name='encoder')
        return graph

    def summary(self):
        print('Convolutional Encoder:')
        self.graph.summary()

    def fit(self):
        self.graph.fit()
