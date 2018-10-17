import gc

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Convolution2D


class HyperparamsDecoder:

    def __init__(self):
        self.latent_dim = 3
        self.activation = 'relu'
        self.output_activation = 'sigmoid'
        self.num_affine_layers = 1
        self.affine_width = [128]


class DecoderConvolution2D:

    def __init__(self, output_shape,  hyperparameters=HyperparamsDecoder()):
        self.output_shape = output_shape
        self.hparams = hyperparameters
        self.input = Input(shape=(self.hparams.latent_dim,), name='z_sampling')

    def __repr__(self):
        return '2D Convolutional Decoder.'

    def _affine_layers(self, x):
        """Compose fully connected layers.

        Parameters
        ----------
        x : tensor
            Input from latent dimension.

        Returns
        -------
        fc_layers : list
            Fully connected layers from embedding to convolution layers.
        """
        if len(self.hparams.affine_width)!=self.hparams.num_affine_layers:
            raise Exception("Number of affine width parameters must equal the number of affine layes")

        fc_layers = []
        for i in range(self.hparams.num_affine_layers):
            x = Dense(self.hparams.affine_width[i],
                      activation=self.hparams.activation)(x)
            fc_layers.append(x)

        def x
        gc.collect()

        return fc_layers

    def _conv_layers(self, x):
        """Compose convolution layers.

        Parameters
        ----------
        x : tensorflow tensor 
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

