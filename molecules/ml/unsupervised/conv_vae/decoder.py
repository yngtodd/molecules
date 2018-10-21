import gc

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose


class HyperparamsDecoder:

    def __init__(self):
        self.num_conv_layers = 3
        self.filters = [64, 64, 64]
        self.kernels = [3, 3, 3]
        self.strides = [1, 2, 1]
        self.latent_dim = 3
        self.activation = 'relu'
        self.output_activation = 'sigmoid'
        self.num_affine_layers = 1
        self.affine_width = [128]


class DecoderConvolution2D:

    def __init__(self, output_shape, enc_conv_params, enc_conv_shape, hyperparameters=HyperparamsDecoder()):
        self.output_shape = output_shape
        self.enc_conv_params = enc_conv_params
        self.enc_conv_shape = enc_conv_shape
        self.hparams = hyperparameters
        self.input = Input(shape=(self.hparams.latent_dim,), name='z_sampling')
        self.graph = self.create_graph(self.input)
        self.generator = Model(self.input, self.graph)

    def __repr__(self):
        return '2D Convolutional Decoder.'

    def summary(self):
        print('Convolutional Decoder')
        return self.generator.summary()

    def generate(self, embedding):
        """Generate images from embeddings.

        Parameters
        ----------
        embedding : np.ndarray

        Returns
        -------
        generated image : np.nddary
        """
        self.generator.predict(embedding)

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
        fc_layers = []
        for width in reversed(self.hparams.affine_width):
            x = Dense(width, activation=self.hparams.activation)(x)
            fc_layers.append(x)

        # Since the networks are symmetric, we need a Dense layer to bridge fc layers and conv.
        x = Dense(self.enc_conv_params, activation=self.hparams.activation)(x)
        fc_layers.append(x)

        del x
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

        # Mirroring the encoder network requires reversing its hyperparameters.
        filters = list(reversed(self.hparams.filters))
        kernels = list(reversed(self.hparams.kernels))
        strides = list(reversed(self.hparams.strides))

        conv2d_layers = []
        for i in range(self.hparams.num_conv_layers-1):
            x = Conv2DTranspose(filters[i],
                                kernels[i],
                                strides=strides[i],
                                activation=self.hparams.activation,
                                padding='same')(x)
            conv2d_layers.append(x)

        # Final output is special.
        x = Conv2DTranspose(self.output_shape[2],
                            kernels[-1],
                            strides=strides[-1],
                            activation=self.hparams.output_activation,
                            padding='same')(x)

        conv2d_layers.append(x)

        del x
        gc.collect()

        return conv2d_layers

    def create_graph(self, input_):
        affine_layers = self._affine_layers(input_)
        reshaped = Reshape(self.enc_conv_shape)(affine_layers[-1])
        out_img = self._conv_layers(reshaped)[-1]
        return out_img
