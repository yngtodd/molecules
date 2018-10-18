import gc

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Flatten


class HyperparamsCVAE:

    def __init__(self):
        self.latent_dim = 3


class CVAE:

    def __init__(self, encoder, hyperparameters=HyperparamsCVAE()):
        self.hparams = hyperparameters
        self.encoder = encoder
        self.graph = self._create_graph()

    def __repr__(self):
        return 'Convolutional Variational Autoencoder.'

    def summary(self):
        print('Convolutional Variational Autoencoder\n')
        self.encoder.summary()

    def _create_graph(self):
        encoder = self.encoder.graph
        z = Lambda(self.sampling)([encoder[-2], encoder[-1]])
        return Model(self.encoder.input, z)

    def sampling(encoder_output):
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
