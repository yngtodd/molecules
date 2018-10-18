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


class DecoderConvolution2D:

    def __init__(self, hyperparameters=HyperparamsDecoder()):
        self.hparams = hyperparameters
        self.input = Input(shape=(self.hparams.latent_dim,), name='z_sampling')
