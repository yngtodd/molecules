import gc

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense


class CVAE:

    def __init__(self, input_shape, encoder, decoder):
        self.input = Input(shape=input_shape)
        self.encoder = encoder
        self.decoder = decoder
        self.graph = self._create_graph()

    def __repr__(self):
        return 'Convolutional Variational Autoencoder.'

    def summary(self):
        print('Convolutional Variational Autoencoder\n')
        self.encoder.summary()
        self.decoder.summary()

    def _create_graph(self):
        encoder = self.encoder.graph
        decoder = self.decoder.graph
        output = decoder(encoder(self.input)[2])
        graph = Model(self.input, output, name='VAE')
        return graph
