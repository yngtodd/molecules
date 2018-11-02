import gc

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

from keras.objectives import binary_crossentropy


class VAE:

    def __init__(self, input_shape, latent_dim, encoder, decoder, optimizer, loss=None):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        self.graph = self._create_graph()
        self.loss = loss if loss is not None else self._vae_loss
        self.graph.compile(optimizer=optimizer, loss=self.loss)

    def __repr__(self):
        return 'Convolutional Variational Autoencoder.'

    def summary(self):
        print('Convolutional Variational Autoencoder\n')
        self.encoder.summary()
        self.decoder.summary()

    def train(self, data, batch_size, epochs=1, shuffle=False,
              validation_data=None, checkpoint=False, path=None, callbacks=None):
        """Train network

        Parameters
        ----------
        data : np.ndarray
            Input data;

        batch_size : int
            Minibatch size

        epochs : int
            Number of epochs to train for

        shuffle : bool
            Whether to shuffle training data.

        validation_data : tuple, optional
            Tuple of np.ndarrays (X,y) representing validation data

        checkpoint : boolean
            Whether to save model after each epoch

        path : str, optional
            Path to save model if checkpoint is set to True
        """
        if checkpoint==True and path==None:
            raise Exception("Please enter a path to save the network")

        if validation_data is not None:
            validation_data = (validation_data, validation_data)

        self.graph.fit(data,data,batch_size,epochs=epochs,shuffle=shuffle,
                       validation_data=validation_data,callbacks=callbacks);

    def decode(self, data):
        """Decode a data point

        Parameters
        -----------
        data : np.ndarray
            Image to be decoded.

        Returns
        -------
        np.ndarray of decodings for data.
        """
        return self.graph.predict(data)

    def embed(self, data):
        """Embed a datapoint into the latent space.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        np.ndarray of embeddings.
        """
        return self.encoder.embed(data)

    def generate(self, embedding):
        """Generate images from embeddings.

        Parameters
        ----------
        embedding : np.ndarray

        Returns
        -------
        generated image : np.nddary
        """
        self.decoder.generate(embedding)

    def save_weights(self, path):
        """Save model weights.

        Parameters
        ----------
        path : str
            Path to save the model weights.
        """
        self.graph.save_weights(path)

    def load_weights(self, path):
        """Load saved model weights.

        Parameters
        ----------
        path: str
            Path to saved model weights.
        """
        self.graph.load_weights(path)

    def _create_graph(self):
        self.input = Input(shape=self.input_shape)
        embed = self.encoder.create_graph(self.input)
        output = self.decoder.create_graph(embed)
        graph = Model(self.input, output, name='CVAE')
        return graph

    def _vae_loss(self, input, output):
        """Loss function for variational autoencoder"""
        input_flat = K.flatten(input)
        output_flat = K.flatten(output)

        xent_loss = self.input_shape[0] * self.input_shape[1] \
                    * binary_crossentropy(input_flat, output_flat)

        kl_loss = - 0.5 * K.mean(1 + self.encoder.z_log_var - K.square(self.encoder.z_mean)
                  - K.exp(self.encoder.z_log_var), axis=-1)

        return xent_loss + kl_loss

