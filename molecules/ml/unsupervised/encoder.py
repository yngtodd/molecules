import gc

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Convolution2D


class EncoderHyperparams:
    
    def __init__(self):
        self.num_conv_layers = 4
        self.filters = [64, 64, 64, 64]
        self.kernels = [3, 3, 3, 3]
        self.strides = [1, 2, 1, 1]
        self.activation = 'relu'
        self.num_affine_layers = 1
        self.fc_width = [128]
        self.dropout = [0]


class ConvolutionalEncoder2D:
    
    def __init__(self, hyperparameters, input_shape):
        self.hparams = hyperparameters
        self.input = Input(shape=input_shape)
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
        fc_layers = []
        for i in range(self.hparams.num_affine_layers):
            x = Dense(self.hparams.fc_width[i], 
                      activation=self.hparams.activation)(Dropout(self.hparams.dropout[i])(x))
            fc_layers.append(x);

        del x
        gc.collect()

        return fc_layers

    
    def _create_graph(self):
        """Create the keras model."""
        conv_layers = self._conv_layers(self.input)
        flattened = Flatten()(conv_layers[-1])
        fc_layers = self._affine_layers(flattened)
        graph = Model(self.input, fc_layers)
        return graph
    
    def summary(self):
        print('Convolutional Encoder:')
        self.graph.summary()
    
    def fit(self):
        self.graph.fit()
