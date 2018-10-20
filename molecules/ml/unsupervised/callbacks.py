import keras
from random import randint


class Embeddings(keras.callbacks.Callback):
    """Saves embeddings of random samples."""
    def __init__(self, data):
        self.data = data

    def on_train_begin(self, logs={}):
        self.embeddings = []
        self.data_index = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        idx = randint(0, len(self.data))
        embedding = self.model.embed(data[idx])
        self.embeddings.append(embedding)
        self.data_index.append(idx)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
