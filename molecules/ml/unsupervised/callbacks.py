import os
import numpy as np

import keras


class EmbeddingCallback(keras.callbacks.Callback):
    """Saves embeddings of random samples.

    Parameters
    ----------
    data : np.ndarray
        Dataset from which to sample for embeddings.
    """
    def __init__(self, data, graph):
        self.data = data
        self.graph = graph

    def on_train_begin(self, logs={}):
        self.embeddings = []
        self.data_index = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        idx = np.random.randint(0, len(self.data))
        embedding = self.graph.embed(self.data[idx-1:idx])
        self.embeddings.append(embedding)
        self.data_index.append(idx)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def save_embeddings(self, path, filename):
        """Save embeddings and index of associated data point.

        Parameters
        ----------
        path : str
            Path to save embeddings and indices.

        filename : str
            Name of the experiment run.
        """
        embed_name = filename + '_embeddings'
        idx_name = filename + '_idx'
        embed_path = os.path.join(path, embed_name)
        idx_path = os.path.join(path, idx_name)
        np.save(embed_path, self.embeddings)
        np.save(idx_path, self.data_index)

