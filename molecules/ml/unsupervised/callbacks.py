import keras
from random import randint


class Embeddings(keras.callbacks.Callback):
    """Saves embeddings of random samples.

    Parameters
    ----------
    data : np.ndarray
        Dataset from which to sample for embeddings.
    """
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

    def save_embeddings(self, path, filename):
        """Save embeddings and index of associated data point.

        Parameters
        ----------
        path : str
            Path to save embeddings and indices.

        filename : str
            Name of the experiment run.
        """
        embeddings = filename + '_embeddings'
        idx = filename + '_data_index'
        embed_path = os.path.join(path, embeddings)
        idx_path = ps.path.join(path, idx)
        np.save(embed_path, self.embeddings)
        np.save(idx_path, self.data_index)

