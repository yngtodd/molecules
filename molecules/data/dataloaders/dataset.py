import os
import numpy as np


class ContactMapDataset:
    """Data handing for Numpy stored contact maps"""
    def __init__(self, path):
        self.path = path

    def load_data(self, shape=None):
        """Load numpy array data.

        Parameters
        ----------
        shape : tuple, optional
            Shape of the data. Format: (H x W x C)

        Returns
        -------
        X_train : np.ndarray
            Training set.

        X_test : np.ndarray
            Test set.
        """
        train = os.path.join(self.path, 'train')
        test = os.path.join(self.path, 'test')
        X_train = np.load(train)
        X_test = np.load(test)

        if shape:
            X_train = X_train.reshape((-1, shape[0], shape[1], shape[2]))
            X_test = X_test.reshape((-1, shape[0], shape[1], shape[2]))

        return X_train, X_test
