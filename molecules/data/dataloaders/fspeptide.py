import os
import os.path
import gzip
import numpy as np
import codecs
from .utils import download_url, makedir_exist_ok


class FSPeptide:
    """FS-Peptide Dataset.

    Parameters
    ----------
    root str :
        Root directory of dataset where ``processed/training.npy``
        ``processed/validation.npy and ``processed/test.npy`` exist.

    partition : str
        dataset partition to be loaded.
        Either 'train', 'validation', or 'test'.

    download : bool, optional
        If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
    """
    urls = [
      'https://raw.githubusercontent.com/yngtodd/moles/master/fs-peptide/train-contactmaps.npz',
      'https://raw.githubusercontent.com/yngtodd/moles/master/fs-peptide/train-labels.npz',
      'https://raw.githubusercontent.com/yngtodd/moles/master/fs-peptide/validation-contactmaps.npz',
      'https://raw.githubusercontent.com/yngtodd/moles/master/fs-peptide/validation-labels.npz',
      'https://raw.githubusercontent.com/yngtodd/moles/master/fs-peptide/test-contactmaps.npz',
      'https://raw.githubusercontent.com/yngtodd/moles/master/fs-peptide/test-labels.npz'
    ]

    training_file = 'training'
    validation_file = 'validation'
    test_file = 'test'

    def __init__(self, root, partition, download=False):
        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.partition = partition
        if self.partition == 'train':
            data_file = self.training_file
        elif self.partition == 'validation':
            data_file = self.validation_file
        elif self.partition == 'test':
            data_file = self.test_file
        else:
            raise ValueError("Partition must either be 'train', 'validation', or 'test'.")

        #self.data, self.targets = np.load(os.path.join(self.processed_folder, data_file))

    def __len__(self):
        return len(self.data)

    def load_data(self):
        return self.data, self.targets

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.validation_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_file))

    @staticmethod
    def extract_array(npz_path, remove_finished=False):
        print('Extracting {}'.format(npz_path))
        with np.load(npz_path) as data:
            arry = data['arry']
        if remove_finished:
            os.unlink(npz_path)

    def download(self):
        """Download the FS-Peptide data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_array(npz_path=file_path, remove_finished=False)

        # process and save as numpy files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-contactmaps.npz')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels.npz'))
        )
        validation_set = (
            read_image_file(os.path.join(self.raw_folder, 'validation-contactmaps.npz')),
            read_label_file(os.path.join(self.raw_folder, 'validation-labels.npz'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 'test-contactmaps.npz')),
            read_label_file(os.path.join(self.raw_folder, 'test-labels.npz'))
        )

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            np.save(training_set[0], f)
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            np.save(training_set[1], f)
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            np.save(validation_set[1], f)
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            np.save(validation_set[1], f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            np.save(test_set[0], f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            np.save(test_set[1], f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.partition
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


def read_label_file(path):
    with np.load(path) as data:
        arry = data['arry']
    return arry


def read_image_file(path):
    with np.load(path) as data:
        arry = data['arry']
    return arry.reshape(-1, 21, 21)
