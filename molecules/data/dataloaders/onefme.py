import os
import numpy as np

from .utils import download_url, makedir_exist_ok


class OneFME:
    """1FME Dataset.

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
      'https://raw.githubusercontent.com/yngtodd/moles/master/1fme/train-contactmaps.npz',
      'https://raw.githubusercontent.com/yngtodd/moles/master/1fme/validation-contactmaps.npz',
      'https://raw.githubusercontent.com/yngtodd/moles/master/1fme/test-contactmaps.npz',
    ]

    training_contactmap_file = 'train_contactmaps.npy'
    validation_contactmap_file = 'validation_contactmaps.npy'
    test_contactmap_file = 'test_contactmaps.npy'

    def __init__(self, root, partition, download=False):
        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.partition = partition
        if self.partition == 'train':
            data_file = self.training_contactmap_file
        elif self.partition == 'validation':
            data_file = self.validation_contactmap_file
        elif self.partition == 'test':
            data_file = self.test_contactmap_file
        else:
            raise ValueError("Partition must either be 'train', 'validation', or 'test'.")

        self.data = np.load(os.path.join(self.processed_folder, data_file))

    def __len__(self):
        return len(self.data)

    def load_data(self):
        return self.data

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_contactmap_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.validation_contactmap_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_contactmap_file))

    @staticmethod
    def extract_array(npz_path, remove_finished=False):
        print('Extracting {}'.format(npz_path))
        with np.load(npz_path) as data:
            arry = data['arry']
        if remove_finished:
            os.unlink(npz_path)

    def download(self):
        """Download the 1FME data if it doesn't exist in processed_folder already."""

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
        )
        validation_set = (
            read_image_file(os.path.join(self.raw_folder, 'validation-contactmaps.npz')),
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 'test-contactmaps.npz')),
        )

        # Save processed training data
        train_data_path = os.path.join(self.processed_folder, self.training_contactmap_file)
        np.save(train_data_path, training_set[0])

        # Save processed valdation data
        val_data_path = os.path.join(self.processed_folder, self.validation_contactmap_file)
        np.save(val_data_path, validation_set[0])

        #Save processed test data
        test_data_path = os.path.join(self.processed_folder, self.test_contactmap_file)
        np.save(test_data_path, test_set[0])

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
    # Array shape (B x H x W x C)
    return arry.reshape(-1, 28, 28, 1)
