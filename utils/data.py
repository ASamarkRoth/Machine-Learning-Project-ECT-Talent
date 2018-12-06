"""Module used for loading data.

Author: Ryan Strauss
"""
import h5py
import numpy as np
import os
import tensorflow as tf
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

CLASS_NAMES = ['proton', 'carbon', 'junk']


class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        # Initialize the iterator with the data feed_dict
        self.iterator_initializer_func(session)


def data_to_stream(data, batch_size):
    """Transforms a numpy array to a batched data stream.

    Parameters:
        data: a numpy array
        batch_size: batch size of returned stream

    Returns:
        data_feeder: an iterable data stream
    """
    iterator_initializer_hook = IteratorInitializerHook()
    with tf.device('/cpu:0'):
        placeholder = tf.placeholder(data.dtype, data.shape)
        dataset = tf.data.Dataset.from_tensor_slices(placeholder)
        dataset = dataset.batch(batch_size, drop_remainder=True).repeat()
        iterator = dataset.make_initializable_iterator()
        iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer,
                                                                                    feed_dict={placeholder: data})
    return iterator, iterator_initializer_hook


def decode_predictions(preds, top=3):
    """Decodes an array of predictions by returning the top classes (as specified) with their class names.

    Parameters:
        preds: An array of predictions from a model.
        top (int): The number of predictions to be returned. Default is 3.

    Return:
        decoded_preds: A list of the top predictions, sorted from highest probability to lowest, as tuples
                       with the class name and probability.
    """
    decoded = [(CLASS_NAMES[i], p) for i, p in enumerate(preds)]
    decoded.sort(key=lambda o: o[1], reverse=True)
    top = min(top, len(decoded))
    return decoded[:top]


def load_image_h5(path,
                  categorical=False,
                  flatten=False,
                  max_charge=False,
                  binary=False):
    """Loads and returns the requested image data.

        Reads in the specified training data from HDF5 files and returns that data as a numpy array.

        Args:
            path (str): Path to the HDF5 file that should be loaded.
            categorical (bool): Indicator of whether or not targets should be returned as a 2D one-hot encoded array.
            flatten (bool): If true, each training example will be flattened.
            max_charge (bool): Specifies whether or not to return the training set's maximum charge.
            binary (bool): Specifies whether or not the data should be framed as two-class
                           (i.e. proton vs. nonproton).

        Returns:
            features (tuple): The requested data as a tuple of the form (train, test), where train and
                                   test each have the form (X, y).
            targets (tuple): The corresponding targets.
            max_charge (float, optional): The maximum charge from the returned training set, before normalization.

    """
    h5 = h5py.File(path, 'r')

    if 'images' in h5:
        return h5['images'][:]

    train_features = h5['train_features'][:]
    train_targets = h5['train_targets'][:]
    test_features = h5['test_features'][:]
    test_targets = h5['test_targets'][:]
    mc = h5['max_charge'][:][0]
    h5.close()

    num_categories = np.unique(train_targets).shape[0]

    if binary:
        for i in range(train_targets.shape[0]):
            if train_targets[i] != 0:
                train_targets[i] = 1
        for i in range(test_targets.shape[0]):
            if test_targets[i] != 0:
                test_targets[i] = 1

        num_categories = 2

    if categorical:
        train_targets = to_categorical(train_targets, num_categories).astype(np.int8)
        test_targets = to_categorical(test_targets, num_categories).astype(np.int8)

    if flatten:
        train_features = np.reshape(train_features, (len(train_features), -1))
        test_features = np.reshape(test_features, (len(test_features), -1))

    if max_charge:
        return (train_features, train_targets), (test_features, test_targets), mc

    return (train_features, train_targets), (test_features, test_targets)


def load_discretized_data(dir,
                          prefix='',
                          categorical=False,
                          standardize=True,
                          binary=False):
    """Loads and returns the requested discretized data.

        Args:
            dir (str): Path to the directory containing the data.
            prefix (str): Filename prefix of the data to be loaded.
            categorical (bool): Indicator of whether or not targets should be returned as a 2D one-hot encoded array.
            standardize (bool): Specifies whether or not to standardize the data.
            binary (bool): Specifies whether or not the data should be framed as two-class
                           (i.e. proton vs. nonproton).

        Returns:
            features (tuple): The requested data as a tuple of the form (train, test), where train and
                                   test each have the form (X, y).
            targets (tuple): The corresponding targets.

    """
    targets_filename = os.path.join(dir, prefix + 'targets.h5')
    h5 = h5py.File(targets_filename, 'r')
    train_targets = h5['train_targets'][:]
    test_targets = h5['test_targets'][:]
    h5.close()

    train_features = load_npz(os.path.join(dir, prefix + 'train-features.npz'))
    test_features = load_npz(os.path.join(dir, prefix + 'test-features.npz'))

    if standardize:
        scaler = StandardScaler(with_mean=False)
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

    num_categories = np.unique(train_targets).shape[0]

    if binary:
        for i in range(train_targets.shape[0]):
            if train_targets[i] != 0:
                train_targets[i] = 1
        for i in range(test_targets.shape[0]):
            if test_targets[i] != 0:
                test_targets[i] = 1

        num_categories = 2

    if categorical:
        train_targets = to_categorical(train_targets, num_categories).astype(np.int8)
        test_targets = to_categorical(test_targets, num_categories).astype(np.int8)

    return (train_features, train_targets), (test_features, test_targets)
