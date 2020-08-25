"""Module for handling data.

Author: Ryan Strauss
"""
import h5py
import numpy as np
import os
import tensorflow as tf
from scipy.sparse import load_npz
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
import re
import pandas as pd

DATA_FILE = "Mg22_alphaalpha_digiSim.h5"

X_COL = 0
Y_COL = 1
Z_COL = 2
CHARGE_COL = 4

CLASS_NAMES = ['beam', 'reaction']

def get_event_by_index(hf, i):
    return hf["Event_[{}]".format(i)][:]

def get_event_from_key(key):
    re_m = re.match("Event_\[(\d*)\]", key)
    if re_m:
        return int(re_m.groups()[0])
    else: return None

def get_label(key):
    """ Return label for an event number: 
            even: beam = 0
            odd: reaction = 1
    """
    re_m = re.match("Event_\[(\d*)\]", key)
    if re_m:
        event = int(re_m.groups()[0])
        return event % 2
    else:
        print("WARNING: could not determine label for key:", key)
        return None
    
def get_label_name(key):
    return CLASS_NAMES[get_label(key)]
    
def read_and_label_data(data_dir):
    """Read data into numpy arrays and label it. Return dictionary with data and labels."""
    print('Reading and labelling data...')
    data = {}
    
    filename = os.path.join(data_dir, DATA_FILE)
    h5_file = h5py.File(filename, "r")

    for key in h5_file.keys():
        xyzs = np.asarray(pd.DataFrame(h5_file[key][:]))
        if xyzs.shape[0] > 0:
            #data.append([xyzs, get_label(key)])
            data[get_event_from_key(key)] = ([xyzs, get_label(key)])
        else:
            print("WARNING,", key, "has no pads firing. Removing event ...")
    h5_file.close()
    print("\tDone")
    return data

def load_image_h5(path,
                  categorical=False,
                  flatten=False,
                  max_charge=False,
                  binary=False,
                  reverse_labels=False):
    """Loads and returns the requested image data.

        Reads in the specified training data from HDF5 files and returns that data as a numpy array.

        Args:
            path (str): Path to the HDF5 file that should be loaded.
            categorical (bool): Indicator of whether or not targets should be returned as a 2D one-hot encoded array.
            flatten (bool): If true, each training example will be flattened.
            max_charge (bool): Specifies whether or not to return the training set's maximum charge.
            binary (bool): Specifies whether or not the data should be framed as two-class
            (i.e. proton vs. nonproton).
            reverse_labels (bool): If true, labels will be reversed.

        Returns:
            data (ndarray, optional): If loading unlabelled data, the only item returned will be an ndarray containing
            that data.
            features (tuple): The requested data as a tuple of the form (train, test), where train and
            test each have the form (X, y).
            targets (tuple): The corresponding targets.
            max_charge (float, optional): The maximum charge from the returned training set, before normalization.

    """
    
    print("Loading images from file:", path)
    
    h5 = h5py.File(path, 'r')

    if 'images' in h5:
        return h5['images'][:]

    train_features = h5['train_features'][:]
    train_targets = h5['train_targets'][:]
    test_features = h5['test_features'][:]
    test_targets = h5['test_targets'][:]
    if max_charge:
        mc = h5['max_charge'][:][0]
    h5.close()

    num_categories = np.unique(train_targets).shape[0]

    print("\tNumber of categories:", num_categories)

    #if binary:
    #    for i in range(train_targets.shape[0]):
    #        if train_targets[i] != 0:
    #            train_targets[i] = 1
    #    for i in range(test_targets.shape[0]):
    #        if test_targets[i] != 0:
    #            test_targets[i] = 1

    #    num_categories = 2

    #if reverse_labels:
    #    train_targets = train_targets.max() - train_targets
    #    test_targets = test_targets.max() - test_targets

    #if categorical:
    #    train_targets = to_categorical(train_targets, num_categories).astype(np.int8)
    #    test_targets = to_categorical(test_targets, num_categories).astype(np.int8)

    #if flatten:
    #    train_features = np.reshape(train_features, (len(train_features), -1))
    #    test_features = np.reshape(test_features, (len(test_features), -1))

    #if max_charge:
    #    return (train_features, train_targets), (test_features, test_targets), mc

    return (train_features, train_targets), (test_features, test_targets)


def load_discretized_data(dir,
                          prefix='',
                          categorical=False,
                          normalize=True,
                          binary=False):
    """Loads and returns the requested discretized data.

        Args:
            dir (str): Path to the directory containing the data.
            prefix (str): Filename prefix of the data to be loaded.
            categorical (bool): Indicator of whether or not targets should be returned as a 2D one-hot encoded array.
            normalize (bool): Specifies whether or not to normalize the data.
            binary (bool): Specifies whether or not the data should be framed as two-class
            (i.e. proton vs. nonproton).

        Returns:
            features (tuple): The requested data as a tuple of the form (train, test), where train and
            test each have the form (X, y).
            targets (tuple): The corresponding targets.

    """

    filename = os.path.join(dir, prefix + 'voxels.h5')

    print("Loading discretized data from:", filename)

    h5 = h5py.File(filename, 'r')
    train_targets = h5['train_targets'][:]
    test_targets = h5['test_targets'][:]
    h5.close()

    train_features = load_npz(os.path.join(dir, prefix + 'train-features.npz'))
    test_features = load_npz(os.path.join(dir, prefix + 'test-features.npz'))

    if normalize:
        normalizer = Normalizer()
        train_features = normalizer.fit_transform(train_features)
        test_features = normalizer.transform(test_features)

    num_categories = np.unique(train_targets).shape[0]

    #if binary:
    #    for i in range(train_targets.shape[0]):
    #        if train_targets[i] != 0:
    #            train_targets[i] = 1
    #    for i in range(test_targets.shape[0]):
    #        if test_targets[i] != 0:
    #            test_targets[i] = 1

    #    num_categories = 2

    #if categorical:
    #    train_targets = to_categorical(train_targets, num_categories).astype(np.int8)
    #    test_targets = to_categorical(test_targets, num_categories).astype(np.int8)

    return (train_features, train_targets), (test_features, test_targets)
