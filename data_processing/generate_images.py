#!/usr/bin/env python3
"""Generate images for CNNs from ATTPC events.

Author: Anton Såmark-Roth

Original Author: Ryan Strauss
"""
import math

import click
import h5py
import matplotlib
import numpy as np
import os
import pandas as pd
import re
#import pytpc
from sklearn.utils import shuffle

from data_processing.data import read_and_label_data, X_COL, Y_COL, Z_COL, CHARGE_COL

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Currently we're setting the image pixel values as the logarithm of the charge!
def _l(a):
    return 0 if a == 0 else math.log10(a)

def transform_data(data):
    """ Transform, shuffle and scale for image data"""
    
    print("Transform, shuffle and scale data ...")
    
    #transform
    log = np.vectorize(_l)
    for event in data:
        event[0][:, CHARGE_COL] = log(event[0][:, CHARGE_COL])
        
    # scale
    max_charge = np.array(list(map(lambda x: x[0][:, CHARGE_COL].max(), data))).max() #wrt to max in data set

    for e in data:
        for point in e[0]:
            point[CHARGE_COL] = point[CHARGE_COL] / max_charge

    # Shuffle data
    data = shuffle(data)
    
    return data, max_charge

def make_train_test_data(data, fraction_train=0.8):
    """Make train test data split"""
    
    print("Split into train and test sets ...")
    partition = int(len(data) * fraction_train)
    train = data[:partition]
    test = data[partition:]

    return train, test

def make_image_features_targets(data, projection, image_size):
    """Generate image features and targets in numpy arrays to be used in training and evaluation"""
    
    print("Make image features and targets ...")
    
    # Make numpy sets
    features = np.empty((len(data), image_size, image_size, 3), dtype=np.uint8)
    targets = np.empty((len(data),), dtype=np.uint8)

    for i, event in enumerate(data):
        e = event[0]
        if e is None:
            print("Event, ", i, "is None:", e)
        if projection == 'zy':
            x = e[:, Z_COL].flatten()
            z = e[:, Y_COL].flatten()
            c = e[:, CHARGE_COL].flatten()
        elif projection == 'xy':
            x = e[:, X_COL].flatten()
            z = e[:, Y_COL].flatten()
            c = e[:, CHARGE_COL].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=image_size)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        image = np.delete(image, 3, axis=2)
        features[i] = image
        targets[i] = event[1]
        plt.close()
    return features, targets

    
def generate_image_data_set(projection, data_dir, save_path, prefix, image_size):
    print("Generating image data set ...")
    
    data = list(read_and_label_data(data_dir).values()) #from dict to list
    #print("Shape:\n\tdata:", len(data))
    data, max_charge = transform_data(data)
    train, test = make_train_test_data(data, fraction_train=0.8)
    
    #print("Shape:\n\ttrain:", len(train), "\n\ttest:", len(test))
    
    train_features, train_targets = make_image_features_targets(train, 'xy', image_size)
    test_features, test_targets = make_image_features_targets(test, 'xy', image_size)
    
    print('Saving to HDF5 file...')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.join(save_path, prefix + 'images.h5')

    # Save to HDF5
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('train_features', data=train_features)
    h5.create_dataset('train_targets', data=train_targets)
    h5.create_dataset('test_features', data=test_features)
    h5.create_dataset('test_targets', data=test_targets)
    h5.create_dataset('max_charge', data=np.array([max_charge]))
    h5.close()


@click.command()
@click.argument('projection', type=click.Choice(['xy', 'zy']), nargs=1)
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), nargs=1)
@click.option('--save_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), default='images/',
              help='Where to save the generated data.')
@click.option('--prefix', type=click.STRING, default='',
              help='Prefix for the saved file names and/or files to read in. By default, there is no prefix.')
@click.option('--image_size', type=click.INT, default=48, nargs=1, help='Pixels per side of image, must be at least 48x48 for transfer learning with imagenet.')
def main(projection, data_dir, save_dir, prefix, image_size):
    """
    This script will generate and save images from processed and simulated ATTPC event data to be used for CNN training.
    """
    
    generate_image_data_set(projection, data_dir, save_dir, prefix, image_size)

if __name__ == '__main__':
    main()
