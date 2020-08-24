#!/usr/bin/env python3
"""Script for generating discretized ATTPC data.

Author: Anton Såmark-Roth

Original Author: Ryan Strauss
"""
import click
import h5py
import numpy as np
import os
import pandas as pd
#import pytpc
import scipy as sp
from sklearn.utils import shuffle

import data_processing.generate_images as gi

import math

X_DISC = 20
Y_DISC = 20
Z_DISC = 20

DATA_FILE = "Mg22_alphaalpha_digiSim.h5"

def generate_voxelised_data_set(data_dir, save_dir, prefix):
    # Create empty array to hold data
    data = []

    run_filename = os.path.join(data_dir, DATA_FILE)
    raw_data = list(gi.read_and_label_data(data_dir).values())
    for i, l in enumerate(raw_data):
        xyzs = raw_data[i][0]
        data.append([discretize_grid_charge(xyzs, X_DISC, Y_DISC, Z_DISC), raw_data[i][1]])

    # Split into train and test sets
    data = shuffle(data)
    partition = int(len(data) * 0.8)
    train = data[:partition]
    test = data[partition:]

    train_features = [t[0] for t in train]
    train_targets = [t[1] for t in train]
    test_features = [t[0] for t in test]
    test_targets = [t[1] for t in test]

    train_features = sp.sparse.vstack(train_features, format='csr')
    test_features = sp.sparse.vstack(test_features, format='csr')

    # Save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sp.sparse.save_npz(os.path.join(save_dir, '{}train-features.npz'.format(prefix)), train_features)
    sp.sparse.save_npz(os.path.join(save_dir, '{}test-features.npz'.format(prefix)), test_features)
    h5 = h5py.File(os.path.join(save_dir, '{}voxels.h5'.format(prefix)), 'w')
    h5.create_dataset('train_targets', data=train_targets)
    h5.create_dataset('test_targets', data=test_targets)
    h5.close()

DETECTOR_LENGTH = 1250.0
DETECTOR_RADIUS = 275.0

def discretize_grid_charge(xyz, x_disc, y_disc, z_disc):
    """Discretizes AT-TPC point cloud data-processing using a grid geometry by totalling
    charge of hits in a given rectangular bucket.

    Parameters
    ----------
    xyz    : point cloud data-processing with shape (n,5) where n is the number of traces
    x_disc : number of slices in x
    y_disc : number of slices in y
    z_disc : number of slices in z

    Returns
    -------
    The discretized data-processing in a csr sparse matrix of shape (1, x_disc*y_disc*z_disc)
    """

    # calculate desired discreuniform_param_generatortization resolution
    disc_elements = x_disc * y_disc * z_disc

    buckets = []
    charges = []

    for point in xyz:
        # check that z-coordinate of point is in appropriate range
        if point[2] > DETECTOR_LENGTH:
            continue

        x_bucket = math.floor(((point[0] + DETECTOR_RADIUS) / (2 * DETECTOR_RADIUS)) * x_disc)
        y_bucket = math.floor(((point[1] + DETECTOR_RADIUS) / (2 * DETECTOR_RADIUS)) * y_disc)
        z_bucket = math.floor((point[2] / DETECTOR_LENGTH) * z_disc)

        bucket_num = z_bucket * x_disc * y_disc + x_bucket + y_bucket * x_disc

        buckets.append(bucket_num)

        # scaling by factor of 1000
        charges.append(point[3] / 1000)

    cols = buckets
    rows = np.zeros(len(cols))
    data = charges

    # automatically sums charge values for data-processing occuring at the (row, col)
    discretized_data_sparse_CHARGE = sp.sparse.csr_matrix((data, (rows, cols)), shape=(1, disc_elements))
    return discretized_data_sparse_CHARGE



@click.command()
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), nargs=1)
@click.option('--save_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), default='',
              help='Where to save the generated data.')
@click.option('--prefix', type=click.STRING, default='',
              help='Prefix for the saved file names and/or files read in. By default, there is no prefix.')

def main(data_dir, save_dir, prefix):
    """This script will discretize and save ATTPC event data.
    """

    generate_voxelised_data_set(data_dir, save_dir, prefix)

if __name__ == '__main__':
    main()
