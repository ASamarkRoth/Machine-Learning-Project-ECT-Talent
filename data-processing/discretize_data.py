"""Script for generating discretized ATTPC data.

Author: Ryan Strauss
"""
import click
import h5py
import numpy as np
import os
import pandas as pd
import pytpc
import scipy as sp
from sklearn.utils import shuffle

from utils import data_discretization as dd

X_DISC = 30
Y_DISC = 30
Z_DISC = 30
RUNS = ['0130', '0210']


def real(data_dir, save_dir, prefix):
    # Create empty array to hold data
    data = []

    for run in RUNS:
        run_filename = os.path.join(data_dir, 'run_{}.h5'.format(run))
        labels_filename = os.path.join(data_dir, 'run_{}_labels.csv'.format(run))
        events = pytpc.HDFDataFile(run_filename, 'r')
        labels = pd.read_csv(labels_filename, sep=',')

        proton_indices = labels.loc[(labels['label'] == 'p')]['evt_id'].values
        carbon_indices = labels.loc[(labels['label'] == 'c')]['evt_id'].values
        junk_indices = labels.loc[(labels['label'] == 'j')]['evt_id'].values

        for evt_id in proton_indices:
            event = events[str(evt_id)]
            xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                              baseline_correction=False,
                              cg_times=False)

            data.append([dd.discretize_grid_charge(xyzs, X_DISC, Y_DISC, Z_DISC), 0])

        for evt_id in carbon_indices:
            event = events[str(evt_id)]
            xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                              baseline_correction=False,
                              cg_times=False)

            data.append([dd.discretize_grid_charge(xyzs, X_DISC, Y_DISC, Z_DISC), 1])

        for evt_id in junk_indices:
            event = events[str(evt_id)]
            xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                              baseline_correction=False,
                              cg_times=False)

            data.append([dd.discretize_grid_charge(xyzs, X_DISC, Y_DISC, Z_DISC), 2])

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
    sp.sparse.save_npz(os.path.join(save_dir, '{}train-features.npz'.format(prefix)), train_features)
    sp.sparse.save_npz(os.path.join(save_dir, '{}test-features.npz'.format(prefix)), test_features)
    h5 = h5py.File(os.path.join(save_dir, '{}targets.h5'.format(prefix)), 'w')
    h5.create_dataset('train_targets', data=train_targets)
    h5.create_dataset('test_targets', data=test_targets)
    h5.close()


def simulated(data_dir, save_dir, prefix, noise):
    print('Starting...')

    proton_events = pytpc.HDFDataFile(os.path.join(data_dir, prefix + 'proton.h5'), 'r')
    carbon_events = pytpc.HDFDataFile(os.path.join(data_dir, prefix + 'carbon.h5'), 'r')

    # Create empty array to hold data
    data = []

    # Add proton events to data array
    for i, event in enumerate(proton_events):
        xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                          baseline_correction=False, cg_times=False)

        if noise:
            xyzs = dd.add_noise(xyzs).astype('float32')

        data.append([dd.discretize_grid_charge(xyzs, X_DISC, Y_DISC, Z_DISC), 0])

        if i % 50 == 0:
            print('Proton event ' + str(i) + ' added.')

    # Add carbon events to data array
    for i, event in enumerate(carbon_events):
        xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                          baseline_correction=False, cg_times=False)

        if noise:
            xyzs = dd.add_noise(xyzs).astype('float32')

        data.append([dd.discretize_grid_charge(xyzs, X_DISC, Y_DISC, Z_DISC), 1])

        if i % 50 == 0:
            print('Carbon event ' + str(i) + ' added.')

    # Create junk events
    for i in range(len(proton_events)):
        xyzs = np.empty([1, 4])
        xyzs = dd.add_noise(xyzs).astype('float32')
        data.append([dd.discretize_grid_charge(xyzs, X_DISC, Y_DISC, Z_DISC), 2])

        if i % 50 == 0:
            print('Junk event ' + str(i) + ' added.')

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
    sp.sparse.save_npz(os.path.join(save_dir, '{}train-features.npz'.format(prefix)), train_features)
    sp.sparse.save_npz(os.path.join(save_dir, '{}test-features.npz'.format(prefix)), test_features)
    h5 = h5py.File(os.path.join(save_dir, '{}targets.h5'.format(prefix)), 'w')
    h5.create_dataset('train_targets', data=train_targets)
    h5.create_dataset('test_targets', data=test_targets)
    h5.close()


@click.command()
@click.argument('type', type=click.Choice(['real', 'sim']), nargs=1)
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), nargs=1)
@click.option('--save_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), default='',
              help='Where to save the generated data.')
@click.option('--prefix', type=click.STRING, default='',
              help='Prefix for the saved file names and/or files read in. By default, there is no prefix.')
@click.option('--noise', type=click.BOOL, default=True,
              help='Whether or not to add artificial noise to simulated data.')
def main(type, data_dir, save_dir, prefix, noise):
    """This script will discretize and save ATTPC event data.

    When using real data, this script will look for runs 0130 and 0210, as these are the runs that have
    been partially hand-labeled.
    """
    if type == 'real':
        real(data_dir, save_dir, prefix)
    else:
        simulated(data_dir, save_dir, prefix, noise)


if __name__ == '__main__':
    main()
