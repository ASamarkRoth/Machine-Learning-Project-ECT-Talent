"""Discretize point cloud data-processing.

This module contains functions for discretizing 3D point cloud data-processing produced by
the Active-Target Time Projection Chamber.

Author: Jack Taylor
"""
import math

import numpy as np
import pytpc
import scipy as sp

DETECTOR_LENGTH = 1250.0
DETECTOR_RADIUS = 275.0


def discretize_cylindrical(xyz, z_disc, radial_disc, angular_disc):
    """(Inefficiently) Discretizes AT-TPC point cloud data-processing using a cylindrical
    geometry. We found that this strategy forces an inappropriate geometry on
    our data-processing.

    Parameters
    ----------
    xyz          : point cloud data-processing with shape (n,5) where n is the number of traces
    z_disc       : number of slices in z
    radial_disc  : number of radial slices/concentric circles
    angular_disc : number of angular wedge slices

    Returns
    -------
    The discretized data-processing in an array of shape (1, z_disc*radial_disc*angular_disc)
    """
    # calculate dimensional increments
    z_inc = DETECTOR_LENGTH / z_disc
    radial_inc = DETECTOR_RADIUS / radial_disc
    angular_inc = (2 * math.pi) / angular_disc

    # create slice boundary arrays
    z_slices = np.arange(DETECTOR_LENGTH, 0.0 - z_inc, -z_inc)
    radial_slices = np.arange(DETECTOR_RADIUS, 0.0 - radial_inc, -radial_inc)
    angular_slices = np.arange(-math.pi, math.pi + angular_inc, angular_inc)

    discretized_data = np.zeros((1, z_disc * radial_disc * angular_disc))
    bucket_num = 0
    num_pts = 0

    for i in range(len(z_slices) - 1):
        for j in range(len(radial_slices) - 1):
            for k in range(len(angular_slices) - 1):
                for point in xyz:
                    if ((z_slices[i] > point[2] > z_slices[i + 1]) and
                            (radial_slices[j] > math.sqrt(point[0] ** 2 + point[1] ** 2) > radial_slices[j + 1]) and
                            (angular_slices[k] < math.atan2(point[1], point[0]) < angular_slices[k + 1])):
                        discretized_data[0][bucket_num] = 1
                        num_pts += 1

            bucket_num += 1

    return discretized_data


def discretize_grid(xyz, x_disc, y_disc, z_disc):
    """Discretizes AT-TPC point cloud data-processing using a grid geometry based on
    whether or not a point exists in a given rectangular bucket.

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

    # calculate desired discretization resolution
    disc_elements = x_disc * y_disc * z_disc

    buckets = []

    for point in xyz:
        x_bucket = math.floor(((point[0] + DETECTOR_RADIUS) / (2 * DETECTOR_RADIUS)) * x_disc)
        y_bucket = math.floor(((point[1] + DETECTOR_RADIUS) / (2 * DETECTOR_RADIUS)) * y_disc)
        z_bucket = math.floor((point[2] / DETECTOR_LENGTH) * z_disc)

        bucket_num = z_bucket * x_disc * y_disc + x_bucket + y_bucket * x_disc
        buckets.append(bucket_num)

    cols = np.unique(buckets)
    rows = np.zeros(len(cols))
    data = np.ones(len(cols))

    discretized_data = sp.sparse.csr_matrix((data, (rows, cols)), shape=(1, disc_elements))

    return discretized_data


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


def add_noise(cleanxyz):
    """Adds random noise to an undiscretized Event object.

    Parameters
    ----------
    cleanxyz : point cloud data-processing with shape (n,5)

    Returns
    -------
    Point cloud data-processing augmented with random noise
    """
    num_noisepts = np.random.randint(20, 300, )

    # generate x and y based on random pad numbers
    paddresses = np.random.randint(0, 10240, (num_noisepts, 1))
    pads = pytpc.generate_pad_plane()
    pcenters = pads.mean(1)
    xys = pcenters[paddresses].reshape(num_noisepts, 2)

    # z and charge values are generated randomly in realistic ranges
    zs = np.random.uniform(0, DETECTOR_LENGTH, (num_noisepts, 1))
    charges = np.random.uniform(1, 4000, (num_noisepts, 1))

    noise_mat = np.hstack((xys, zs, charges))
    return np.vstack((cleanxyz, noise_mat))


def create_noise_events(num_evts, x_disc, y_disc, z_disc, charge):
    """Creates discretized random noise events.

    Parameters
    ----------
    num_evts : the number of desired noise events
    x_disc  : number of slices in x
    y_disc  : number of slices in y
    z_disc  : number of slices in z

    Returns
    -------
    A numpy array of shape (num_evts, x_disc*y_disc*z_disc).
    """
    disc_evts = []

    for evt_id in range(num_evts):
        empty_evt = np.empty([1, 4])
        noise_evt = add_noise(empty_evt)

        if charge:
            disc_evts.append(discretize_grid_charge(noise_evt, x_disc, y_disc, z_disc))
        else:
            disc_evts.append(discretize_grid(noise_evt, x_disc, y_disc, z_disc))
        if evt_id % 1000 == 0:
            print("Discretized event " + str(evt_id))

    discretized_data = sp.sparse.vstack(disc_evts, format='csr')
    print("Data discretization complete.")
    return discretized_data


def bulk_discretize(hdfPath, x_disc, y_disc, z_disc, charge, noise):
    """Discretizes all events in an HDF5 file using a grid geometry.

    Parameters
    ----------
    hdfPath : the system path to the hdf5 file to be
    x_disc  : number of slices in x
    y_disc  : number of slices in y
    z_disc  : number of slices in z
    charge  : boolean variable denoting whether or not charge will be included
              in the discretization
    noise   : boolean variable to add noise to (simulated) data-processing

    Returns
    -------
    A numpy array of shape (n, x_disc*y_disc*z_disc) where n is the number of
    events in the provided hdf5 file.
    """
    disc_evts = []

    with pytpc.HDFDataFile(hdfPath, 'r') as f:
        n_evts = len(f)
        evt_id = 0

        while evt_id < n_evts:
            curEvt = f[evt_id]
            curxyz = curEvt.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                                 baseline_correction=False, cg_times=False)

            if noise:
                curxyz = add_noise(curxyz)

            if charge:
                disc_evts.append(discretize_grid_charge(curxyz, x_disc, y_disc, z_disc))
            else:
                disc_evts.append(discretize_grid(curxyz, x_disc, y_disc, z_disc))
            if evt_id % 1000 == 0:
                print("Discretized event " + str(evt_id))
            evt_id += 1

    discretized_data = sp.sparse.vstack(disc_evts, format='csr')
    print("Data discretization complete.")

    return discretized_data
