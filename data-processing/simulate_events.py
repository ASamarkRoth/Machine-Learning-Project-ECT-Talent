"""Generates and saves simulated proton and carbon events using pytpc's simulation module.

Author: Ryan Strauss
"""
import math

import click
import os
import pytpc
import yaml
from effsim.effsim import EventSimulator
from effsim.paramgen import uniform_param_generator
from pytpc.hdfdata import HDFDataFile


def _mean_dist(event):
    """Calculates the average distance from origin in the xy plane of all the points in an event."""
    s = 0
    for point in event:
        s += math.sqrt(point[0] ** 2 + point[1] ** 2)
    return s / len(event)


@click.command()
@click.argument('save_path', type=click.Path(exists=False, file_okay=False, dir_okay=True), nargs=1)
@click.option('--tilt', type=click.BOOL, default=True, help='Whether or not the events should be simulated with tilt.')
@click.option('--point_cutoff', type=click.INT, default=150,
              help='The minimum number of points required to be in an event.')
@click.option('--mean_dist', type=click.INT, default=135, help='The maximum average point distance of the events.')
@click.option('--num_events', type=click.INT, default=40000, help='The number of events that should be created.')
@click.option('--prefix', type=click.STRING, default='',
              help='Prefix for the saved file names. By default, there is no prefix.')
def simulate_events(save_path, tilt, point_cutoff, mean_dist, num_events, prefix):
    """Simulates proton and carbon events as specified by the given options.

    The path to which events should be saved is given as an argument.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Proton Events
    with open('config/config_e15503b_p{}.yml'.format('_tilt' if tilt else ''), 'r') as f:
        config = yaml.load(f)

    beam_enu0 = config['beam_enu0']
    beam_mass = config['beam_mass']
    beam_charge = config['beam_charge']
    mass_num = config['mass_num']
    max_beam_angle = (config['max_beam_angle'] * math.pi) / 180
    beam_origin_z = config['beam_origin_z']

    gas = pytpc.gases.InterpolatedGas('isobutane', 19.2)

    pgen = uniform_param_generator(beam_enu0, beam_mass, beam_charge, mass_num, max_beam_angle, beam_origin_z, gas,
                                   num_events * 10)

    sim = EventSimulator(config)

    proton_file_path = os.path.join(save_path, prefix + 'proton.h5')

    with HDFDataFile(proton_file_path, 'w') as hdf:
        evt_id = 0
        for p in pgen:
            if evt_id >= num_events:
                break
            else:
                try:
                    evt, ctr = sim.make_event(p[0][0], p[0][1], p[0][2], p[0][3], p[0][4], p[0][5])
                except IndexError:
                    continue

            pyevt = sim.convert_event(evt, evt_id)

            xyzs = pyevt.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                              baseline_correction=False,
                              cg_times=False)

            if xyzs.shape[0] < point_cutoff or _mean_dist(xyzs) > (mean_dist if mean_dist > 0 else math.inf):
                continue
            else:
                hdf.write_get_event(pyevt)
                evt_id += 1

            if evt_id % 50 == 0:
                print('Proton event {} written...'.format(evt_id))

    # Carbon Events
    with open('config/config_e15503b_C{}.yml'.format('_tilt' if tilt else ''), 'r') as f:
        config = yaml.load(f)

    beam_enu0 = config['beam_enu0']
    beam_mass = config['beam_mass']
    beam_charge = config['beam_charge']
    mass_num = config['mass_num']
    max_beam_angle = (config['max_beam_angle'] * math.pi) / 180
    beam_origin_z = config['beam_origin_z']

    gas = pytpc.gases.InterpolatedGas('isobutane', 19.2)

    Cgen = uniform_param_generator(beam_enu0, beam_mass, beam_charge, mass_num, max_beam_angle, beam_origin_z, gas,
                                   num_events * 10)

    sim = EventSimulator(config)

    carbon_file_path = os.path.join(save_path, prefix + 'carbon.h5')

    with HDFDataFile(carbon_file_path, 'w') as hdf:
        evt_id = 0
        for C in Cgen:
            if evt_id >= num_events:
                break
            else:
                try:
                    evt, ctr = sim.make_event(C[0][0], C[0][1], C[0][2], C[0][3], C[0][4], C[0][5])
                except IndexError:
                    continue

            pyevt = sim.convert_event(evt, evt_id)

            xyzs = pyevt.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                              baseline_correction=False,
                              cg_times=False)

            if xyzs.shape[0] < point_cutoff or _mean_dist(xyzs) > (mean_dist if mean_dist > 0 else math.inf):
                continue
            else:
                hdf.write_get_event(pyevt)
                evt_id += 1

            if evt_id % 50 == 0:
                print('Carbon event {} written...'.format(evt_id))


if __name__ == '__main__':
    simulate_events()
