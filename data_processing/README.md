# Data

This directory contains code related to the processing of data ATTPC data.

## Event Simulation

The script `simulate_events.py` allows for the simulation of proton and carbon events using the
experimental parameters of the Argon 46 experiment. The scripts will save two HDF5 files, one with
proton events and one with carbon events. The required command-line argument is a path where the
data should be saved. There are three optional arguments that can affect the simulated events:

* `--tilt`

  Specifies whether or not the events should be simulated as though the detector were tilted.
  
* `--point_cutoff`

  Specifies the minimum number of _xyz_ points that must be in a simulated event in order for
  the event to be saved.
  
* `--mean_dist`

  When an event is simulated, the distance is measured between each point in the event and the
  center of the detector in the _xy_ plane. These distances are averaged to produce a single
  measurement for each event. If that average is greated than `mean_dist`, the event is not
  saved.
  
## Generating Images

The script `generate_images.py` can be used to convert a set of events into images for use
with CNNs. The script will save an HDF5 file containing a training and testing dataset.
The saved files can later be read in with `utils.data.load_image_h5`.

The script has three required arguments, `type`, `projection`, and `data_dir`, in that order.
`type` can either be `real` or `sim`, specifiying if the event data was simulated
(using `simulate_events.py`) or is real ATTPC data. `projection` can either be `xy` or `zy`,
specifying which dimensions to use when projecting the events into a 2-dimensional image.
`data_dir` is the directory that contains the event data. Optional arguments are described
in the command-line interface (pass `--help` as an argument).

When using real data, this script will look for runs 0130 and 0210, as these are the runs that have
been partially hand-labeled.