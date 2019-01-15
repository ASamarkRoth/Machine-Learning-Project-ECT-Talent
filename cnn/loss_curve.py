"""Script for generating loss curves of CNN training.

Author: Ryan Strauss
"""
import glob
import json

import click
import matplotlib.pyplot as plt
import os


@click.command()
@click.argument('root_dir', type=click.Path(exists=True, dir_okay=True, file_okay=False), nargs=1)
@click.option('--json_filename', type=click.STRING, nargs=1, default='history.json',
              help='Name of the JSON files containing training history.')
def main(root_dir, json_filename):
    """This script generates loss curves for CNN models from data saved during training.

    The only argument should be a path to a directory. The script will then generate curves for
    every model that directory tree. Curves are saved as PNG images within the corresponding
    directories.
    """
    assert json_filename.endswith('.json')

    glob_path = os.path.join(root_dir, '**', json_filename)
    paths = glob.glob(glob_path, recursive=True)

    for path in paths:
        directory = path[:-len(json_filename)]
        with open(path, 'r') as fp:
            try:
                history = json.load(fp)
            except json.JSONDecodeError:
                continue
        plt.figure(figsize=(11, 6), dpi=200)
        plt.plot(history['loss'], 'o-', label='Training Loss')
        plt.plot(history['val_loss'], 'o:', color='r', label='Validation Loss')
        plt.legend(loc='best')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(range(0, len(history['loss'])), range(1, len(history['loss']) + 1))
        plt.savefig(os.path.join(directory, 'loss_curve.png'))


if __name__ == '__main__':
    main()
