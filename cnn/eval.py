"""Script for evaluating a CNN classifier.

Author: Ryan Strauss
"""
import click
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from cyclegan.generate import clean_real_images
from utils.data import load_image_h5, CLASS_NAMES

FEATURES = 0
TARGETS = 1


@click.command()
@click.argument('model_file', type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=1)
@click.argument('data', type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=1)
@click.option('--data_combine', is_flag=True,
              help='If flag is set, the training and test sets within the HDF5 file pointed '
                   'to by `data` will be combined into a single training set.')
@click.option('--binary', type=click.BOOL, default=True, nargs=1,
              help='If true, the labels will be collapsed to binary values, where any non-zero label will become a 1.')
@click.option('--examples_limit', type=click.INT, default=-1, nargs=1,
              help='Limit on the number of examples to use during testing.')
@click.option('--seed', type=click.INT, default=None, nargs=1, help='Random seed.')
@click.option('--reverse_labels', is_flag=True, help='If flag is set, labels will be reversed.')
@click.option('--cyclegan_cleaner', type=click.Path(), default=None, nargs=1)
def main(model_file, data, data_combine, binary, examples_limit, seed, reverse_labels, cyclegan_cleaner):
    """This script will evaluate a CNN classifier.

    Loss, accuracy, and classification metrics are printed to the console.
    """
    assert model_file.endswith('.h5'), 'model_file must point to an HDF5 file'
    assert data.endswith('.h5'), 'data must point to an HDF5 file'

    # Set random seeds
    if seed is not None:
        np.random.seed(seed)
        tf.set_random_seed(seed)

    # Load data
    if data_combine:
        a, b = load_image_h5(data, categorical=True, binary=binary, reverse_labels=reverse_labels)
        test = np.concatenate([a[FEATURES], b[FEATURES]], axis=0), np.concatenate([a[TARGETS], b[TARGETS]], axis=0)
    else:
        _, test = load_image_h5(data, categorical=True, binary=binary, reverse_labels=reverse_labels)

    if cyclegan_cleaner:
        test = clean_real_images(cyclegan_cleaner, test[FEATURES]), test[TARGETS]

    test = test[FEATURES] / 255., test[TARGETS]

    if examples_limit == -1:
        examples_limit = test[TARGETS].shape[0]

    # Load the model
    model = tf.keras.models.load_model(model_file)

    # Evaluate the model
    loss, acc = model.evaluate(test[FEATURES][:examples_limit], test[TARGETS][:examples_limit])

    # Make predictions
    preds = np.argmax(model.predict(test[FEATURES][:examples_limit]), axis=1)

    target_names = CLASS_NAMES

    if binary:
        target_names = [CLASS_NAMES[0], 'non-' + CLASS_NAMES[0]]

    if reverse_labels:
        target_names = target_names[::-1]

    # Get classification metrics
    report = classification_report(np.argmax(test[TARGETS][:examples_limit], axis=1), preds,
                                   target_names=target_names,
                                   digits=2)

    # Print the results
    print('****Evaluation Report****')
    print('Loss: {}\nAccuracy: {}\n'.format(loss, acc))
    print('Classification Report:\n')
    print(report)


if __name__ == '__main__':
    main()
