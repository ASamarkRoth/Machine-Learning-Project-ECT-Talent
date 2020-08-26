"""Script for evaluating a FCNN classifier.

Author: Ryan Strauss
"""
import click
import numpy as np
import tensorflow as tf
from scipy import sparse
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import itertools

from data_processing.data import CLASS_NAMES, load_discretized_data

FEATURES = 0
TARGETS = 1


@click.command()
@click.argument('model_file', type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=1)
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), nargs=1)
@click.option('--data_combine', is_flag=True,
              help='If flag is set, the training and test sets within the HDF5 file pointed '
                   'to by `data` will be combined into a single training set.')
@click.option('--prefix', type=click.STRING, default='', nargs=1,
              help='Filename prefix of the data to be loaded. No prefix by default.')
@click.option('--binary', type=click.BOOL, default=True, nargs=1,
              help='If true, the labels will be collapsed to binary values, where any non-zero label will become a 1.')
@click.option('--examples_limit', type=click.INT, default=-1, nargs=1,
              help='Limit on the number of examples to use during testing.')
@click.option('--seed', type=click.INT, default=71, nargs=1, help='Random seed.')
def main(model_file, data_dir, data_combine, prefix, binary, examples_limit, seed):
    eval(model_file, data, data_combine, binary, examples_limit, seed, reverse_labels)

def eval(model_file, data, name='CNN', data_combine=False, binary=True, examples_limit=-1, seed=71, reverse_labels=False):
    """This script will evaluate an FCNN classifier.

    Loss, accuracy, and classification metrics are printed to the console.
    """
    assert model_file.endswith('.h5'), 'model_file must point to an HDF5 file'

    # Set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)

    ## Load data
    #if data_combine:
    #    a, b = load_discretized_data(data_dir, prefix, categorical=True, binary=binary)
    #    test = sparse.vstack([a[FEATURES], b[FEATURES]]), np.concatenate([a[TARGETS], b[TARGETS]], axis=0)
    #else:
    #    _, test = load_discretized_data(data_dir, prefix, categorical=True, binary=binary)

    if examples_limit == -1:
        examples_limit = data[TARGETS].shape[0]

    # Load the model
    model = tf.keras.models.load_model(model_file)

    # Evaluate the model
    loss, acc = model.evaluate(data[FEATURES][:examples_limit], data[TARGETS][:examples_limit])

    # Make predictions
    preds = np.round(model.predict(data[FEATURES][:examples_limit]))
    target_names = CLASS_NAMES

    #if binary:
    #    target_names = [CLASS_NAMES[0], 'non-' + CLASS_NAMES[0]]
    #else:
    #    target_names = CLASS_NAMES

    # Get classification metrics
    report = classification_report(data[TARGETS][:examples_limit], preds, target_names=target_names, digits=2)

    # Plot confusion matrix
    cm = confusion_matrix(y_true=data[TARGETS][:examples_limit], y_pred=preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp = disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
    plt.show()

    # Print the results
    print("\nClassification Report for: {}\n".format(name))
    print('Loss: {}, Accuracy: {}\n'.format(loss, acc))
    print(report)

if __name__ == '__main__':
    main()
