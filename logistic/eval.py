"""Script for evaluating a logistic regression classifier.

Author: Ryan Strauss
"""
import pickle

import click
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

from data_processing.data import CLASS_NAMES, load_discretized_data

FEATURES = 0
TARGETS = 1


@click.command()
@click.argument('model_file', type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=1)
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), nargs=1)
@click.option('--prefix', type=click.STRING, default='', nargs=1,
              help='Filename prefix of the data to be loaded. No prefix by default.')
@click.option('--binary', type=click.BOOL, default=True, nargs=1,
              help='If true, the labels will be collapsed to binary values, where any non-zero label will become a 1.')
@click.option('--examples_limit', type=click.INT, default=-1, nargs=1,
              help='Limit on the number of examples to use during testing.')
@click.option('--seed', type=click.INT, default=71, nargs=1, help='Random seed.')

def main(model_file, data_dir, prefix, binary, examples_limit, seed):
    eval(model_file, data_dir, prefix, binary, examples_limit, seed)
    
def eval(model_file, data, name='LogisticRegression', binary=True, examples_limit=-1, seed=71):
    """This function will evaluate a logistic regression classifier.

    Accuracy and classification metrics are printed to the console.
    """
    assert model_file.endswith('.pkl'), 'model_file must point to a pickle file'

    # Set random seeds
    np.random.seed(seed)

    # Load data
    #_, test = load_discretized_data(data_dir, prefix=prefix, binary=binary)

    if examples_limit == -1:
        examples_limit = data[TARGETS].shape[0]

    # Load the model
    model = pickle.load(open(model_file, 'rb'))

    # Evaluate the model
    acc = model.score(data[FEATURES][:examples_limit], data[TARGETS][:examples_limit])

    # Make predictions
    preds = model.predict(data[FEATURES][:examples_limit])

    #if binary:
    #    target_names = [CLASS_NAMES[0], 'non-' + CLASS_NAMES[0]]
    #else:
    #    target_names = CLASS_NAMES
        
    target_names = CLASS_NAMES

    # Get classification metrics
    report = classification_report(data[TARGETS][:examples_limit], preds,
                                   target_names=target_names,
                                   digits=2)
    
    plot_confusion_matrix(model, data[FEATURES][:examples_limit], data[TARGETS][:examples_limit], display_labels=CLASS_NAMES)

    # Print the results
    print("\nClassification Report for: {}\n".format(name))
    print('Accuracy: {}\n'.format(acc))
    print(report)


if __name__ == '__main__':
    main()
