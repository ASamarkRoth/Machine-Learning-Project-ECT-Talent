"""Script for training a logistic regression classifier.

Author: Ryan Strauss
"""

import pickle

import click
import numpy as np
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

from utils.data import load_discretized_data

FEATURES = 0
TARGETS = 1


@click.command()
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), nargs=1)
@click.argument('log_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), nargs=1)
@click.option('--prefix', type=click.STRING, default='', nargs=1,
              help='Filename prefix of the data to be loaded. No prefix by default.')
@click.option('--rebalance', is_flag=True,
              help='If flag is set, class weighting will be used during training to rebalance '
                   'an uneven distribution of classes in the training set.')
@click.option('--binary', type=click.BOOL, default=True, nargs=1,
              help='If true, the labels will be collapsed to binary values, where any non-zero label will become a 1.')
@click.option('--examples_limit', type=click.INT, default=-1, nargs=1,
              help='Limit on the number of training examples to use during training.')
@click.option('--seed', type=click.INT, default=71, nargs=1, help='Random seed.')

def main(data_dir, log_dir, prefix, rebalance, binary, examples_limit, seed):
    train(data_dir, log_dir, prefix, rebalance, binary, examples_limit, seed)

def train(data_dir, log_dir, prefix='', rebalance=False, binary=True, examples_limit=-1, seed=71,
          solver='saga', penalty='l2', regularization_strength=20, cv_fold=5, max_iter=100, tol=1e-4,
         ):
    """This function will train a logistic regression classifier."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set random seeds
    np.random.seed(seed)

    # Load data
    train, _ = load_discretized_data(data_dir, prefix=prefix, binary=binary)

    if examples_limit == -1:
        examples_limit = train[TARGETS].shape[0]

    if rebalance:
        class_weight = 'balanced'
    else:
        class_weight = None

    # Build model
    model = LogisticRegressionCV(
    #model = LogisticRegression(
            solver='saga', n_jobs=-1, class_weight=class_weight,
            penalty='l2', #regularization (penalization)
            Cs=regularization_strength, #inverse regularization strength (if int, set in a scale)
            cv=cv_fold, #cross-validation fold
            max_iter=max_iter,
            tol=1e-4
            )

    # Train the model
    model.fit(train[FEATURES][:examples_limit],
              train[TARGETS][:examples_limit])

    model_filename = os.path.join(log_dir, 'model.p')
    pickle.dump(model, open(model_filename, 'wb'))


if __name__ == '__main__':
    main()
