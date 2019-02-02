"""Script for training a FCNN classifier.

Author: Ryan Strauss
"""

import click
import numpy as np
import os
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from utils.data import load_discretized_data

FEATURES = 0
TARGETS = 1


@click.command()
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), nargs=1)
@click.argument('log_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), nargs=1)
@click.argument('hidden_layers', type=click.INT, nargs=-1)
@click.option('--prefix', type=click.STRING, default='', nargs=1,
              help='Filename prefix of the data to be loaded. No prefix by default.')
@click.option('--epochs', type=click.INT, default=10, nargs=1, help='Number of training epochs.')
@click.option('--batch_size', type=click.INT, default=32, nargs=1, help='Batch size to use for training.')
@click.option('--rebalance', is_flag=True,
              help='If flag is set, class weighting will be used during training to rebalance '
                   'an uneven distribution of classes in the training set.')
@click.option('--binary', type=click.BOOL, default=True, nargs=1,
              help='If true, the labels will be collapsed to binary values, where any non-zero label will become a 1.')
@click.option('--dropout', type=click.FLOAT, default=0.5, nargs=1, help='Dropout rate.')
@click.option('--lr', type=click.FLOAT, default=0.0001, nargs=1, help='Learning rate to use during training.')
@click.option('--decay', type=click.FLOAT, default=0., nargs=1, help='Learning rate decay to use during training.')
@click.option('--validation_split', type=click.FLOAT, default=0.15, nargs=1,
              help='Percentage of training set to use for validation. Should be in range (0, 1). Defaults to 0.15.')
@click.option('--examples_limit', type=click.INT, default=-1, nargs=1,
              help='Limit on the number of training examples to use during training.')
@click.option('--seed', type=click.INT, default=71, nargs=1, help='Random seed.')
def main(data_dir, log_dir, hidden_layers, prefix, epochs, batch_size, rebalance, binary, dropout, lr, decay,
         validation_split, examples_limit, seed):
    """This script will train a FCNN classifier."""
    assert 0 < validation_split < 1, 'validation_split must be in range (0, 1)'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set random seeds
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Load data
    train, _ = load_discretized_data(data_dir, prefix=prefix, categorical=True, binary=binary)

    num_categories = train[TARGETS].shape[1]

    if examples_limit == -1:
        examples_limit = train[TARGETS].shape[0]

    # Build model
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(hidden_layers[0], input_dim=train[FEATURES].shape[1], activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    for neurons in hidden_layers[1:]:
        model.add(tf.keras.layers.Dense(neurons, input_dim=train[FEATURES].shape[1], activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_categories, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=lr, decay=decay)
    loss = 'binary_crossentropy' if num_categories == 2 else 'categorical_crossentropy'

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])

    os.makedirs(os.path.join(log_dir, 'ckpt'))
    ckpt_path = os.path.join(log_dir, 'ckpt', 'epoch-{epoch:02d}.h5')

    # Get class weights
    if rebalance:
        targets_argmax = np.argmax(train[TARGETS], axis=1)
        class_weight = compute_class_weight('balanced', np.unique(targets_argmax), targets_argmax)
        class_weight = dict(enumerate(class_weight))
    else:
        class_weight = None

    # Setup checkpoint callback
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                       save_weights_only=False,
                                                       period=1,
                                                       save_best_only=False,
                                                       monitor='val_loss')

    # Setup TensorBoard callback
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir)

    # Train the model
    model.fit(train[FEATURES][:examples_limit],
              train[TARGETS][:examples_limit],
              epochs=epochs,
              batch_size=batch_size,
              validation_split=validation_split,
              verbose=1,
              class_weight=class_weight,
              callbacks=[tb_callback, ckpt_callback])


if __name__ == '__main__':
    main()
