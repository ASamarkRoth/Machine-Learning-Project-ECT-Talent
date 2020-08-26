"""Script for training a FCNN classifier.

Author: Ryan Strauss
"""

import datetime
import json
import sys

import click
import numpy as np
import os
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from data_processing.data import load_discretized_data

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
def main(data_dir, log_dir, hidden_layers, prefix, epochs, batch_size, rebalance, binary, dropout, lr, decay, validation_split, examples_limit, seed):
    train(data_dir, log_dir, hidden_layers, prefix, epochs, batch_size, rebalance, binary, dropout, lr, decay, validation_split, examples_limit, seed)

def train(train, log_dir, hidden_layers, prefix='', epochs=10, batch_size=32, rebalance=False, binary=True, use_dropout=False, dropout=0.5, lr=1e-4, decay=0., validation_split=0.15, examples_limit=-1, seed=71):
    """This script will train a FCNN classifier."""
    assert 0 < validation_split < 1, 'validation_split must be in range (0, 1)'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load data
    #train, _ = load_discretized_data(data_dir, prefix=prefix, categorical=True, binary=binary)

    #num_categories = train[TARGETS].shape[1]
    num_categories = 1

    # Build model
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(hidden_layers[0], input_dim=train[FEATURES].shape[1], activation='relu'))
    if use_dropout:
        model.add(tf.keras.layers.Dropout(dropout))
    for neurons in hidden_layers[1:]:
        model.add(tf.keras.layers.Dense(neurons, input_dim=train[FEATURES].shape[1], activation='relu'))
        if use_dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_categories, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(lr=lr, decay=decay)
    loss = 'binary_crossentropy'# if num_categories == 2 else 'categorical_crossentropy'

    #print("Loss:", loss)

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])

    print(model.summary())
    #os.makedirs(os.path.join(log_dir, 'ckpt'), exist_ok=True)
    #ckpt_path = os.path.join(log_dir, 'ckpt', 'epoch-{epoch:02d}.h5')
    
    log_run = "nodes{}_dropout{}_lr{}_decay{}_samples{}".format(hidden_layers[0], use_dropout, lr, decay, examples_limit)
    #print("Log for run:", log_run)
    
    log_dir = os.path.join(log_dir, log_run, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print("\nWriting fits to:", log_dir)
    ckpt_path = os.path.join(log_dir, 'epoch-{epoch:02d}.h5')
    print("Checkpoint path:", ckpt_path)

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
                                                       save_frequency=1,
                                                       save_best_only=False,
                                                       monitor='val_loss')

    # Setup TensorBoard callback
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir)

    if examples_limit == -1:
        examples_limit = train[TARGETS].shape[0]

    if examples_limit > train[TARGETS].shape[0]:
        warnings.warn('`examples_limit` is larger than the number of examples in the training set. The entire training '
                      'set will be used.')
        examples_limit = train[TARGETS].shape[0]

    # Train the model
    train_start_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

    history = model.fit(train[FEATURES][:examples_limit],
                        train[TARGETS][:examples_limit],
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=1,
                        class_weight=class_weight,
                        callbacks=[tb_callback, ckpt_callback])

    train_end_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

    history_filename = os.path.join(log_dir, 'history.json')
    info_filename = os.path.join(log_dir, 'info.txt')
    
    model_filename = os.path.join(log_dir, 'saved_model.h5')
    model.save(model_filename)

    with open(history_filename, 'w') as file:
        json.dump(history.history, file)

    with open(info_filename, 'w') as file:
        file.write('***Training Info***\n')
        file.write('Training Start: {}'.format(train_start_time))
        file.write('Training End: {}\n'.format(train_end_time))
        file.write('Arguments:\n')
        for arg in sys.argv:
            file.write('\t{}\n'.format(arg))


if __name__ == '__main__':
    main()
