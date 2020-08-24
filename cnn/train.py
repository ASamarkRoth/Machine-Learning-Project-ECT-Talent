"""Script for training a CNN classifier using the VGG16 architecture with ImageNet weights.
The command line interface provides various options for training. The program expects data that can be loaded
by `utils.data.load_image_h5`.

Author: Ryan Strauss
"""
import datetime
import json
import sys

import click
import numpy as np
import os
import tensorflow as tf
import warnings
from sklearn.utils.class_weight import compute_class_weight
import sklearn.preprocessing

from utils.data import load_image_h5

FEATURES = 0
TARGETS = 1


@click.command()
@click.argument('data', type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=1)
@click.argument('log_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), nargs=1)
@click.option('--epochs', type=click.INT, default=10, nargs=1, help='Number of training epochs.')
@click.option('--batch_size', type=click.INT, default=32, nargs=1, help='Batch size to use for training.')
@click.option('--data_combine', is_flag=True,
              help='If flag is set, the training and test sets within the HDF5 file pointed '
                   'to by `data` will be combined into a single training set.')
@click.option('--rebalance', is_flag=True,
              help='If flag is set, class weighting will be used during training to rebalance '
                   'an uneven distribution of classes in the training set.')
@click.option('--binary', type=click.BOOL, default=True, nargs=1,
              help='If true, the labels will be collapsed to binary values, where any non-zero label will become a 1.')
@click.option('--lr', type=click.FLOAT, default=0.00001, nargs=1, help='Learning rate to use during training.')
@click.option('--decay', type=click.FLOAT, default=0., nargs=1, help='Learning rate decay to use during training.')
@click.option('--validation_split', type=click.FLOAT, default=0.15, nargs=1,
              help='Percentage of training set to use for validation. Should be in range (0, 1). Defaults to 0.15.')
@click.option('--freeze', is_flag=True,
              help='If flag is set, the convolutional layers of the model will be frozen. Only the '
                   'fully-connected classification layers will have their weights updated.')
@click.option('--examples_limit', type=click.INT, default=-1, nargs=1,
              help='Limit on the number of training examples to use during training.')
@click.option('--seed', type=click.INT, default=71, nargs=1, help='Random seed.')
@click.option('--reverse_labels', is_flag=True, help='If flag is set, labels will be reversed.')
@click.option('--validation_size', type=click.INT, default=None, nargs=1,
              help='If None, a random 15% of the training data will be selected for validation. Otherwise, the '
                   'the last `validation_size` examples from the training set will be used. This will '
                   'override `validation_split`.')

def main(data, log_dir, epochs, batch_size, data_combine, rebalance, binary, lr, decay, validation_split, freeze, examples_limit, seed, reverse_labels, validation_size):
    train(data, log_dir, epochs, batch_size, data_combine, rebalance, binary, lr, decay, validation_split, freeze, examples_limit, seed, reverse_labels, validation_size)


def train(data, log_dir, epochs=10, batch_size=32, data_combine=False, rebalance=False, binary=False, lr=0.00001, decay=0., validation_split=0.15, freeze=False,
         examples_limit=-1, seed=71, reverse_labels=True, validation_size=None):
    """This function will train a CNN classifier using the VGG16 architecture with ImageNet weights."""
    assert data.endswith('.h5'), 'train_path must point to an HDF5 file'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load data
    if data_combine:
        a, b = load_image_h5(data, categorical=True, binary=binary, reverse_labels=reverse_labels)
        train = np.concatenate([a[FEATURES], b[FEATURES]], axis=0), np.concatenate([a[TARGETS], b[TARGETS]], axis=0)
    else:
        train, _ = load_image_h5(data, categorical=False, binary=binary, reverse_labels=reverse_labels)
        
    #train = sklearn.preprocessing.StandardScaler().fit_transform(train)

    print("TARGETS shape:", len(train),train[TARGETS].shape)
    print("FEATURES shape:", len(train),train[FEATURES].shape, train[FEATURES].shape[1:])
    #num_categories = train[TARGETS].shape[1]
    num_categories = 1


    # Build model
    vgg16_base = tf.keras.applications.VGG16(include_top=False, input_shape=train[FEATURES].shape[1:], weights='imagenet')
    net = vgg16_base.output
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(256, activation=tf.nn.relu)(net)
    net = tf.keras.layers.Dropout(0.5)(net)
    preds = tf.keras.layers.Dense(num_categories, activation=tf.nn.sigmoid)(net) #should use softmax?
    model = tf.keras.Model(vgg16_base.input, preds)

    # Freeze convolutional layers if needed
    if freeze:
        for layer in model.layers[:-4]:
            layer.trainable = False

    opt = tf.keras.optimizers.Adam(lr=lr, decay=decay)
    loss = 'binary_crossentropy'# if num_categories == 2 else 'categorical_crossentropy'

    print("Loss:", loss)

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])

    os.makedirs(os.path.join(log_dir, 'ckpt'), exist_ok=True)
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
                                                       save_frequency=1,
                                                       save_best_only=False,
                                                       monitor='val_loss')

    # Setup TensorBoard callback
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir)

    val = None

    if validation_size is not None:
        if validation_size >= train[TARGETS].shape[0]:
            raise ValueError('The given validation size must be smaller than the size of the training set ({}).'.format(
                train[TARGETS].shape[0]))
        val = train[FEATURES][-validation_size:], train[TARGETS][-validation_size:]
        train = train[FEATURES][:-validation_size], train[TARGETS][:-validation_size]

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
                        validation_data=val,
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
