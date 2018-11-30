"""Script for training a CNN classifying using the VGG16 architecture with ImageNet weights.
The command line interface provides various options for training. The program expects data that can be loaded
by `utils.data.load_image_h5`.

Author: Ryan Strauss
"""
import json

import click
import numpy as np
import os
import tensorflow as tf

from utils.data import load_image_h5

FEATURES = 0
TARGETS = 1


@click.command()
@click.argument('train_path', type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=1)
@click.argument('save_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), nargs=1)
@click.option('--epochs', type=click.INT, default=10, nargs=1, help='Number of training epochs.')
@click.option('--batch_size', type=click.INT, default=32, nargs=1, help='Batch size to use for training.')
@click.option('--train_combine', is_flag=True,
              help='If flag is set, the training and test sets within the HDF5 file pointed '
                   'to by train_path will be combined into a single training set.')
@click.option('--binary', type=click.BOOL, default=True, nargs=1,
              help='If true, the labeled will be collapsed to binary values, where any non-zero label will become a 1.')
@click.option('--lr', type=click.FLOAT, default=0.00001, nargs=1, help='Learning rate to use during training.')
@click.option('--decay', type=click.FLOAT, default=0., nargs=1, help='Learning rate decay to use during training.')
@click.option('--freeze', is_flag=True,
              help='If flag is set, the convolutional layers of the model will be frozen. Only the '
                   'fully-connected classification layers will have their weights updated.')
@click.option('--examples_limit', type=click.INT, default=30000, nargs=1,
              help='Limit on the number of training examples to use during training.')
def main(train_path, save_dir, epochs, batch_size, train_combine, binary, lr, decay, freeze, examples_limit):
    """This script will train a CNN classifier using the VGG16 architecture with ImageNet weights."""
    assert train_path.endswith('.h5'), 'train_path must point to an HDF5 file'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load data
    if train_combine:
        a, b = load_image_h5(train_path, categorical=True, binary=binary)
        train = np.concatenate([a[FEATURES], b[FEATURES]], axis=0), np.concatenate([a[TARGETS], b[TARGETS]], axis=0)
    else:
        train, _ = load_image_h5(train_path, categorical=True, binary=binary)

    num_categories = train[TARGETS].shape[1]

    # Build model
    vgg16_base = tf.keras.applications.VGG16(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
    net = vgg16_base.output
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(256, activation=tf.nn.relu)(net)
    net = tf.keras.layers.Dropout(0.5)(net)
    preds = tf.keras.layers.Dense(num_categories, activation=tf.nn.softmax)(net)
    model = tf.keras.Model(vgg16_base.input, preds)

    # Freeze convolutional layers if needed
    if freeze:
        for layer in model.layers[:-4]:
            layer.trainable = False

    opt = tf.keras.optimizers.Adam(lr=lr, decay=decay)

    model.compile(loss='binary_crossentropy' if num_categories == 2 else 'categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    ckpt_path = os.path.join(save_dir, 'ckpt', 'weights.{epoch:02d}-{val_loss:.2f}')

    # Setup checkpoint callback
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                       save_weights_only=True,
                                                       period=2,
                                                       save_best_only=False,
                                                       monitor='val_loss')
    # Train the model
    history = model.fit(train[FEATURES][:examples_limit],
                        train[TARGETS][:examples_limit],
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.15,
                        verbose=1,
                        callbacks=[ckpt_callback])

    # Save the training history
    hist_json = json.dumps(history.history)
    hist_path = os.path.join(save_dir, 'history.json')
    with open(hist_path, 'w') as file:
        file.write(hist_json)

    # Save the model
    model_path = os.path.join(save_dir, 'model.json')
    with open(model_path, 'w') as file:
        file.write(model.to_json())


if __name__ == '__main__':
    main()
