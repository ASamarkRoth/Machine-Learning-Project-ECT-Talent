"""Defines generator and discriminator networks for CycleGAN.

This implementation of the CycleGAN network architecture is adapted
from the implementation in Tensorflow's research repository.
https://github.com/tensorflow/models/tree/master/research/

Author: Ryan Strauss
"""
import tensorflow as tf

layers = tf.contrib.layers

INSTANCE_NORM_PARAMS = {
    'center': True,
    'scale': True,
    'epsilon': 0.001,
}


def conv2d(inputs, num_filters, kernel_size, stride=1, padding='SAME', scope=None, biases_initializer=None,
           activation_fn=tf.nn.leaky_relu, weights_initializer='truncated', normalizer_fn=layers.instance_norm):
    if weights_initializer == 'truncated':
        weights_initializer = tf.truncated_normal_initializer(0, 0.02)
    return layers.conv2d(inputs=inputs, num_outputs=num_filters, kernel_size=kernel_size,
                         stride=stride, padding=padding, scope=scope,
                         biases_initializer=biases_initializer, activation_fn=activation_fn,
                         normalizer_fn=normalizer_fn, normalizer_params=INSTANCE_NORM_PARAMS,
                         weights_initializer=weights_initializer)


def conv2d_transpose(inputs, num_filters, kernel_size, stride=1, padding='SAME', scope=None, biases_initializer=None,
                     activation_fn=tf.nn.leaky_relu):
    weights_initializer = tf.truncated_normal_initializer(0, 0.02)
    return layers.conv2d_transpose(inputs=inputs, num_outputs=num_filters, kernel_size=kernel_size,
                                   stride=stride, padding=padding, scope=scope,
                                   biases_initializer=biases_initializer, activation_fn=activation_fn,
                                   normalizer_fn=layers.instance_norm, normalizer_params=INSTANCE_NORM_PARAMS,
                                   weights_initializer=weights_initializer)


def cyclegan_resnet_generator(image, num_filters=32, kernel_size=3):
    """Generator network for CycleGAN.

    Args:
        image: Input image to be transformed.
        num_filters: Base number of convolutional filters.
        kernel_size: Convolutional kernel size.

    Returns:
        The transformed image.
    """
    img_size = image.get_shape().as_list()[1]

    # ENCODER
    with tf.variable_scope('encoder'):
        net = tf.pad(image, [[0, 0], [kernel_size, kernel_size], [kernel_size, kernel_size], [0, 0]], 'REFLECT')
        net = conv2d(net, num_filters, 7, stride=1, padding='VALID', scope='conv0')
        net = conv2d(net, num_filters * 2, kernel_size, stride=2, scope='conv1')
        net = conv2d(net, num_filters * 4, kernel_size, stride=2, scope='conv2')

    # RESIDUAL BLOCKS
    for i in range(9 if img_size >= 256 else 6):
        with tf.variable_scope('residual_block_{}'.format(i)):
            r = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            r = conv2d(r, num_filters * 4, kernel_size, padding='VALID', scope='conv0')
            r = tf.pad(r, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            r = conv2d(r, num_filters * 4, kernel_size, padding='VALID', activation_fn=None, scope='conv1')
            net += r

    # DECODER
    with tf.variable_scope('decoder'):
        net = conv2d_transpose(net, num_filters * 2, kernel_size, stride=2, scope='conv0')
        net = conv2d_transpose(net, num_filters, kernel_size, stride=2, scope='conv1')
        net = tf.pad(net, [[0, 0], [kernel_size, kernel_size], [kernel_size, kernel_size], [0, 0]], 'REFLECT')
        net = conv2d(net, 1, 7, stride=1, padding='VALID', activation_fn=tf.nn.tanh, scope='conv2')

    return net


def generator(x):
    """A thin wrapper around the CycleGAN Resnet Generator."""
    return cyclegan_resnet_generator(x)


def pix2pix_discriminator(net, num_filters, padding=2, is_training=False):
    """Creates an image to image translation discriminator.

    Args:
        net: A `Tensor` of size [batch_size, height, width, channels] representing
        the input.
        num_filters: A list of the filters in the discriminator. The length of the
        list determines the number of layers in the discriminator.
        padding: Amount of reflection padding applied before each convolution.
        is_training: Whether or not the model is training.

    Returns:
        A logits `Tensor` of size [batch_size, N, N, 1] where N is the number of
        'patches' we're attempting to discriminate and a dictionary of model end
        points.
    """
    del is_training

    num_layers = len(num_filters)

    # No normalization on the input layer.
    net = tf.pad(net, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')
    net = conv2d(net, num_filters[0], 4, stride=2, normalizer_fn=None, padding='VALID', scope='conv0')

    for i in range(1, num_layers - 1):
        net = tf.pad(net, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')
        net = conv2d(net, num_filters[i], 4, stride=2, padding='VALID', scope='conv{}'.format(i))

    # Stride 1 on the last layer.
    net = tf.pad(net, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')
    net = conv2d(net, num_filters[-1], 4, stride=1, padding='VALID', scope='conv{}'.format(num_layers - 1))

    # 1-dim logits, stride 1, no normalization.
    net = tf.pad(net, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')
    logits = conv2d(net, 1, 4, stride=1, activation_fn=tf.nn.sigmoid, normalizer_fn=None, padding='VALID',
                    scope='conv{}'.format(num_layers))

    return logits


def discriminator(image_batch, unused_conditioning=None):
    """A thin wrapper around the discriminator to conform to TFGAN API."""
    return pix2pix_discriminator(image_batch, [32, 64, 64])
