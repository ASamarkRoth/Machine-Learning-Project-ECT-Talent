"""Defines generator and discriminator networks for CycleGAN.

This implementation of the CycleGAN network architecture is adapted
from the implementation in Tensorflow's research repository.
https://github.com/tensorflow/models/tree/master/research/

Author: Ryan Strauss
"""
import tensorflow as tf

layers = tf.contrib.layers

_instance_norm_params = {
    'center': True,
    'scale': True,
    'epsilon': 1e-5,
}


def _cyclegan_resnet_generator(image, num_filters=32, kernel_size=3):
    img_size = image.get_shape().as_list()[1]
    with tf.contrib.framework.arg_scope(
            [layers.conv2d, layers.conv2d_transpose],
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02), biases_initializer=None,
            normalizer_fn=layers.instance_norm, normalizer_params=_instance_norm_params, activation_fn=tf.nn.leaky_relu,
            kernel_size=kernel_size, reuse=None):
        # ENCODER
        with tf.variable_scope('encoder', reuse=None):
            net = tf.pad(image, [[0, 0], [kernel_size, kernel_size], [kernel_size, kernel_size], [0, 0]], 'REFLECT')
            net = layers.conv2d(net, num_filters, kernel_size=7, stride=1, padding='VALID', scope='conv0')
            net = layers.conv2d(net, num_filters * 2, stride=2, scope='conv1')
            net = layers.conv2d(net, num_filters * 4, stride=2, scope='conv2')

        # TRANSFORM
        for block in range(6 if img_size < 256 else 9):
            with tf.variable_scope('residual_block_{}'.format(block), reuse=None):
                r = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
                r = layers.conv2d(r, num_filters * 4, padding='VALID', scope='conv0')
                r = tf.pad(r, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
                r = layers.conv2d(r, num_filters * 4, padding='VALID', activation_fn=None, scope='conv1')
                net = tf.add(r, net)

        # DECODER
        with tf.variable_scope('decoder', reuse=None):
            net = layers.conv2d_transpose(net, num_filters * 2, stride=2, scope='conv0')
            net = layers.conv2d_transpose(net, num_filters, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [kernel_size, kernel_size], [kernel_size, kernel_size], [0, 0]], 'REFLECT')
            net = layers.conv2d(net, 1, kernel_size=7, stride=1, padding='VALID', activation_fn=tf.nn.tanh,
                                scope='conv2')

    return net


def generator(x):
    """A thin wrapper around the CycleGAN Resnet Generator."""
    return _cyclegan_resnet_generator(x)


def _pix2pix_discriminator(net, num_filters, padding=2, is_training=False):
    """Creates the Image2Image Translation Discriminator.

    Args:
        net: A `Tensor` of size [batch_size, height, width, channels] representing
        the input.
        num_filters: A list of the filters in the discriminator. The length of the
        list determines the number of layers in the discriminator.
        padding: Amount of reflection padding applied before each convolution.
        is_training: Whether or not the model is training or testing.

    Returns:
        A logits `Tensor` of size [batch_size, N, N, 1] where N is the number of
        'patches' we're attempting to discriminate and a dictionary of model end
        points.
    """
    del is_training

    num_layers = len(num_filters)

    with tf.contrib.framework.arg_scope(
            [layers.conv2d],
            kernel_size=[4, 4], stride=2, padding='valid', activation_fn=tf.nn.leaky_relu,
            normalizer_fn=None):
        # No normalization on the input layer.
        net = tf.pad(net, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')
        net = layers.conv2d(net, num_filters[0], normalizer_fn=None, scope='conv0')

        for i in range(1, num_layers - 1):
            net = tf.pad(net, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')
            net = layers.conv2d(net, num_filters[i], scope='conv{}'.format(i))

        # Stride 1 on the last layer.
        net = tf.pad(net, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')
        net = layers.conv2d(net, num_filters[-1], stride=1, scope='conv{}'.format(num_layers - 1))

        # 1-dim logits, stride 1, no activation, no normalization.
        net = tf.pad(net, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')
        logits = layers.conv2d(net, 1, stride=1, activation_fn=None, normalizer_fn=None,
                               scope='conv{}'.format(num_layers))

    return tf.sigmoid(logits)


def discriminator(image_batch, unused_conditioning=None):
    """A thin wrapper around the discriminator to conform to TFGAN API."""
    return _pix2pix_discriminator(image_batch, [32, 64, 64])
