"""Defines generator and discriminator networks for CycleGAN.

This implementation of the CycleGAN network architecture is adapted
from the implementation in Tensorflow's research repository.
https://github.com/tensorflow/models/tree/master/research/

Author: Ryan Strauss
"""
import tensorflow as tf

layers = tf.contrib.layers


def instance_norm(x, epsilon=1e-5):
    """Instance Normalization.
    See Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016).
    Instance Normalization: The Missing Ingredient for Fast Stylization,
    Retrieved from http://arxiv.org/abs/1607.08022
    Parameters
    ----------
    x : TYPE
        Description
    epsilon : float, optional
        Description
    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope('instance_norm'):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable(
            name='scale',
            shape=[x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02),
            dtype=tf.float64)
        offset = tf.get_variable(
            name='offset',
            shape=[x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0),
            dtype=tf.float64)
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
        return out


def cyclegan_resnet_generator(image, num_filters=32, kernel_size=3):
    img_size = image.get_shape().as_list()[1]
    with tf.variable_scope('generator', reuse=None):
        with tf.contrib.framework.arg_scope(
                [layers.conv2d, layers.conv2d_transpose],
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02), biases_initializer=None,
                normalizer_fn=instance_norm, activation_fn=tf.nn.leaky_relu, kernel_size=kernel_size, reuse=None):
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
    return cyclegan_resnet_generator(x)


def pix2pix_discriminator(net, num_filters, padding=2, is_training=False):
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

    def padded(net, scope):
        if padding:
            with tf.variable_scope(scope):
                spatial_pad = tf.constant(
                    [[0, 0], [padding, padding], [padding, padding], [0, 0]],
                    dtype=tf.int32)
                return tf.pad(net, spatial_pad, 'REFLECT')
        else:
            return net

    with tf.contrib.framework.arg_scope(
            [layers.conv2d],
            kernel_size=[4, 4],
            stride=2,
            padding='valid',
            activation_fn=tf.nn.leaky_relu):

        # No normalization on the input layer.
        net = layers.conv2d(
            padded(net, 'conv0'), num_filters[0], normalizer_fn=None, scope='conv0')

        for i in range(1, num_layers - 1):
            net = layers.conv2d(
                padded(net, 'conv%d' % i), num_filters[i], scope='conv%d' % i)

        # Stride 1 on the last layer.
        net = layers.conv2d(
            padded(net, 'conv%d' % (num_layers - 1)),
            num_filters[-1],
            stride=1,
            scope='conv%d' % (num_layers - 1))

        # 1-dim logits, stride 1, no activation, no normalization.
        logits = layers.conv2d(
            padded(net, 'conv%d' % num_layers),
            1,
            stride=1,
            activation_fn=None,
            normalizer_fn=None,
            scope='conv%d' % num_layers)

    return tf.sigmoid(logits)


def discriminator(image_batch, unused_conditioning=None):
    """A thin wrapper around the discriminator to conform to TFGAN API."""
    return pix2pix_discriminator(image_batch, [32, 64, 64])
