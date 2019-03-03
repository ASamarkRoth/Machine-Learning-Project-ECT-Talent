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


def encoder(x, n_filters=32, k_size=3, normalizer_fn=instance_norm,
            activation_fn=tf.nn.leaky_relu, scope=None, reuse=None):
    with tf.variable_scope(scope or 'encoder', reuse=reuse):
        h = tf.pad(x, [[0, 0], [k_size, k_size], [k_size, k_size], [0, 0]],
                   "REFLECT")
        h = layers.conv2d(
            inputs=h,
            num_outputs=n_filters,
            kernel_size=7,
            stride=1,
            padding='VALID',
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=None,
            normalizer_fn=normalizer_fn,
            activation_fn=activation_fn,
            scope='1',
            reuse=reuse)
        h = layers.conv2d(
            inputs=h,
            num_outputs=n_filters * 2,
            kernel_size=k_size,
            stride=2,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=None,
            normalizer_fn=normalizer_fn,
            activation_fn=activation_fn,
            scope='2',
            reuse=reuse)
        h = layers.conv2d(
            inputs=h,
            num_outputs=n_filters * 4,
            kernel_size=k_size,
            stride=2,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=None,
            normalizer_fn=normalizer_fn,
            activation_fn=activation_fn,
            scope='3',
            reuse=reuse)
    return h


def residual_block(x, n_channels=128, normalizer_fn=instance_norm,
                   activation_fn=tf.nn.leaky_relu, kernel_size=3, scope=None, reuse=None):
    with tf.variable_scope(scope or 'residual', reuse=reuse):
        h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        h = layers.conv2d(
            inputs=h,
            num_outputs=n_channels,
            kernel_size=kernel_size,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=None,
            normalizer_fn=normalizer_fn,
            padding='VALID',
            activation_fn=activation_fn,
            scope='1',
            reuse=reuse)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        h = layers.conv2d(
            inputs=h,
            num_outputs=n_channels,
            kernel_size=kernel_size,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=None,
            normalizer_fn=normalizer_fn,
            padding='VALID',
            activation_fn=None,
            scope='2',
            reuse=reuse)
        h = tf.add(x, h)
    return h


def transform(x, img_size=256, reuse=None):
    h = x
    if img_size >= 256:
        n_blocks = 9
    else:
        n_blocks = 6
    for block_i in range(n_blocks):
        with tf.variable_scope('block_{}'.format(block_i), reuse=reuse):
            h = residual_block(h, reuse=reuse)
    return h


def decoder(x, n_filters=32, k_size=3, normalizer_fn=instance_norm,
            activation_fn=tf.nn.leaky_relu, scope=None, reuse=None):
    with tf.variable_scope(scope or 'decoder', reuse=reuse):
        h = layers.conv2d_transpose(
            inputs=x,
            num_outputs=n_filters * 2,
            kernel_size=k_size,
            stride=2,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=None,
            normalizer_fn=normalizer_fn,
            activation_fn=activation_fn,
            scope='1',
            reuse=reuse)
        h = layers.conv2d_transpose(
            inputs=h,
            num_outputs=n_filters,
            kernel_size=k_size,
            stride=2,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=None,
            normalizer_fn=normalizer_fn,
            activation_fn=activation_fn,
            scope='2',
            reuse=reuse)
        h = tf.pad(h, [[0, 0], [k_size, k_size], [k_size, k_size], [0, 0]],
                   "REFLECT")
        h = layers.conv2d(
            inputs=h,
            num_outputs=1,
            kernel_size=7,
            stride=1,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=None,
            padding='VALID',
            normalizer_fn=normalizer_fn,
            activation_fn=tf.nn.tanh,
            scope='3',
            reuse=reuse)
    return h


def cyclegan_resnet_generator(x, scope=None, reuse=None):
    img_size = x.get_shape().as_list()[1]
    with tf.variable_scope(scope or 'generator', reuse=reuse):
        h = encoder(x, reuse=reuse)
        h = transform(h, img_size, reuse=reuse)
        h = decoder(h, reuse=reuse)
    return h


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
