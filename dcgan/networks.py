"""Networks for Deep Convolutional Generative Adversarial Network.

This architecture mostly follows the one described in Radford et. al. "Unsupervised Representation
Learning with Deep Convolutional Generative Adversarial Networks" (https://arxiv.org/abs/1511.06434).

However, one noted difference is the lack of batch normalization in the discriminator.

Author: Ryan Strauss
"""
import tensorflow as tf

layers = tf.contrib.layers


def generator(noise, is_training=True):
    """Core DCGAN generator.

    Args:
        noise: A 2D Tensor of shape [batch size, noise dim].
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch norm uses the exponential
        moving average collected from population statistics.

    Returns:
        A generated image in the range [-1, 1].
    """
    with tf.contrib.framework.arg_scope(
            [layers.conv2d_transpose, layers.fully_connected],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm):
        with tf.contrib.framework.arg_scope([layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(noise, 4 * 4 * 1024)
            net = tf.reshape(net, shape=(-1, 4, 4, 1024))
            net = layers.conv2d_transpose(net, 256, [5, 5], stride=2)
            net = layers.conv2d_transpose(net, 128, [5, 5], stride=2)
            net = layers.conv2d_transpose(net, 64, [5, 5], stride=2)
            net = layers.conv2d_transpose(net, 32, [5, 5], stride=2)
            net = layers.conv2d_transpose(net, 1, [5, 5], stride=2, activation_fn=tf.nn.tanh, normalizer_fn=None)

    return net


def discriminator(img, unused_conditioning):
    """Core DCGAN discriminator.

    Args:
        img: Real or generated images. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which would require extra `condition`
        information to both the generator and the discriminator. Since this GAN is not conditional, we do not
        use this argument.

    Returns:
        Logits for the probability that the image is real.
    """
    with tf.contrib.framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=tf.nn.leaky_relu, normalizer_fn=None):
        net = layers.conv2d(img, 64, [5, 5], stride=2)
        net = layers.conv2d(net, 128, [5, 5], stride=2)
        net = layers.conv2d(net, 128, [5, 5], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024)
        net = layers.fully_connected(net, 1, activation_fn=None)

    return net
