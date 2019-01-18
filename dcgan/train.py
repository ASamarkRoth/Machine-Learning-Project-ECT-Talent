"""Trains a DCGAN to learn the distribution of 2D projections of events.

The network that gets trained is an adaptation of the DCGAN (https://arxiv.org/abs/1511.06434)
that uses the Wasserstein loss (https://arxiv.org/abs/1701.07875).

Author: Ryan Strauss
"""
import click
import numpy as np
import os
import tensorflow as tf

import networks
from utils.data import data_to_stream, load_image_h5

tf.logging.set_verbosity(tf.logging.INFO)

tfgan = tf.contrib.gan


@click.command()
@click.argument('data_path', type=click.Path(file_okay=True, dir_okay=False, exists=True), nargs=1)
@click.argument('log_dir', type=click.Path(file_okay=False, dir_okay=True, exists=False), nargs=1)
@click.option('--batch_size', type=click.INT, default=32, nargs=1, help='Batch size to be used for training.')
@click.option('--latent_dim', type=click.INT, default=64, nargs=1, help='Dimension of the generator\'s input space.')
@click.option('--steps', type=click.INT, default=60000, nargs=1, help='Number of training steps.')
@click.option('--examples_limit', type=click.INT, default=98000, nargs=1,
              help='Maximum number of training examples to use.')
@click.option('--gradient_penalty', type=click.FLOAT, default=1.0, nargs=1,
              help='Indicates how much to weight the gradient penalty.')
@click.option('--generator_lr', type=click.FLOAT, default=0.001, nargs=1, help='Generator learning rate.')
@click.option('--discriminator_lr', type=click.FLOAT, default=0.0001, nargs=1, help='Discriminator learning rate.')
@click.option('--checkpoint_freq', type=click.INT, default=600, nargs=1,
              help='Frequency, in seconds, that model weights are saved during training.')
def main(data_path, batch_size, latent_dim, steps, examples_limit, log_dir, gradient_penalty, generator_lr,
         discriminator_lr, checkpoint_freq):
    """Trains a DCGAN to generate CNN training images."""
    # Load data
    real = load_image_h5(data_path)

    # Process real data
    real_data = np.expand_dims(real[:, :, :, 0], 3)
    real_data = (real_data.astype('float32') - 127.5) / 127.5
    real_data = real_data[:examples_limit]

    # Create data iterator
    data_iterator, iterator_init_hook = data_to_stream(real_data, batch_size)

    # Configure GAN model
    gan_model = tfgan.gan_model(
        generator_fn=networks.generator,
        discriminator_fn=networks.discriminator,
        real_data=data_iterator.get_next(),
        generator_inputs=tf.random_normal([batch_size, latent_dim]))

    tfgan.eval.add_gan_model_image_summaries(gan_model)
    tfgan.eval.add_regularization_loss_summaries(gan_model)

    # Set up loss functions
    gan_loss = tfgan.gan_loss(
        gan_model,
        gradient_penalty_weight=gradient_penalty,
        add_summaries=True)

    # Configure training ops
    generator_opt = tf.train.AdamOptimizer(generator_lr, beta1=0.5)
    discriminator_opt = tf.train.AdamOptimizer(discriminator_lr, beta1=0.5)
    train_ops = tfgan.gan_train_ops(
        gan_model, gan_loss,
        generator_optimizer=generator_opt,
        discriminator_optimizer=discriminator_opt,
        summarize_gradients=True)

    status_message = tf.string_join(['Starting train step: ', tf.as_string(tf.train.get_or_create_global_step())],
                                    name='status_message')

    if os.path.exists(log_dir):
        print('Log directory already exists. Exiting to avoid overwriting other model.')
        exit(0)

    # Begin training
    tfgan.gan_train(
        train_ops,
        logdir=log_dir,
        save_checkpoint_secs=checkpoint_freq,
        hooks=[tf.train.StopAtStepHook(num_steps=steps),
               tf.train.LoggingTensorHook([status_message], every_n_iter=100),
               iterator_init_hook])


if __name__ == '__main__':
    main()
