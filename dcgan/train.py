"""Trains a DCGAN to learn the distribution of 2D projections of events.

The network that gets trained is an adaptation of the DCGAN (https://arxiv.org/abs/1511.06434)
that uses the Wasserstein loss (https://arxiv.org/abs/1701.07875).

Author: Ryan Strauss
"""
from datetime import datetime

import numpy as np
import os
import tensorflow as tf
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter

import networks
from utils.data import data_to_stream, load_image_h5

ex = Experiment('dcgan_event_generation')
ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='attpc-event-generation'))

tf.logging.set_verbosity(tf.logging.INFO)
tfgan = tf.contrib.gan


@ex.config
def config():
    data_path = None
    logdir = None

    batch_size = 32
    latent_dim = 64
    steps = 100000
    gradient_penalty_weight = 1.
    generator_lr = 0.001
    discriminator_lr = 0.001

    checkpoint_freq = 600
    examples_limit = np.inf


@ex.config_hook
def config_hook(config, command_name, logger):
    if command_name == 'main':
        if config['data_path'] is None:
            logger.error('Path to data must be provided.')
            exit(1)
        if config['logdir'] is None:
            logger.error('A log directory must be provided.')
            exit(1)

    return config


@ex.automain
@LogFileWriter(ex)
def main(data_path, batch_size, latent_dim, steps, examples_limit, gradient_penalty_weight, generator_lr,
         discriminator_lr, checkpoint_freq, logdir, seed, _log):
    """Trains a DCGAN to generate CNN training images."""
    tf.set_random_seed(seed)

    logdir = os.path.join(logdir, ex.path, datetime.now().strftime('%Y%m%d%H%M%S'))
    _log.info('Logs will be saved to: {}'.format(logdir))

    # Load data
    real = load_image_h5(data_path)

    # Process real data
    real_data = np.expand_dims(real[:, :, :, 0], 3)
    real_data = (real_data.astype('float32') - 127.5) / 127.5
    examples_limit = min(examples_limit, len(real_data))
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
        gradient_penalty_weight=gradient_penalty_weight,
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

    if os.path.exists(logdir):
        _log.warning('Log directory already exists. Exiting to avoid overwriting other model.')
        exit(0)

    # Begin training
    tfgan.gan_train(
        train_ops,
        logdir=logdir,
        save_checkpoint_secs=checkpoint_freq,
        hooks=[tf.train.StopAtStepHook(num_steps=steps),
               tf.train.LoggingTensorHook([status_message], every_n_iter=100),
               iterator_init_hook])
