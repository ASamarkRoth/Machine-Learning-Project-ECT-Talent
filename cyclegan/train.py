"""Trains a CycleGAN model to transfer between simulated and real event domains.

Author: Ryan Strauss
"""

import click
import numpy as np
import tensorflow as tf

import networks_cleaned as networks
from utils.data import data_to_stream, load_image_h5

tf.logging.set_verbosity(tf.logging.INFO)

tfgan = tf.contrib.gan


def _define_model(images_x, images_y):
    """Defines a CycleGAN model that maps between images_x and images_y.

    Args:
        images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
        images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.

    Returns:
        A `CycleGANModel` namedtuple.
    """
    cyclegan_model = tfgan.cyclegan_model(
        generator_fn=networks.generator,
        discriminator_fn=networks.discriminator,
        data_x=images_x,
        data_y=images_y)

    # Add summaries for generated images.
    tfgan.eval.add_cyclegan_image_summaries(cyclegan_model)

    return cyclegan_model


def _get_lr(base_lr, steps):
    """Returns a learning rate `Tensor`.

    Args:
        base_lr: A scalar float `Tensor` or a Python number.  The base learning
        rate.
        steps: number of training steps

    Returns:
        A scalar float `Tensor` of learning rate which equals `base_lr` when the
        global training step is less than FLAGS.max_number_of_steps / 2, afterwards
        it linearly decays to zero.
    """
    global_step = tf.train.get_or_create_global_step()
    lr_constant_steps = steps // 2

    def _lr_decay():
        return tf.train.polynomial_decay(
            learning_rate=base_lr,
            global_step=(global_step - lr_constant_steps),
            decay_steps=(steps - lr_constant_steps),
            end_learning_rate=0.0)

    return tf.cond(global_step < lr_constant_steps, lambda: base_lr, _lr_decay)


def _get_optimizer(gen_lr, dis_lr):
    """Returns generator optimizer and discriminator optimizer.

    Args:
        gen_lr: A scalar float `Tensor` or a Python number.  The Generator learning
        rate.
        dis_lr: A scalar float `Tensor` or a Python number.  The Discriminator
        learning rate.

    Returns:
        A tuple of generator optimizer and discriminator optimizer.
    """
    gen_opt = tf.train.AdamOptimizer(gen_lr, beta1=0.5, use_locking=True)
    dis_opt = tf.train.AdamOptimizer(dis_lr, beta1=0.5, use_locking=True)
    return gen_opt, dis_opt


def _define_train_ops(cyclegan_model, cyclegan_loss, generator_lr, discriminator_lr, steps):
    """Defines train ops that trains `cyclegan_model` with `cyclegan_loss`.

    Args:
        cyclegan_model: A `CycleGANModel` namedtuple.
        cyclegan_loss: A `CycleGANLoss` namedtuple containing all losses for
        `cyclegan_model`.
        generator_lr: Learning rate for the generator.
        discriminator_lr: Learning rate for the discriminator.
        steps: number of training steps

    Returns:
        A `GANTrainOps` namedtuple.
    """
    gen_lr = _get_lr(generator_lr, steps)
    dis_lr = _get_lr(discriminator_lr, steps)
    gen_opt, dis_opt = _get_optimizer(gen_lr, dis_lr)
    train_ops = tfgan.gan_train_ops(
        cyclegan_model,
        cyclegan_loss,
        generator_optimizer=gen_opt,
        discriminator_optimizer=dis_opt,
        summarize_gradients=True,
        colocate_gradients_with_ops=True,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    tf.summary.scalar('generator_lr', gen_lr)
    tf.summary.scalar('discriminator_lr', dis_lr)
    return train_ops


@click.command()
@click.argument('data_real_path', type=click.Path(file_okay=True, dir_okay=False, exists=True), nargs=1)
@click.argument('data_sim_path', type=click.Path(file_okay=True, dir_okay=False, exists=True), nargs=1)
@click.argument('log_dir', type=click.Path(file_okay=False, dir_okay=True, exists=False), nargs=1)
@click.option('--batch_size', type=click.INT, default=1, help='Batch size to be used for training.')
@click.option('--steps', type=click.INT, default=100000, help='Number of training steps.')
@click.option('--generator_lr', type=click.FLOAT, default=0.0002, help='The generator learning rate.')
@click.option('--discriminator_lr', type=click.FLOAT, default=0.0001, help='The discriminator learning rate.')
@click.option('--cycle_loss_weight', type=click.FLOAT, default=10., help='The weight of cycle consistency loss.')
@click.option('--examples_limit', type=click.INT, default=98000,
              help='The maximum number of training examples to use for each domain.')
@click.option('--checkpoint_freq', type=click.INT, default=900, nargs=1,
              help='Frequency, in seconds, that model weights are saved during training.')
@click.option('--seed', type=click.INT, default=None, nargs=1)
def main(data_real_path, data_sim_path, log_dir, batch_size, steps, generator_lr, discriminator_lr, cycle_loss_weight,
         examples_limit, checkpoint_freq, seed):
    """Trains a CycleGAN model."""
    # Set random seed
    if seed:
        tf.set_random_seed(seed)

    # Load data
    real = load_image_h5(data_real_path)
    sim = load_image_h5(data_sim_path)

    # Process real data
    data_real = np.expand_dims(real[:, :, :, 0], 3)
    data_real = (data_real.astype('float64') - 127.5) / 127.5
    data_real = data_real[:examples_limit]

    # Process simulated data
    sim_filtered = np.expand_dims(sim[:, :, :, 0], 3)
    sim_filtered = (sim_filtered.astype('float64') - 127.5) / 127.5
    blanks = np.ones((98000 - sim_filtered.shape[0], 128, 128, 1))
    data_sim = np.concatenate([sim_filtered, blanks])
    data_sim = data_sim[:examples_limit]

    real_data_iterator, real_iterator_init_hook = data_to_stream(data_real, batch_size)
    sim_data_iterator, sim_iterator_init_hook = data_to_stream(data_sim, batch_size)

    # Define CycleGAN model.
    cyclegan_model = _define_model(sim_data_iterator.get_next(), real_data_iterator.get_next())

    # Define CycleGAN loss.
    cyclegan_loss = tfgan.cyclegan_loss(
        cyclegan_model,
        cycle_consistency_loss_weight=cycle_loss_weight,
        tensor_pool_fn=tfgan.features.tensor_pool)

    # Define CycleGAN train ops.
    train_ops = _define_train_ops(cyclegan_model, cyclegan_loss, generator_lr, discriminator_lr, steps)

    # Training
    train_steps = tfgan.GANTrainSteps(1, 1)

    status_message = tf.string_join(
        [
            'Starting train step: ',
            tf.as_string(tf.train.get_or_create_global_step())
        ],
        name='status_message')

    tfgan.gan_train(
        train_ops,
        logdir=log_dir,
        save_checkpoint_secs=checkpoint_freq,
        get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
        hooks=[
            tf.train.StopAtStepHook(num_steps=steps),
            tf.train.LoggingTensorHook([status_message], every_n_iter=100),
            real_iterator_init_hook,
            sim_iterator_init_hook
        ])


if __name__ == '__main__':
    main()
