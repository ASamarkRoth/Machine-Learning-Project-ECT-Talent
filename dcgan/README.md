[1]: https://arxiv.org/abs/1511.06434
[2]: https://arxiv.org/abs/1701.07875
[3]: https://arxiv.org/abs/1406.2661

# Deep Convolutional Generative Adversarial Networks

This directory contains code for training and evaluating a Deep Convolutional Generative Adversarial Network (DCGAN)
on image projections of AT-TPC data.

Generative Adversarial Networks (GANs), as proposed in [Goodfellow et. al. 2014][3],
are a class of generative models in which two neural networks, a _generator_ and a _discriminator_, compete in a
two-player zero-sum game. At convergence, the generator network is capable of producing new data that matches the
distribution of the real data provided during training.

This early exploration of the use of GANs with AT-TPC data is motivated by the question of whether or not these models
can be used to create more realistic simulations of AT-TPC data, potentially as a means for improving CNN classification.

The network architecture used here (defined in [`networks.py`](networks.py)) largely follows the one described by
[Radford et. al. 2015][1], with some minor differences. The training procedure uses the [Wasserstein loss][2].

## Training

Use `train.py` to train a DCGAN on a set of image projections of AT-TPC events.

The first argument should be a path to an HDF5 file in the format expected by `utils.data.load_image_h5`. This is the
file containing the **unlabeled** training images. The second argument is a path to where the training logs should be saved
(logs include TensorBoard summaries and model checkpoints). There are several command-line options available to specify
training settings (run the script with the `--help` flag for details).

## Generation

With `generate.py`, a trained DCGAN model can be used to generate new images. At this stage, the GAN's discriminator
network is ignored, and the generator is used to create new images which ideally match the distribution of those
provided during training. The script will save a composite image of the generated images tiled in a grid. The only
argument is the path to the directory containing the model checkpoints. The latest checkpoint will be used.