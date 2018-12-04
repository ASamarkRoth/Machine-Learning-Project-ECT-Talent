[1]: https://arxiv.org/abs/1409.1556

# Convolutional Neural Networks

This directory contains code for training and evaluating convolutional neural networks (CNNs) on image representations
of ATTPC events (see [`generate_images.py`](../data-processing/generate_images.py)). 

## Training

Use `train.py` to train a model using the [VGG16][1] architecture with ImageNet weights. The weights can either
be fine-tuned or frozen (in which case they are merely a means of feature extraction).

The first argument should be a path to an HDF5 file in the format expected by `utils.data.load_image_h5`. The second
argument is a path to where the training logs should be saved (logs include TensorBoard summaries and model weights
for each epoch). There are several command-line options available to specify training settings (run the script with
the `--help` flag for details).

## Evaluation

Models can be evaluated using `eval.py`. Loss, accuracy, and a classification report will be printed to the console.
The first argument should be a path to a model file (in HDF5 format), and the second argument is a path to
testing data (in the format expected by `utils.data.load_image_h5`).