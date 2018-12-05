# Fully-connected Neural Networks

This directory contains code for training and evaluating FCNN models on a discretized representation
of AT-TPC data. 

## Training

Use `train.py` to train a simple FCNN model with dropout.

The first argument should be a path to the directory containing the discretized data (use the `--prefix` flag to specify
which data to use if the directory contains multiple). The second argument is a path to where the training logs should
be saved (logs include TensorBoard summaries and model weights for each epoch). The third argument is some number of
integers, where the n<sup>th</sup> integer represents the number of neurons in the n<sup>th</sup> hidden layer.
There are several command-line options available to specify training settings (run the script with the `--help`
flag for details).

## Evaluation

Models can be evaluated using `eval.py`. Accuracy and a classification report will be printed to the console.
The first argument should be a path to a model file (in pickle format), and the second argument is a directory
containing testing data (use `--prefix` flag as mentioned above).