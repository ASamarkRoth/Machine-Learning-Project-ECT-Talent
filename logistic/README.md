[1]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

# Logistic Regression

This directory contains code for training and evaluating logistic regression models on a discretized representation
of AT-TPC data. 

## Training

Use `train.py` to train a model using the [`scikit-learn` implementation][1] of Logistic Regression.

The first argument should be a path to the directory containing the discretized data (use the `--prefix` flag to specify
which data to use if the directory contains multiple). The second argument is a path to where the model should be saved.
There are several command-line options available to specify training settings (run the script with the `--help` flag
for details).

## Evaluation

Models can be evaluated using `eval.py`. Accuracy and a classification report will be printed to the console.
The first argument should be a path to a model file (in pickle format), and the second argument is a directory
containing testing data (use `--prefix` flag as mentioned above).