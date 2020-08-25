"""Script for evaluating a CNN classifier.

Author: Ryan Strauss
"""
import click
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import itertools

from data_processing.data import CLASS_NAMES, load_image_h5

FEATURES = 0
TARGETS = 1


@click.command()
@click.argument('model_file', type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=1)
@click.argument('data', type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=1)
@click.option('--data_combine', is_flag=True,
              help='If flag is set, the training and test sets within the HDF5 file pointed '
                   'to by `data` will be combined into a single training set.')
@click.option('--binary', type=click.BOOL, default=True, nargs=1,
              help='If true, the labels will be collapsed to binary values, where any non-zero label will become a 1.')
@click.option('--examples_limit', type=click.INT, default=-1, nargs=1,
              help='Limit on the number of examples to use during testing.')
@click.option('--seed', type=click.INT, default=71, nargs=1, help='Random seed.')
@click.option('--reverse_labels', is_flag=True, help='If flag is set, labels will be reversed.')
def main(model_file, data, data_combine, binary, examples_limit, seed, reverse_labels):
    eval(model_file, data, data_combine, binary, examples_limit, seed, reverse_labels)
    
def eval(model_file, data, name='CNN', data_combine=False, binary=True, examples_limit=-1, seed=71, reverse_labels=False):
    """This function will evaluate a CNN classifier.

    Loss, accuracy, and classification metrics are printed to the console.
    """
    assert model_file.endswith('.h5'), 'model_file must point to an HDF5 file'
    #assert data.endswith('.h5'), 'data must point to an HDF5 file'

    # Set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load data
    #if data_combine:
    #    a, b = load_image_h5(data, categorical=True, binary=binary, reverse_labels=reverse_labels)
    #    test = np.concatenate([a[FEATURES], b[FEATURES]], axis=0), np.concatenate([a[TARGETS], b[TARGETS]], axis=0)
    #else:
    #    _, test = load_image_h5(data, categorical=True, binary=binary, reverse_labels=reverse_labels)

    if examples_limit == -1:
        examples_limit = data[TARGETS].shape[0]

    # Load the model
    model = tf.keras.models.load_model(model_file)

    # Evaluate the model
    loss, acc = model.evaluate(data[FEATURES][:examples_limit], data[TARGETS][:examples_limit])

    # Make predictions
    preds = np.round(model.predict(data[FEATURES][:examples_limit]))
    
    #if binary:
    #    if reverse_labels:
    #        target_names = [CLASS_NAMES[-1], 'non-' + CLASS_NAMES[-1]]
    #    else:
    #        target_names = [CLASS_NAMES[0], 'non-' + CLASS_NAMES[0]]
    #else:
    #    target_names = CLASS_NAMES
    
    target_names = CLASS_NAMES

    if reverse_labels:
        target_names = target_names[::-1]

    # Get classification metrics
    report = classification_report(data[TARGETS][:examples_limit], preds, target_names=target_names, digits=2)

    # Plot confusion matrix
    cm = confusion_matrix(y_true=data[TARGETS][:examples_limit], y_pred=preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp = disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
    plt.show()
    
    #plot_confusion_matrix(cm=cm, classes=target_names, title='', normalize=False, cmap='viridis')
    
    #plot_confusion_matrix(model, data[FEATURES][:examples_limit], data[TARGETS][:examples_limit], display_labels=CLASS_NAMES)

    # Print the results
    print("\nClassification Report for: {}\n".format(name))
    print('Loss: {}, Accuracy: {}\n'.format(loss, acc))
    print(report)

#def plot_confusion_matrix(cm, classes,
#                        normalize=False,
#                        title='Confusion matrix',
#                        cmap=plt.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
#
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, cm[i, j],
#            horizontalalignment="center",
#            color="white" if cm[i, j] > thresh else "black")
#
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')

if __name__ == '__main__':
    main()
