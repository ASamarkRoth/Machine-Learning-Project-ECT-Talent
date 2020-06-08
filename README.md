[1]: https://arxiv.org/abs/1810.10350
[2]: https://github.com/ATTPC/pytpc

# Machine Learning Methods for AT-TPC Event Classification

This repository contains research code that explores the use of machine learning methods to classify AT-TPC events
based on the product of the reaction. This work was done using data from Argon 46 experiments.

See [Machine Learning Methods for Track Classification in the AT-TPC][1].

## Prerequisites

* `pytpc` (found [here][2])
* `click`
* `pandas`
* `numpy`
* `tensorflow<2`
* `matplotlib`
* `h5py`
* `pyyaml`
* `scipy`
* `scikit-learn`


## Usage

The following shows how to run the CNN training script. You can pass the `--help`
flag to see all available options in the command-line interface.

```
python cnn/train.py
```

You should make sure that the repository's root directory has been added to the
Python path in order to properly run the scripts.
