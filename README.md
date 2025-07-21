# Crypten and Torch Experiments

## Setup
Crypten requires Python version 3.7; one way to switch to this version is to use pyenv (what we used).

Additionally, Crypten needs old sklearn. To install this version, you can use the following command:

```export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True```

Then install crypten:
```pip install crypten```

Before running the experiments, double-check the global variables are set correctly in each `.py` script:
- NUM_EPOCHS: number of epochs to run the training for.
- BATCH_SIZE: batch size for training.
- OUTFILE: filename to save the results.
- LEARNING_RATE: learning rate for training.
- NUM_TRIALS: number of trials to run.

Then, run the scripts, e.g. `python crypten_nn_cifar.py`.


## Directory: crypten_experiments
Contains the experimental results for CrypTen on FashionMNIST and CIFAR.

The columns in the csv files (under `***_nn.csv`) are as follows:
- Time taken in epoch
- Epoch number
- Batch size
- Number of training images
- Accuracy in training
- Test time
- Number of agents (if applicable)

## Directory: torch_experiments
Contains the experimental results for Torch on FashionMNIST and CIFAR. Also contains the scripts used to run the experiments.

The columns in the csv files (under `***_total_time.csv`) are as follows:
- Total time: Total time taken for the experiment in seconds.
- Data Size: The number of total training images used.

The columns in the csv files (under `***.csv`) are as follows:
- Time taken in epoch
- Epoch number
- Batch size
- Number of training images
- Accuracy in training
- Test time