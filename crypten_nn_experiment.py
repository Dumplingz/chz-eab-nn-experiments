import crypten
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch_nn_modules import ExampleNet

import torch.nn as nn

def download_mnist():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )


def train_encrypted_nn(model, train_loader, test_loader, loss_fn, optimizer):
    """
    Trains an encrypted model on data.

    Params:
    - model: the model to train
    - train_loader: the data loader for training data
    - test_loader: the data loader for test data
    - loss_fn: the loss function to use
    - optimizer: the optimizer to use

    Returns:
    - None
    """
    for epoch in range(10):
        for batch, (X, y) in enumerate(train_loader):
            # Encrypt the data
            X_enc = crypten.cryptensor(X)
            y_enc = crypten.cryptensor(y)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(X_enc)
            loss = loss_fn(output, y_enc)

            # Backward pass


def main():
    crypten.init()

if __name__ == '__main__':
    main()