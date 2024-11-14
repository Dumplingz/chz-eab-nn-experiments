import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, PILToTensor
import time
import csv
import mnist_reader
import gzip
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

# code is modified from mnist_reader.py in fashion-mnist repository
def load_mnist(image_path, label_path):
    with open(label_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with open(image_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


class FashionMNISTDataset(Dataset):
    def __init__(self, X_labels, y_labels, transform=None):
        self.transform = transform
        self.X_labels = X_labels
        self.y_labels = y_labels
    
    def __len__(self):      
        return len(self.X_labels)

    def __getitem__(self, index):
        x = self.X_labels[index]
        y = self.y_labels[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


if __name__ == "__main__":
    torch.set_num_threads(1)

    # Define training parameters
    NUM_EPOCHS = 3
    BATCH_SIZE = 64
    OUTFILE = "datasize_nn.csv"
    LEARNING_RATE = 0.001
    NUM_TRIALS = 1

    # Load data
    # X_train, y_train = load_mnist(
    #     "../fashion-mnist/data/fashion/train-images-idx3-ubyte.gz",
    #     "../fashion-mnist/data/fashion/train-labels-idx1-ubyte.gz",
    # )
    # X_test, y_test = load_mnist(
    #     "../fashion-mnist/data/fashion/t10k-images-idx3-ubyte.gz",
    #     "../fashion-mnist/data/fashion/t10k-labels-idx1-ubyte.gz",
    # )
    X_train, y_train = load_mnist(
        "../data/FashionMNIST/raw/train-images-idx3-ubyte",
        "../data/FashionMNIST/raw/train-labels-idx1-ubyte",
    )
    X_test, y_test = load_mnist(
        "../data/FashionMNIST/raw/t10k-images-idx3-ubyte",
        "../data/FashionMNIST/raw/t10k-labels-idx1-ubyte",
    )
    
    print(X_train.shape)
    print(type(X_train))
    
    X_train = torch.from_numpy(X_train).float()/255.0
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()/255.0
    y_test = torch.from_numpy(y_test).long()
    
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape)
        print(data[0])
        print(target.shape)
        print(target)
        break