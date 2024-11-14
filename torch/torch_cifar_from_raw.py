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
import pickle

def load_cifar(cifar_path):
    with open(cifar_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == "__main__":
    torch.set_num_threads(1)

    # Define training parameters
    NUM_EPOCHS = 3
    BATCH_SIZE = 64
    OUTFILE = "datasize_nn.csv"
    LEARNING_RATE = 0.001
    NUM_TRIALS = 1

    # Load data
    cifar_data = load_cifar("../data/cifar-10-batches-py/data_batch_1")
    X_train = cifar_data[b'data']
    y_train = cifar_data[b'labels']
    
    print(type(cifar_data))
    print(type(X_train))
    
    print(y_train[0])
    
    X_train = torch.from_numpy(X_train).float()/255.0
    print(X_train[0])
    y_train = torch.tensor(y_train).long()
    
    X_train = X_train.reshape(-1, 3, 32, 32)

    X_train = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(X_train)
    
    print(X_train[0])
    
    print(X_train.shape)
    
    train_dataset = TensorDataset(X_train, y_train)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape)
        print(data[0])
        print(target.shape)
        print(target)
        break