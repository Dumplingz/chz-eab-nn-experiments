import os
import sys
import torch
import pickle
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, PILToTensor, Compose, Normalize
import torch.nn.functional as F
import time
import csv
from torch_helpers import download_cifar, CifarNetwork, device, train, test

def load_cifar(cifar_paths_train, cifar_path_test):
    """
    Takes in a list of 5 cifar paths plus the test path, and returns the data in np arrays
    """
    X_train = None
    y_train = None
    for cifar_path in cifar_paths_train:
        with open(cifar_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        
        if X_train is None:
            X_train = data[b'data']
            y_train = data[b'labels']
        else:
            X_train = np.vstack((X_train, data[b'data']))
            y_train = np.hstack((y_train, data[b'labels']))
    with open(cifar_path_test, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    X_test = data[b'data']
    y_test = data[b'labels']
    return X_train, y_train, X_test, y_test

def get_np_cifar_data(train_data_idxs, test_data_idx):
    """
    Returns the cifar data in np arrays
    """
    cifar_paths_train = []
    for idx in train_data_idxs:
        cifar_paths_train.append(f"data/cifar-10-batches-py/data_batch_{idx}")
    cifar_path_test = f"data/cifar-10-batches-py/test_batch"
    X_train, y_train, X_test, y_test = load_cifar(cifar_paths_train, cifar_path_test)
    return X_train, y_train, X_test, y_test

def create_cifar_dataloader(X_train, y_train, X_test, y_test, data_size, batch_size):
    # convert from numpy to tensor and normalize
    X_train = torch.from_numpy(X_train).float()/255.0
    X_test = torch.from_numpy(X_test).float()/255.0
    
    # convert to tensor
    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()

    # reshape
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    # normalize over -1 to 1
    X_train = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(X_train)
    X_test = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(X_test)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_dataset = Subset(train_dataset, range(data_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    num_trials = int(sys.argv[1])
    
    os.makedirs("cifar", exist_ok=True)
    torch.set_num_threads(1)
    
    training_data, test_data = download_cifar()

    X_train, y_train, X_test, y_test = get_np_cifar_data([1,2,3,4,5], 1)
    # training_data = datasets.CIFAR10()
    
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 4

    epochs = 3
    for datasize in [6250,12500,25000,50000]:
        for trial in range(num_trials):
            total_time_start = time.perf_counter()
            model = CifarNetwork().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

            for epoch in range(epochs):
                train_dataloader_subset, test_dataloader = create_cifar_dataloader(X_train, y_train, X_test, y_test, datasize, batch_size)
                print(f"Epoch {epoch+1}\n-------------------------------")
                epoch_time_start = time.perf_counter()
                train(train_dataloader_subset, model, loss_fn, optimizer)
                epoch_time_end = time.perf_counter()

                test_start_time = epoch_time_end
                accuracy = test(test_dataloader, model, loss_fn)
                test_end_time = time.perf_counter()

                epoch_duration = epoch_time_end - epoch_time_start
                test_duration = test_end_time - test_start_time

                print(f"Epoch {epoch+1} took {epoch_duration} seconds")
                with open("cifar/cifar.csv", "a") as fp:
                    wr = csv.writer(fp, dialect='excel')
                    # epoch_duration, epoch, batch_size, data_size, accuracy, test_duration
                    wr.writerow([epoch_duration, epoch, batch_size, datasize, accuracy, test_duration])
            total_time_end = time.perf_counter()
            with open("cifar/cifar_total_time.csv", "a") as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow([total_time_end - total_time_start, datasize])

print("Done!")
