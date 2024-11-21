import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, PILToTensor
import time
import csv
import pandas as pd
from torch_helpers import download_mnist, NeuralNetwork, device, train, test

BATCH_SIZE = 64

def create_mnist_dataloader(training_data, test_data, data_size, batch_size):
        
    X_train, y_train = training_data.data.numpy(), training_data.targets.numpy()
    X_test, y_test = test_data.data.numpy(), test_data.targets.numpy()
    
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
    
    train_dataset = Subset(train_dataset, range(data_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    num_trials = int(sys.argv[1])
    
    os.makedirs("mnist", exist_ok=True)

    torch.set_num_threads(1)
    
    training_data, test_data = download_mnist()
    
    loss_fn = nn.CrossEntropyLoss()
    # train_dataloader, test_dataloader = create_mnist_dataloader(training_data, test_data, 60000, BATCH_SIZE)

    epochs = 3
    for datasize in [7500,15000,30000,60000]:
        for trial in range(num_trials):
            total_time_start = time.perf_counter()
            model = NeuralNetwork().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            for epoch in range(epochs):
                # subset_training_data = Subset(training_data, range(datasize))
                # train_dataloader_subset = DataLoader(subset_training_data, batch_size=BATCH_SIZE)
                train_dataloader_subset, test_dataloader = create_mnist_dataloader(training_data, test_data, datasize, BATCH_SIZE)

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
                with open("mnist/mnist.csv", "a") as fp:
                    wr = csv.writer(fp, dialect='excel')
                    # epoch_duration, epoch, batch_size, data_size, accuracy, test_duration
                    wr.writerow([epoch_duration, epoch, BATCH_SIZE, datasize, accuracy, test_duration])
            total_time_end = time.perf_counter()
        with open("mnist/mnist_total_time.csv", "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow([total_time_end - total_time_start])

    print("Done!")