import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, PILToTensor
import time
import csv
import pandas as pd
from torch_helpers import download_mnist, NeuralNetwork, device, train, test

BATCH_SIZE = 64


if __name__ == "__main__":
    torch.set_num_threads(1)
    
    training_data, test_data = download_mnist()
    
    loss_fn = nn.CrossEntropyLoss()
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    epochs = 3
    for datasize in [7500,15000,30000,60000]:
        model = NeuralNetwork().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            subset_training_data = Subset(training_data, range(datasize))
            train_dataloader_subset = DataLoader(subset_training_data, batch_size=BATCH_SIZE)
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
            with open("datasize_nn.csv", "a") as fp:
                wr = csv.writer(fp, dialect='excel')
                # epoch_duration, epoch, batch_size, data_size, accuracy, test_duration
                wr.writerow([epoch_duration, epoch, BATCH_SIZE, datasize, accuracy, test_duration])

    print("Done!")