import crypten
import crypten.optim
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch_nn_modules import ExampleNet
from torch.utils.data import DataLoader
import crypten.mpc as mpc
import time
import torch.nn as nn
import csv

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
    
    return training_data, test_data

@mpc.run_multiprocess(world_size=2)
def train_encrypted_nn(train_loader, test_loader):
    """
    Trains an encrypted model on data provided by train_loader.

    Params:
    - train_loader: the data loader for training data
    - test_loader: the data loader for test data

    Returns:
    - None
    """
    # dummy input for crypten model
    dummy_input = torch.empty(1, 1, 28, 28)

    pytorch_model = ExampleNet()
    print("converting pytorch model to crypten model")
    model = crypten.nn.from_pytorch(pytorch_model, dummy_input)
    print("model converted")
    optimizer = crypten.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = crypten.nn.CrossEntropyLoss()

    crypten.print("encrypting model")
    model.encrypt()
    # Set train mode
    crypten.print("setting train mode")
    model.train()

    print("ready to train")
    # Define training parameters
    num_epochs = 3
    size = len(train_loader.dataset)

    for epoch in range(num_epochs):
        epoch_time_start = time.perf_counter()
        for batch, (X, y) in enumerate(train_loader):
            
            start_time = time.perf_counter()

            y_eye = torch.eye(10)
            y_one_hot = y_eye[y]

            # Encrypt the data
            X_enc = crypten.cryptensor(X)
            y_enc = crypten.cryptensor(y_one_hot)

            # Forward pass
            output = model(X_enc)
            loss = loss_fn(output, y_enc)

            # perform backward pass: 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            if batch % 100 == 0:
                end_time = time.perf_counter()

                # Print progress every batch:
                batch_loss = loss.get_plain_text()

                current = (batch + 1) * len(X)
                crypten.print(f"loss: {batch_loss}  [{current}/{size}], time: {end_time-start_time}")
                
            
        epoch_time_end = time.perf_counter()
        epoch_duration = epoch_time_end - epoch_time_start
        crypten.print(f"Epoch {epoch} took {epoch_duration} seconds")
        with open(f"crypten_experiments/basic_nn.csv", "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow([epoch_duration, epoch])



def main():
    
    training_data, test_data = download_mnist()
    
    batch_size = 64
    
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    

    crypten.init()
    torch.set_num_threads(1)
    
    print("training encrypted model")
    for trial in range(9):
        train_encrypted_nn(train_dataloader, test_dataloader)

if __name__ == '__main__':
    main()