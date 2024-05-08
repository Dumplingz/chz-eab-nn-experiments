import crypten
import crypten.optim
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch_nn_modules import ExampleNet, test
from torch.utils.data import DataLoader
import crypten.mpc as mpc
import time
import torch.nn as nn
import csv

# Define training parameters
NUM_EPOCHS = 3
BATCH_SIZE = 64
OUTFILE = "crypten_experiments/datasize_nn.csv"
LEARNING_RATE = 0.001
NUM_TRIALS = 1

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
def train_encrypted_nn(train_data, train_labels, test_loader, batch_size=BATCH_SIZE):
    """
    Trains an encrypted model on data provided by train_loader.

    Params:
    - train_data: training data
    - train_labels: training labels

    Returns:
    - None
    """
    # dummy input for crypten model
    dummy_input = torch.empty(1, 1, 28, 28)

    pytorch_model = ExampleNet()
    print("converting pytorch model to crypten model")
    model = crypten.nn.from_pytorch(pytorch_model, dummy_input)
    print("model converted")
    loss_fn = crypten.nn.CrossEntropyLoss()

    crypten.print("encrypting model")
    model.encrypt()
    # Set train mode
    crypten.print("setting train mode")
    model.train()

    print("ready to train")
    size = len(train_data)

    y_eye = torch.eye(10)
    train_labels_one_hot = y_eye[train_labels]

    encrypted_train_data = crypten.cryptensor(train_data)
    encrypted_train_labels_one_hot = crypten.cryptensor(train_labels_one_hot, requires_grad=True)


    num_batches = encrypted_train_data.size(0) // batch_size

    for epoch in range(NUM_EPOCHS):
        epoch_time_start = time.perf_counter()
        prev_params = [None] * 6
        for batch in range(5):
            start, end = batch * batch_size, (batch + 1) * batch_size

            X_enc = encrypted_train_data[start:end]
            y_enc = encrypted_train_labels_one_hot[start:end]

            # y_enc.requires_grad = True

            # print(y_enc)
            start_time = time.perf_counter()

            # Forward pass
            output = model(X_enc)
            loss = loss_fn(output, y_enc)

            model.zero_grad()

            # perform backward pass:
            loss.backward()
            model.update_parameters(LEARNING_RATE)

            # print the crypten model parameters in plaintext
            plain_params = model.parameters()
            for i, p in enumerate(plain_params):
                # if(p.grad is None):
                #     crypten.print(f"grad: false")
                params = p.get_plain_text()
                # crypten.print(f"Model parameters [{i}]: {params}, {type(params)}")
                if (prev_params[i] is not None) and (torch.equal(params, prev_params[i])):
                    crypten.print(f"Model parameters [{i}] did not change")
                prev_params[i] = params


            if batch % 100 == 0:
                end_time = time.perf_counter()

                # Print progress every batch:
                batch_loss = loss.get_plain_text()

                current = (batch + 1) * len(X_enc)
                crypten.print(f"loss: {batch_loss}  [{current}/{size}], time: {end_time-start_time}")


        epoch_time_end = time.perf_counter()

        test_start_time = epoch_time_end

        test(test_loader, model, loss_fn)
        model.train()

        test_end_time = time.perf_counter()

        epoch_duration = epoch_time_end - epoch_time_start
        crypten.print(f"Epoch {epoch} took {epoch_duration} seconds")
        with open(OUTFILE, "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow([epoch_duration, epoch, batch_size, encrypted_train_data.size(0)])



def main():
    crypten.common.serial.register_safe_class(ExampleNet)
    training_data, test_data = download_mnist()

    print(type(training_data))
    array_training_data = torch.tensor(training_data.data).float()
    array_training_labels = torch.tensor(training_data.targets).long()
    print(array_training_data.shape)

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break


    crypten.init()
    torch.set_num_threads(1)

    print("training encrypted model")
    for trial in range(NUM_TRIALS):
        for data_size in [7500, 15000, 30000, 60000]:
            print(f"trial {trial} batch size {data_size}")
            # train_encrypted_nn(train_dataloader, test_dataloader)
            train_encrypted_nn(array_training_data[:data_size], array_training_labels[:data_size], test_dataloader)

if __name__ == '__main__':
    main()