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

    # create model and convert to crypten
    pytorch_model = ExampleNet()
    model = crypten.nn.from_pytorch(pytorch_model, dummy_input)

    # crypten loss function
    loss_fn = crypten.nn.CrossEntropyLoss()

    # set encrypt mode
    model.encrypt()
    # Set train mode
    model.train()

    crypten.print("ready to train")
    size = len(train_data)

    y_eye = torch.eye(10)
    train_labels_one_hot = y_eye[train_labels]

    encrypted_train_data = crypten.cryptensor(train_data)
    encrypted_train_labels_one_hot = crypten.cryptensor(train_labels_one_hot, requires_grad=True)

    # number of batches
    num_batches = encrypted_train_data.size(0) // batch_size

    for epoch in range(NUM_EPOCHS):
        epoch_time_start = time.perf_counter()
        for batch in range(num_batches):
            start, end = batch * batch_size, (batch + 1) * batch_size

            X_enc = encrypted_train_data[start:end]
            y_enc = encrypted_train_labels_one_hot[start:end]


            # print(y_enc)
            start_time = time.perf_counter()

            # Forward pass
            output = model(X_enc)
            loss = loss_fn(output, y_enc)

            model.zero_grad()

            # perform backward pass:
            loss.backward()

            curr_loss = loss.get_plain_text()
            model.update_parameters(LEARNING_RATE)


            if batch % 100 == 0:
                end_time = time.perf_counter()

                # Print progress every batch:
                batch_loss = loss.get_plain_text()

                current = (batch + 1) * len(X_enc)
                crypten.print(f"loss: {batch_loss}  [{current}/{size}], time: {end_time-start_time}")


        epoch_time_end = time.perf_counter()

        test_start_time = epoch_time_end

        accuracy = test(test_loader, model, loss_fn)
        model.train()

        test_end_time = time.perf_counter()

        test_duration = test_end_time - test_start_time

        epoch_duration = epoch_time_end - epoch_time_start
        crypten.print(f"Epoch {epoch} took {epoch_duration} seconds")
        with open(OUTFILE, "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            # epoch_duration, epoch, batch_size, data_size, accuracy, test_duration
            wr.writerow([epoch_duration, epoch, batch_size, encrypted_train_data.size(0), accuracy, test_duration])



def main():
    crypten.common.serial.register_safe_class(ExampleNet)
    training_data, test_data = download_mnist()

    int_training_data = torch.tensor(training_data.data).float()
    int_training_labels = torch.tensor(training_data.targets).long()

    crypten.init()
    torch.set_num_threads(1)

    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    print("training encrypted model")
    for trial in range(NUM_TRIALS):
        for batch_size in [32,64,128,256]:
            # Create data loaders ... this is hacky af
            train_dataloader = DataLoader(training_data, batch_size=60000)

            # batch size is now data size, and only one batch is taken at a time...
            for array_training_data,array_training_labels in train_dataloader:
                print(f"trial {trial} batch size {batch_size}")
                # train_encrypted_nn(train_dataloader, test_dataloader)
                train_encrypted_nn(array_training_data, array_training_labels, test_dataloader, batch_size)
                break

            # train with non-normalized int [0, 255] data
            # train_encrypted_nn(int_training_data, int_training_labels, test_dataloader, batch_size)

if __name__ == '__main__':
    main()