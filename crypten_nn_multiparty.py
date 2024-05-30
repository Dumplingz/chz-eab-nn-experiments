import crypten
import crypten.optim
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch_nn_helpers import ExampleNet, test, split_tensor_along_party, download_mnist
from torch.utils.data import DataLoader
import crypten.mpc as mpc
import time
import torch.nn as nn
import csv

# Define training parameters
NUM_EPOCHS = 3
BATCH_SIZE = 64
OUTFILE = "crypten_experiments/multiparty_nn.csv"
LEARNING_RATE = 0.001
NUM_TRIALS = 1
WORLD_SIZE = 2

def train_encrypted_nn(train_data, train_labels, test_loader, batch_size=BATCH_SIZE):
    """
    Trains an encrypted model on data provided by train_loader.

    Params:
    - train_data: training data
    - train_labels: training labels

    Returns:
    - None
    """
    # print("hi")
    # return
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

    # make y data one hot
    y_eye = torch.eye(10)
    train_labels_one_hot = y_eye[train_labels]

    # split data evenly among all parties, then concat them
    split_train_data = split_tensor_along_party(train_data,WORLD_SIZE)
    split_train_labels_one_hot = split_tensor_along_party(train_labels_one_hot,2)
    encrypted_split_train_data = []
    encrypted_split_train_labels_one_hot = []
    for i, (data, labels) in enumerate(zip(split_train_data, split_train_labels_one_hot)):
        encrypted_split_train_data.append(crypten.cryptensor(data, src=i))
        encrypted_split_train_labels_one_hot.append(crypten.cryptensor(labels, src=i))
    encrypted_train_data = crypten.cat(encrypted_split_train_data, dim=0)
    encrypted_train_labels_one_hot = crypten.cat(encrypted_split_train_labels_one_hot, dim=0)

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
        for num_parties in [2,3,4]:
            # Create data loaders ... this is hacky af
            train_dataloader = DataLoader(training_data, batch_size=60000)

            # batch size is now data size, and only one batch is taken at a time...
            for array_training_data,array_training_labels in train_dataloader:
                print(f"trial {trial} party size {num_parties}")
                mpc.run_multiprocess(world_size=num_parties)(train_encrypted_nn)(array_training_data, array_training_labels, test_dataloader, 64)
                break

if __name__ == '__main__':
    main()