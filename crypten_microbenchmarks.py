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


def main():
    crypten.init()
    torch.set_num_threads(1)

    loss_fn = crypten.nn.CrossEntropyLoss()


    # Create torch tensor
    x = torch.tensor([1.2e7, 2e9, -3e10])
    y = torch.tensor([1.0,0,0])

    # Encrypt x,y
    x_enc = crypten.cryptensor(x)
    y_enc = crypten.cryptensor(y)

    loss = loss_fn(x_enc, y_enc)


    # Decrypt x
    x_dec = x_enc.get_plain_text()
    crypten.print(x_dec)

    loss_dec = loss.get_plain_text()
    crypten.print("Crypten Loss", loss_dec)

    dec_loss_fn = nn.CrossEntropyLoss()

    dec_loss = dec_loss_fn(x,y)

    print("Plaintext Loss" , dec_loss)

if __name__ == "__main__":
    main()