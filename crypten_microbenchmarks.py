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
    plain_loss_fn = nn.CrossEntropyLoss()


    # Create torch tensor
    x = torch.tensor([[0.01,0.02,0.001],[4,5,6]])
    y = torch.tensor([[1.0,0,0],[0,0,1]])

    # Encrypt x,y
    x_enc = crypten.cryptensor(x)
    y_enc = crypten.cryptensor(y)

    x_softmax_enc = x_enc.softmax(dim=1)
    print("softmax enc", x_softmax_enc.get_plain_text())
    print("softmax plain", x.softmax(dim=1))

    dec_loss = plain_loss_fn(x,y)
    print("CrossEntropy Plaintext Loss" , dec_loss)


    loss = loss_fn(x_enc, y_enc)

    # Decrypt x
    x_dec = x_enc.get_plain_text()
    crypten.print("decrypted x", x_dec)

    loss_dec = loss.get_plain_text()
    crypten.print("CrossEntropy Crypten Loss", loss_dec)

    # random loss tests
    rand = torch.randn(3,5) * 100
    print("base rand", rand)
    target = torch.randn(3, 5).softmax(dim=1)
    print("target", target)
    plain_loss = plain_loss_fn(rand, target)



    # crypten random loss
    rand_enc = crypten.cryptensor(rand)
    target_enc = crypten.cryptensor(target)

    print("softmax plain", rand.softmax(dim=1))
    print("softmax enc", rand_enc.softmax(dim=1).get_plain_text())

    loss = loss_fn(rand_enc, target_enc)

    dec_loss = loss.get_plain_text()

    print("plaintext loss", plain_loss)

    crypten.print("crypten loss", dec_loss)

    rand_enc_softmax = rand_enc.softmax(dim=1)

    loss_values = rand_enc_softmax.log(input_in_01=True).mul_(target_enc).neg_()
    crypten.print("log, mul, neg:", loss_values.get_plain_text())
    final_values = loss_values.sum().div_(target_enc.size(0))
    crypten.print("sum, div:", final_values.get_plain_text())



    zeros_test = torch.zeros(3,5)
    zeros_log_test = zeros_test.log()
    print("zeros test", zeros_log_test)

    zeros_test_enc = crypten.cryptensor(torch.zeros(3,5))
    zeros_log_test_enc = zeros_test_enc.log(input_in_01=True)
    print("zeros test", zeros_log_test_enc.get_plain_text())


    large_x = torch.tensor([100000200302512352])
    print(large_x)

    large_x_enc = crypten.cryptensor(large_x)

    print(large_x_enc.get_plain_text())

if __name__ == "__main__":
    main()