import torch.nn as nn
import torch.nn.functional as F
import torch
import crypten
from crypten import CrypTensor


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#Define an example network
class ExampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

def test(dataloader, model, loss_fn, print_rate = 50):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    y_eye = torch.eye(10)

    batch_num = 0

    with CrypTensor.no_grad():
        for X, y in dataloader:
            X = crypten.cryptensor(X)
            pred = model(X)
            y_one_hot = crypten.cryptensor(y_eye[y])

            test_loss += loss_fn(pred, y_one_hot)

            plaintext_pred = pred.get_plain_text()
            correct += (plaintext_pred.argmax(1) == y).type(torch.float).sum().item()
            batch_num += 1
            if batch_num % print_rate == 1:
                crypten.print(test_loss.get_plain_text())
                crypten.print(f"batch number: {batch_num}/{num_batches}")
    test_loss = test_loss.get_plain_text() / num_batches
    correct /= size
    crypten.print(f"Test Error: \n Accuracy: {correct}, Avg loss: {test_loss} \n")

    return correct