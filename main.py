import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Capsule(nn.Module):
    """It wraps a capsule."""
    def __init__(self):
        super(Capsule, self).__init__()

        self.b = Variable(torch.zeros(1024).float())
        self.softmax = nn.Softmax()

    def forward(self, u_hat):
        c = self.softmax(self.b)
        s = torch.dot(c, u_hat)
        return s

    def route(self, u_hat, v):
        self.b += torch.dot(u_hat, v)


class Net(nn.Module):
    """Capsule network."""
    def __init__(self, caps_num):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(3, 6, kernel_size=4)

        self.fc = nn.Linear(7840, 1024)
        self.capsules = [Capsule() for i in range(caps_num)]

    def forward(self, u):
        u_hat = self.fc(u.view(1, -1))

        s = torch.cat([self.capsules[i](u_hat)
                       for i in range(len(self.capsules))])

        s_mag = torch.sqrt(torch.sum(torch.cat([s[i] ** 2
                                                for i in range(len(s))])))
        s_mag_squared = s_mag ** 2
        v = s_mag_squared * s / ((1 + s_mag_squared) * s_mag)

        return v

    def route(self, u):
        u_hat = self.fc(u.view(1, -1))
        v = self(u_hat)
        for i in range(len(self.capsules)):
            self.capsules[i].route(u_hat, v)


if __name__ == '__main__':
    batch_size = 10
    test_batch_size = 10
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=test_batch_size, shuffle=True)

    # caps_net = Net(10)

    # for batch_idx, (data, target) in enumerate(train_loader):
    #     data, target = Variable(data), Variable(target)
