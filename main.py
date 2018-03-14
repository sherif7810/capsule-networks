import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Capsule(nn.Module):
    """It wraps a capsule."""
    def __init__(self, j):
        super(Capsule, self).__init__()

        self.b = Variable(torch.zeros(j).float())
        self.softmax = nn.Softmax()

    def forward(self, u_hat):
        c = self.softmax(self.b)
        s = torch.dot(c, u_hat)
        return s

    def route(self, u_hat, v):
        self.b += torch.dot(u_hat, v)


class Net(nn.Module):
    """Capsule network."""
    def __init__(self, input_size, j, caps_num):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(3, 6)

        self.fc = nn.Linear(input_size, j)
        self.capsules = [Capsule(j) for i in range(caps_num)]

    def forward(self, u_hat):
        s = torch.cat([self.capsule[i](u_hat)
                       for i in range(len(self.capsules))])

        s_mag = torch.sqrt(torch.sum(torch.cat([s[i] ** 2
                                                for i in range(len(s))])))
        s_mag_squared = s_mag ** 2
        v = s_mag_squared * s / ((1 + s_mag_squared) * s_mag)

        return v

    def route(self, u):
        u_hat = self.fc(u)
        v = self(u_hat)
        for i in range(len(self.capsules)):
            self.capsules[i].route(u_hat, v)
