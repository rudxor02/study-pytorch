import torch.nn as nn


class Stem(nn.Module):
    def __init__(self, in_channels: int):
        super(Stem, self).__init__()
        self.c1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=0)
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.c3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.c4 = nn.Conv2d(64, 80, kernel_size=1, stride=1, padding=0)
        self.c5 = nn.Conv2d(80, 192, kernel_size=3, stride=1, padding=0)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.mp1(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.mp2(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch1_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch1_3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)

        self.branch2_1 = nn.Conv2d(in_channels, 48, kernel_size=1)
        self.branch2_2 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        raise NotImplementedError
