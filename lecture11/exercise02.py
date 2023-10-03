import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import transforms

# Training settings
batch_size = 64

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True
)

test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=False
)


class ResBlockConv3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockConv3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out_conv1 = self.conv1(x)
        out_bn1 = self.bn1(out_conv1)
        out_relu1 = self.relu(out_bn1)
        out_conv2 = self.conv2(out_relu1)
        out_bn2 = self.bn2(out_conv2)
        out = out_bn2
        if self.stride == 1:
            out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels: int):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resblock64_1 = ResBlockConv3(in_channels=64, out_channels=64)
        self.resblock64_2 = ResBlockConv3(in_channels=64, out_channels=64)
        self.resblock64_3 = ResBlockConv3(in_channels=64, out_channels=64)

        self.resblock128_1 = ResBlockConv3(in_channels=64, out_channels=128, stride=2)
        self.resblock128_2 = ResBlockConv3(in_channels=128, out_channels=128)
        self.resblock128_3 = ResBlockConv3(in_channels=128, out_channels=128)
        self.resblock128_4 = ResBlockConv3(in_channels=128, out_channels=128)

        self.resblock256_1 = ResBlockConv3(in_channels=128, out_channels=256, stride=2)
        self.resblock256_2 = ResBlockConv3(in_channels=256, out_channels=256)
        self.resblock256_3 = ResBlockConv3(in_channels=256, out_channels=256)
        self.resblock256_4 = ResBlockConv3(in_channels=256, out_channels=256)
        self.resblock256_5 = ResBlockConv3(in_channels=256, out_channels=256)
        self.resblock256_6 = ResBlockConv3(in_channels=256, out_channels=256)

        self.resblock512_1 = ResBlockConv3(in_channels=256, out_channels=512, stride=2)
        self.resblock512_2 = ResBlockConv3(in_channels=512, out_channels=512)
        self.resblock512_3 = ResBlockConv3(in_channels=512, out_channels=512)

        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)

        print(x.shape)

        x = self.resblock64_1(x)
        x = self.resblock64_2(x)
        x = self.resblock64_3(x)

        print(x.shape)

        x = self.resblock128_1(x)
        x = self.resblock128_2(x)
        x = self.resblock128_3(x)
        x = self.resblock128_4(x)

        print(x.shape)

        x = self.resblock256_1(x)
        x = self.resblock256_2(x)
        x = self.resblock256_3(x)
        x = self.resblock256_4(x)
        x = self.resblock256_5(x)
        x = self.resblock256_6(x)

        print(x.shape)

        x = self.resblock512_1(x)
        x = self.resblock512_2(x)
        x = self.resblock512_3(x)
        print(x.shape)
        x = self.avgpool(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


model = ResNet(in_channels=3)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


for epoch in range(1, 10):
    train(epoch)
    test()
