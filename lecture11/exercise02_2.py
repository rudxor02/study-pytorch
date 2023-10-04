import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import cuda
from torch.autograd import Variable
from torchvision import transforms

# Training settings
batch_size = 64
device = "cuda" if cuda.is_available() else "cpu"
print(f'Training ResNet on {device}\n{"=" * 44}')

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

PATH = './cifar_net.pth'


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


class Res110Net(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 10):
        super(Res110Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU(inplace=True)

        self.resblock16 = [ResBlockConv3(in_channels=16, out_channels=16) for _ in range(32)]
        self.resblock16 = nn.ModuleList(self.resblock16)

        self.resblock32_1 = ResBlockConv3(in_channels=16, out_channels=32, stride=2)
        self.resblock32_2 = [ResBlockConv3(in_channels=32, out_channels=32) for _ in range(31)]
        self.resblock32_2 = nn.ModuleList(self.resblock32_2)

        self.resblock64_1 = ResBlockConv3(in_channels=32, out_channels=64, stride=2)
        self.resblock64_2 = [ResBlockConv3(in_channels=64, out_channels=64) for _ in range(31)]
        self.resblock64_2 = nn.ModuleList(self.resblock64_2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        # print(x.shape)
        for i in range(32):
            x = self.resblock16[i](x)
        # print(x.shape)
        x = self.resblock32_1(x)
        for i in range(31):
            x = self.resblock32_2[i](x)
        # print(x.shape)
        x = self.resblock64_1(x)
        for i in range(31):
            x = self.resblock64_2[i](x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


model = Res110Net(in_channels=3)
model.to(device)




def train(epoch, optimizer, model):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 30 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data, target = data.to(device), target.to(device)
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


if __name__ == "__main__":
    load_model = True
    if load_model:
        model.load_state_dict(torch.load(PATH))
        test(model=model)
    model = model
    for epoch in range(1, 20):
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        train(epoch, optimizer=optimizer, model=model)
        test(model=model)
        torch.save(model.state_dict(), PATH)
    # for epoch in range(1, 30):
    #     optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    #     train(epoch, optimizer=optimizer, model=model)
    #     test(model=model)
    #     torch.save(model.state_dict(), PATH)
    # for epoch in range(1, 30):
    #     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #     train(epoch, optimizer=optimizer, model=model)
    #     test(model=model)
    #     torch.save(model.state_dict(), PATH)