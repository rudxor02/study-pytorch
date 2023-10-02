from __future__ import print_function

import time

import pandas as pd
import torch.nn.functional as F
from torch import cuda, float32, from_numpy, nn, optim
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset

batch_size = 64
device = "cuda" if cuda.is_available() else "cpu"
print(f'Training Otto Model on {device}\n{"=" * 44}')


class OttoDataset(Dataset):
    def __init__(self):
        df = pd.read_csv("data/otto_train.csv")
        df = df.drop(["id"], axis=1)
        df = df.dropna(axis=0)
        df = df.reset_index(drop=True)

        self.x_data = from_numpy(df.drop(["target"], axis=1).values)

        def encode(x):
            if x == "Class_1":
                return 0
            elif x == "Class_2":
                return 1
            elif x == "Class_3":
                return 2
            elif x == "Class_4":
                return 3
            elif x == "Class_5":
                return 4
            elif x == "Class_6":
                return 5
            elif x == "Class_7":
                return 6
            elif x == "Class_8":
                return 7
            elif x == "Class_9":
                return 8

        target = df["target"].apply(encode)
        self.y_data = one_hot(from_numpy(target.values), num_classes=9)
        self.len = df.shape[0]
        print(self.x_data.shape, self.y_data.shape)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


otto_dataset = OttoDataset()

# Data Loader (Input Pipeline)
train_loader = DataLoader(dataset=otto_dataset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(93, 50)
        self.l2 = nn.Linear(50, 25)
        self.l3 = nn.Linear(25, 12)
        self.l4 = nn.Linear(12, 9)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)


model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(float32), target.to(float32)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print(
                "Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


if __name__ == "__main__":
    since = time.time()
    for epoch in range(1, 10):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f"Training time: {m:.0f}m {s:.0f}s")
