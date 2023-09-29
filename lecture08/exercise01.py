import pandas as pd
from torch import float32, from_numpy, nn, optim
from torch.utils.data import DataLoader, Dataset


class TitanicDataset(Dataset):
    def __init__(self):
        df = pd.read_csv("data/titanic_train.csv")
        df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
        df = df.dropna(axis=0)
        df = df.reset_index(drop=True)
        df["Sex"] = df["Sex"].apply(lambda x: 1 if "Male" else 0)
        df["Embarked"] = df["Embarked"].apply(
            lambda x: 0 if x == "S" else 1 if x == "C" else 2
        )

        self.x_data = from_numpy(df.drop(["Survived"], axis=1).values)
        self.y_data = from_numpy(df["Survived"].values).reshape(-1, 1)
        self.len = df.shape[0]
        print(self.x_data.shape, self.y_data.shape)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(7, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


dataset = TitanicDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

model = Model()
criterion = nn.BCELoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(float32)
        labels = labels.to(float32)
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(f"Loss: {loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
