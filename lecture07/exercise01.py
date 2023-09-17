import numpy as np
from torch import from_numpy, nn, optim

xy = np.loadtxt("data/diabetes.csv.gz", delimiter=",", dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, [-1]])
print(f"X's shape: {x_data.shape} | Y's shape: {y_data.shape}")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l1_1 = nn.Linear(6, 6)
        self.l1_2 = nn.Linear(6, 6)
        self.l1_3 = nn.Linear(6, 6)
        self.l1_4 = nn.Linear(6, 6)
        self.l1_5 = nn.Linear(6, 6)
        self.l2 = nn.Linear(6, 4)
        self.l2_1 = nn.Linear(4, 4)
        self.l2_2 = nn.Linear(4, 4)
        self.l2_3 = nn.Linear(4, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out1 = self.sigmoid(self.l1_1(out1))
        out1 = self.sigmoid(self.l1_2(out1))
        out1 = self.sigmoid(self.l1_3(out1))
        out1 = self.sigmoid(self.l1_4(out1))
        out1 = self.sigmoid(self.l1_5(out1))
        out2 = self.sigmoid(self.l2(out1))
        out2 = self.sigmoid(self.l2_1(out2))
        out2 = self.sigmoid(self.l2_2(out2))
        out2 = self.sigmoid(self.l2_3(out2))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


model = Model()
criterion = nn.BCELoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f"Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
