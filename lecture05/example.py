from typing import Type

import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


def train(optimizer_cls: Type[torch.optim.Optimizer], print_every=True):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            y_pred = self.linear(x)
            return y_pred

    criterion = torch.nn.MSELoss(size_average=False)
    model = Model()
    optimizer = optimizer_cls(model.parameters(), lr=0.01)
    for epoch in range(500):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        if print_every:
            print(epoch, loss.data.item())
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    hour_var = Variable(torch.Tensor([[4.0]]))
    print(
        f"predict (after training) with {optimizer_cls.__name__}",
        4,
        model.forward(hour_var).data.item(),
    )


if __name__ == "__main__":
    train(torch.optim.SGD)
