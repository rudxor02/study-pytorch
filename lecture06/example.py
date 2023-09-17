import torch

x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0], [1.0]])


def train(activation: torch.nn.Module, print_every=True):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return activation(self.linear(x))

    model = Model()
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        if print_every:
            print(f"Epoch {epoch}: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    hour_var_1 = torch.tensor([[1.0]])
    print(
        f"Prediction 1 with {activation.__class__.__name__}: {model(hour_var_1).item() > 0.5}"
    )
    hour_var_7 = torch.tensor([[7.0]])
    print(
        f"Prediction 7 with {activation.__class__.__name__}: {model(hour_var_7).item() > 0.5}"
    )


if __name__ == "__main__":
    train(torch.nn.Sigmoid())
