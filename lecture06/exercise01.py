import torch

from lecture06.example import train

activations = [
    # torch.nn.ReLU(),
    # torch.nn.ReLU6(),
    # torch.nn.ELU(),
    # torch.nn.SELU(),
    # torch.nn.LeakyReLU(),
    torch.nn.Threshold(0.5, 0.0),
    torch.nn.Hardtanh(min_val=0.0, max_val=1.0),
    torch.nn.Sigmoid(),
    # torch.nn.Tanh(),
]

for activation in activations:
    train(activation, print_every=False)
