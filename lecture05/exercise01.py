import torch

from lecture05.example import train

optimizers = [
    torch.optim.Adagrad,
    torch.optim.Adam,
    torch.optim.Adamax,
    torch.optim.ASGD,
    # torch.optim.LBFGS, # TypeError: LBFGS.step() missing 1 required positional argument: 'closure'
    torch.optim.RMSprop,
    torch.optim.Rprop,
    torch.optim.SGD,
]

for optimizer in optimizers:
    train(optimizer, print_every=False)
