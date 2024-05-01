from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        for input_size, output_size in zip(layers[:-1], layers[1:]):
            self.layers.append(nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Tanh(),
            ))
        self.output = nn.Linear(layers[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x).squeeze()


def train(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        net = _training_loop(net, trainloader, device, criterion, optimizer)


def _training_loop(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.MSELoss,
    optimizer: torch.optim.SGD,
) -> nn.Module:
    for features, targets in trainloader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(net(features), targets)
        loss.backward()
        optimizer.step()
    return net


def val(
    net: nn.Module, valloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    criterion = torch.nn.MSELoss(reduction="sum")
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for features, targets in valloader:
            features, targets = features.to(device), targets.to(device)
            outputs = net(features)
            loss += criterion(outputs, targets).item()
    if len(valloader.dataset) == 0:
        raise ValueError("Valloader can't be 0, exiting...")
    loss /= len(valloader.dataset)
    return loss