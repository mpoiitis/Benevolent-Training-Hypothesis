import torch
import torch.nn as nn


class MLP(nn.Module):
    """
        MLP model for experiment 1
    """
    def __init__(self, in_dim, n):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, 1)
        )
        self.layers[0].weight.data.copy_(torch.eye(in_dim))  # initialize 1st layer's weights to identity
        self.layers[0].weight.requires_grad = False  # freeze weights of the first layer. Only bias is trained on the 1st layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP2(nn.Module):
    """
        MLP model for experiment 2
    """
    def __init__(self, in_dim, n):
        super(MLP2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, n),
            nn.ReLU(),
            nn.Linear(n, 1)
        )
        self.layers[0].weight.data.copy_(torch.eye(in_dim))  # initialize 1st layer's weights to identity
        self.layers[0].weight.requires_grad = False  # freeze weights of the first layer. Only bias is trained on the 1st layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP_MNIST(nn.Module):
    """
        MLP model for MNIST experiment
    """
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, 1 * 28 * 28),
            nn.ReLU(),
            nn.Linear(1 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.layers[1].weight.data.copy_(torch.eye(1 * 28 * 28))  # initialize 1st layer's weights to identity
        self.layers[1].weight.requires_grad = False  # freeze weights of the first layer. Only bias is trained on the 1st layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CNN(nn.Module):
    """
        CNN model for experiments 1 and 2
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 3 * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (3, 32, 32)),
            nn.Conv2d(3, 96, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(96, 384, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Flatten(),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )
        self.layers[1].weight.data.copy_(torch.eye(3 * 32 * 32))  # initialize 1st layer's weights to identity
        self.layers[1].weight.requires_grad = False  # freeze weights of the first layer. Only bias is trained on the 1st layer

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        return x


class CNN_MNIST(nn.Module):
    """
        CNN model for MNIST experiment
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, 1 * 28 * 28),
            nn.ReLU(),
            nn.Unflatten(1, (1, 28, 28)),
            nn.Conv2d(1, 96, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(96, 384, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Flatten(),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )
        self.layers[1].weight.data.copy_(torch.eye(1 * 28 * 28))  # initialize 1st layer's weights to identity
        self.layers[1].weight.requires_grad = False  # freeze weights of the first layer. Only bias is trained on the 1st layer

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        return x