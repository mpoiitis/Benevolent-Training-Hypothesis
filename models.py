import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, n):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            # nn.InstanceNorm1d(in_dim),
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
            if isinstance(layer, torch.nn.InstanceNorm1d):
                x = x.view(x.shape[0], 1, x.shape[1])  # batch size, channels, dim
                x = layer(x)
                x = x.view(x.shape[1], x.shape[2])
            else:
                x = layer(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 3 * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (3, 32, 32)),
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), # with 2x2 pooling
            nn.ReLU(),
            nn.Linear(120, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )
        self.layers[1].weight.data.copy_(torch.eye(3 * 32 * 32))  # initialize 1st layer's weights to identity
        self.layers[1].weight.requires_grad = False  # freeze weights of the first layer. Only bias is trained on the 1st layer

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            # if idx == len(self.layers) - 2:  # layer after convolutions. x is after relu
            #     x = x.view(x.shape[0], x.shape[1], -1)  # concatenate width and height
            #     x = torch.mean(x, dim=2)  # sum for each channel
            #     x = layer(x)
            # else:
            #     x = layer(x)
            x = layer(x)
        return x
