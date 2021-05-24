import torch
import torch.nn as nn

# experiment 1
# class MLP(nn.Module):
#     def __init__(self, in_dim, n):
#         super(MLP, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(in_dim, in_dim),
#             # nn.InstanceNorm1d(in_dim),
#             nn.ReLU(),
#             nn.Linear(in_dim, n),
#             nn.ReLU(),
#             nn.Linear(n, n),
#             nn.ReLU(),
#             nn.Linear(n, n),
#             nn.ReLU(),
#             nn.Linear(n, 1)
#         )
#         self.layers[0].weight.data.copy_(torch.eye(in_dim))  # initialize 1st layer's weights to identity
#         self.layers[0].weight.requires_grad = False  # freeze weights of the first layer. Only bias is trained on the 1st layer
#
#     def forward(self, x):
#         for layer in self.layers:
#             if isinstance(layer, torch.nn.InstanceNorm1d):
#                 x = x.view(x.shape[0], 1, x.shape[1])  # batch size, channels, dim
#                 x = layer(x)
#                 x = x.view(x.shape[1], x.shape[2])
#             else:
#                 x = layer(x)
#         return x

# experiment 2
class MLP(nn.Module):
    def __init__(self, in_dim, n):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            # nn.InstanceNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, n),
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
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.layers[1].weight.data.copy_(torch.eye(3 * 32 * 32))  # initialize 1st layer's weights to identity
        self.layers[1].weight.requires_grad = False  # freeze weights of the first layer. Only bias is trained on the 1st layer

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            # print(x.shape)
            if idx == len(self.layers) - 4:  # after convs and flatten, add a residual connection
                # identity = x
                x = layer(x)
                # x += identity
            else:
                x = layer(x)
        return x
