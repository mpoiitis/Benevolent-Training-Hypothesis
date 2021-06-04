import os
import matplotlib.colors as mc
import colorsys
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torchvision
from scipy.optimize import linprog


class CustomDataset(torch.utils.data.Dataset):

  def __init__(self, x, y, device):

    self.x = torch.tensor(x, dtype=torch.float32).to(device)
    self.y = torch.tensor(y, dtype=torch.float32).to(device)

  def __len__(self):
    return len(self.x)  # required

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    sample = (self.x[idx], self.y[idx])
    return sample


class CustomCIFAR10(torchvision.datasets.CIFAR10):
    """
    This dataset keeps only the CIFAR10 classes that are not contained in the exclude_list argument.
    This applies to both data and targets.
    """
    def __init__(self, *args, exclude_list=[], **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)
        classDict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
        if exclude_list == []:
            return
        class_exclude_list = [list(classDict.keys())[list(classDict.values()).index(idx)] for idx in exclude_list]

        labels = np.array(self.targets)
        classes = np.array(self.classes)
        exclude = np.array(exclude_list).reshape(1, -1)
        class_exclude = np.array(class_exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)
        class_mask = ~(classes.reshape(-1, 1) == class_exclude).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()
        self.classes = classes[class_mask].tolist()

        # renumber remaining labels from 0
        mapping = {v: k for k, v in enumerate(set(self.targets))}
        self.targets = [mapping[y] for y in self.targets]


class CustomMNIST(torchvision.datasets.MNIST):
    """
    This dataset keeps only the CIFAR10 classes that are not contained in the exclude_list argument.
    This applies to both data and targets.
    """
    def __init__(self, *args, exclude_list=[], **kwargs):
        super(CustomMNIST, self).__init__(*args, **kwargs)
        classDict = {'0 - zero': 0, '1 - one': 1, '2 - two': 2, '3 - three': 3, '4 - four': 4, '5 - five': 5, '6 - six': 6, '7 - seven': 7, '8 - eight': 8, '9 - nine': 9}
        if exclude_list == []:
            return
        class_exclude_list = [list(classDict.keys())[list(classDict.values()).index(idx)] for idx in exclude_list]

        labels = np.array(self.targets)
        classes = np.array(self.classes)
        exclude = np.array(exclude_list).reshape(1, -1)
        class_exclude = np.array(class_exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)
        class_mask = ~(classes.reshape(-1, 1) == class_exclude).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()
        self.classes = classes[class_mask]

        # renumber remaining labels from 0
        mapping = {v: k for k, v in enumerate(set(self.targets))}
        self.targets = [mapping[y] for y in self.targets]
        self.targets = np.array(self.targets)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('-f', '--freq', type=float, default=0.5, help='Frequency of cos wave')
    parser.add_argument('-N', '--N', type=int, default=100, help='Number of samples')
    parser.add_argument('-e', '--epochs', type=int, default=3000, help='Number of epochs')
    parser.add_argument('-bs', '--bs', type=int, default=1, help='Batch size')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-c', '--corrupt', type=float, default=0, help='Label corruption probability')
    parser.add_argument('-d', '--decay', type=int, default=50, help='Learning rate decay factor. Specifically, in how many epochs to decrease LR by 1 order with exponential decay.')
    parser.add_argument('-ld', '--load-data', action='store_true', default=False, help='If given, loads data from pickles')
    parser.add_argument('-lm', '--load-model', action='store_true', default=False, help='If given, loads model from weight dict')
    parser.add_argument('-cnn', '--cnn', action='store_true', default=False, help='If given, it runs the CNN model')
    parser.add_argument('-r', '--repeats', type=int, default=1, help='Number of repeats')
    args = parser.parse_args()
    return args


def generate_cos_wave(freq, x, y):
    out = np.cos(2 * np.pi * freq * x) * np.cos(2 * np.pi * freq * y)
    return out


def get_file_count(directory, str_to_search):
    """
    Given a specific string (file name), it returns the index of the newest instance
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    path, dirs, files = next(os.walk("{}".format(directory)))
    files = [file for file in files if str_to_search in file]
    file_count = len(files)
    return file_count


def smooth(scalars, weight=0.95):
    """
    Smoothing of a list of values, similar to Tensorboard's smoothing
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def corrupt_labels(dataset, corrupt_prob):
    """
    Randomly corrupt a percentage of labels (set by corrupt_prob).
    Corruption means that a binary label is randomly set to 0 or 1.
    """
    labels = np.array(dataset.targets)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(len(dataset.classes), mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    dataset.targets = labels
    return dataset


def optimal_x_for_basis_pursuit(S_T, s_x, lambda_T_transpose):
    """
    Linear Programming analogue for basis pursuit problem. Adjusted objective λ_T * α is used, according to the original paper
    """
    x_dim, y_dim = S_T.shape[1], s_x.shape[0]
    eye = np.eye(x_dim)

    # original version of basis pursuit uses np.ones(x_dim) instead of c
    obj = np.concatenate([np.zeros(x_dim), lambda_T_transpose])

    lhs_ineq = np.concatenate([np.concatenate([eye, -eye], axis=1), np.concatenate([-eye, -eye], axis=1)], axis=0)
    rhs_ineq = np.zeros(2 * x_dim)

    lhs_eq = np.concatenate([S_T, np.zeros((y_dim, x_dim))], axis=1)
    rhs_eq = s_x

    bnd = [*((None, None) for _ in range(x_dim)), *((0, None) for _ in range(x_dim))]

    res = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method="revised simplex")
    return res.x[:x_dim], res.fun, res.status


def alter(alist, col, factor=1.1):
    tmp = np.array(alist)
    tmp[:, col] = tmp[:, col] * factor
    tmp[tmp > 1] = 1
    tmp[tmp < 0] = 0

    new = []
    for row in tmp.tolist():
        new.append(tuple(row))

    return new


def rgb2hls(alist):
    alist = alist[:]
    for i, row in enumerate(alist):
        hls = colorsys.rgb_to_hls(row[0], row[1], row[2])
        alist[i] = hls
    return alist


def hls2rgb(alist):
    alist = alist[:]
    for i, row in enumerate(alist):
        hls = colorsys.hls_to_rgb(row[0], row[1], row[2])
        alist[i] = hls
    return alist


def saturate(alist, increase=0.2):
    factor = 1 + increase
    hls = rgb2hls(alist)
    new = alter(hls, 2, factor=factor)
    rgb = hls2rgb(new)
    return rgb


def keep_sample(dataset, N):
    class_1_idx = np.argwhere(dataset.targets == 0).reshape(-1)
    class_2_idx = np.argwhere(dataset.targets == 1).reshape(-1)
    class_1_idx = class_1_idx[:N]
    class_2_idx = class_2_idx[:N]
    idx = list(class_1_idx)
    idx.extend(class_2_idx)
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    return dataset
