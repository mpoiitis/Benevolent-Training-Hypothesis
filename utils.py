import os
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torchvision


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
    # def __init__(self, *args, exclude_list=[], **kwargs):
    #     super(CustomCIFAR10, self).__init__(*args, **kwargs)
    #     classDict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    #     if exclude_list == []:
    #         return
    #
    #     labels = np.array(self.targets)
    #     exclude = np.array(exclude_list).reshape(1, -1)
    #     mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)
    #
    #     self.data = np.concatenate((self.data[mask], self.data[~mask]), axis=0)
    #     self.targets = [0 for i in labels[mask]] + [1 for i in labels[~mask]]
    #     self.classes = ['class_1', 'class_2']



def parse_args():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
  parser.add_argument('-f', '--freq', type=float, default=1, help='Frequency of cos wave')
  parser.add_argument('-N', '--N', type=int, default=10, help='Number of samples')
  parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
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
    if not os.path.exists(directory):
        os.makedirs(directory)
    path, dirs, files = next(os.walk("{}".format(directory)))
    files = [file for file in files if str_to_search in file]
    file_count = len(files)
    return file_count


def smooth(scalars, weight=0.95):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def corrupt_labels(dataset, corrupt_prob):

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