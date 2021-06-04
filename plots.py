import torch
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
import seaborn as sns
from models import MLP2
from utils import smooth, generate_cos_wave, CustomDataset, optimal_x_for_basis_pursuit, saturate
import os
from tqdm import tqdm

plt.rcParams.update({'font.size': 18, 'legend.fontsize': 16, 'lines.linewidth': 2})

colormap = cm.get_cmap('RdYlGn_r', 4)
colors = colormap(range(4))[::-1]


def plot_data(surface, samples, freq, N, type='imshow'):
    colormap = sns.color_palette('RdYlGn_r', n_colors=100)
    colormap = saturate(colormap, 0.2)
    colormap = cm.colors.ListedColormap(sns.color_palette(colormap).as_hex())

    fig = plt.figure()
    if type == 'surface':

        ax = fig.gca(projection='3d')
        ax.plot_surface(surface['x'], surface['y'], surface['f'], rstride=1, cstride=1, alpha=0.3, zorder=0)
        ax.scatter(samples['x'], samples['y'], samples['z'], color='black', s=20, zorder=10)
        ax.set_zlabel('f')
        ax.set_zlim(-1, 1)
        ax.zaxis.set_major_locator(LinearLocator(5))
    else:
        surface['f'] = (surface['f'] - surface['f'].min()) / (surface['f'].max() - surface['f'].min())
        surface['f'] = surface['f'].reshape((N, N))
        ax = fig.add_subplot(111)
        cax = plt.contourf(surface['x'], surface['y'], surface['f'], levels=np.linspace(0, 1, 100), cmap=colormap)
        ax.scatter(samples['x'], samples['y'], color='blue', s=20, zorder=10)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('images/{}/freq_{}_samples_{}.png'.format(type, freq, len(samples['x'])), format='png')
    plt.show()


def plot_metrics(mean_train_losses, mean_test_losses, avg_errors, N, bs):

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    ax1.plot(mean_train_losses, label='train')
    ax1.plot(mean_test_losses, label='test')
    ax1.plot(avg_errors, label='average error')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    plt.title('Metrics during training. Samples: {}, Batch size:{}'.format(N, bs))
    plt.tight_layout()
    plt.show()


def plot_spatial_error_distributon(model, dataset, surface, samples):
    model.eval()
    y_pred = model(dataset.x)
    y_pred = torch.squeeze(y_pred)

    error = torch.abs(dataset.y - y_pred)
    error = error.cpu().detach().numpy()

    coolwarm = plt.cm.get_cmap('coolwarm', len(error))
    colors = coolwarm(np.arange(0, len(error)))
    sorted_idx = np.argsort(error) # sort errors and get indices e.g. if errors was [0.2, 1, 0.5], the result is [0, 2, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(surface['f'], cmap=plt.cm.BrBG, interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])
    for i in range(len(samples['x'])):
        ax.scatter(samples['x'][i], samples['y'][i], color=coolwarm(error[i]), s=40, zorder=10, edgecolors='black')
    # sample_colors = []
    # for i in range(len(samples['x'])):
    #     sample_colors.append(colors[sorted_idx[i]])
    # ax.scatter(samples['x'], samples['y'], color=sample_colors, s=40, zorder=10, edgecolors='black')
    cset = ax.contourf(surface['x'], surface['y'], surface['f'], 100, cmap=plt.cm.BrBG)
    plt.colorbar(cset)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_locator(LinearLocator(5))
    plt.title('Marker color: lower error is more blue-ish and higher more red-ish')
    plt.tight_layout()
    plt.show()


def plot_freq_metrics(N, freqs, epochs):
    dirs = []
    for freq in freqs:
        dirs.append('pickles/experiment1/metrics/{}_samples_{}_freq_{}_epochs'.format(N, freq, epochs))

    train_files = {}
    test_files = {} # key = frequency value = list of files
    train_error_files = {}
    test_error_files = {}
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk("{}".format(dir)))
        train_files[freqs[idx]] = [file for file in files if 'train_loss' in file]
        test_files[freqs[idx]] = [file for file in files if 'test_loss' in file]
        train_error_files[freqs[idx]] = [file for file in files if 'avg_error' in file]
        test_error_files[freqs[idx]] = [file for file in files if 'avg_test_error' in file]

    for freq in train_files.keys(): # each iter different freq
        train_losses = []
        test_losses = []
        train_errors = []
        test_errors = []
        for i in tqdm(range(len(train_files[freq]))):
            train = pickle.load(open('pickles/experiment1/metrics/{}_samples_{}_freq_{}_epochs/train_loss_{}.pickle'.format(N, freq, epochs, i), 'rb'))
            test = pickle.load(open('pickles/experiment1/metrics/{}_samples_{}_freq_{}_epochs/test_loss_{}.pickle'.format(N, freq, epochs, i), 'rb'))
            train_error = pickle.load(open('pickles/experiment1/metrics/{}_samples_{}_freq_{}_epochs/avg_error_{}.pickle'.format(N, freq, epochs, i), 'rb'))
            test_error = pickle.load(open('pickles/experiment1/metrics/{}_samples_{}_freq_{}_epochs/avg_test_error_{}.pickle'.format(N, freq, epochs, i), 'rb'))
            # convert tensors to float
            train = [float(i) for i in train]
            test = [float(i) for i in test]
            train_error = [float(i) for i in train_error]
            test_error = [float(i) for i in test_error]
            # append to iteration lists
            train_losses.append(train)
            test_losses.append(test)
            train_errors.append(train_error)
            test_errors.append(test_error)

        # convert to numpy for ease
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)
        train_errors = np.array(train_errors)
        test_errors = np.array(test_errors)

        # calc stats
        mean_train_losses = np.mean(train_losses, axis=0)
        mean_test_losses = np.mean(test_losses, axis=0)
        mean_train_errors = np.mean(train_errors, axis=0)
        mean_test_errors = np.mean(test_errors, axis=0)
        std_train_losses = np.std(train_losses, axis=0)
        std_test_losses = np.std(test_losses, axis=0)
        std_train_errors = np.std(train_errors, axis=0)
        std_test_errors = np.std(test_errors, axis=0)

        # create the utmost dictionary for each freq
        train_files[freq] = {'mean': mean_train_losses, 'std': std_train_losses}
        test_files[freq] = {'mean': mean_test_losses, 'std': std_test_losses}
        train_error_files[freq] = {'mean': mean_train_errors, 'std': std_train_errors}
        test_error_files[freq] = {'mean': mean_test_errors, 'std': std_test_errors}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, freq in enumerate(list(train_files.keys())[::-1]):
        ax.plot(train_files[freq]['mean'], label='freq={}'.format(freq), color=colors[idx])
        ax.fill_between(np.arange(epochs), train_files[freq]['mean'] - train_files[freq]['std'], train_files[freq]['mean'] + train_files[freq]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train loss')
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.set_ylim(-0.01, 0.175)
    ax.grid(axis='y')
    plt.legend(loc='upper right', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/train_loss.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, freq in enumerate(list(test_files.keys())[::-1]):
        ax.plot(test_files[freq]['mean'], label='freq={}'.format(freq), color=colors[idx])
        ax.fill_between(np.arange(epochs), test_files[freq]['mean'] - test_files[freq]['std'], test_files[freq]['mean'] + test_files[freq]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test loss')
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.set_ylim(-0.01, 0.175)
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/test_loss.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, freq in enumerate(list(train_error_files.keys())[::-1]):
        ax.plot(train_error_files[freq]['mean'], label='freq={}'.format(freq), color=colors[idx])
        ax.fill_between(np.arange(epochs), train_error_files[freq]['mean'] - train_error_files[freq]['std'],
                        train_error_files[freq]['mean'] + train_error_files[freq]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train error')
    ax.set_ylim(0, 0.5)
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/train_error.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, freq in enumerate(list(test_error_files.keys())[::-1]):
        ax.plot(test_error_files[freq]['mean'], label='freq={}'.format(freq), color=colors[idx])
        ax.fill_between(np.arange(epochs), test_error_files[freq]['mean'] - test_error_files[freq]['std'],
                        test_error_files[freq]['mean'] + test_error_files[freq]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test error')
    ax.set_ylim(0, 0.5)
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/test_error.pdf', format='pdf')
    plt.show()
# plot_freq_metrics(100, [0.25, 0.5, 0.75, 1.0], 3000)


def plot_corr_metrics(corrupts, bs, epochs, lr):
    dirs = []
    for corr in corrupts:
        dirs.append('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs_{}_lr'.format(corr, bs, epochs, lr))

    train_files = {}
    test_files = {} # key = corruption value = list of files
    train_error_files = {}
    test_error_files = {}
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk("{}".format(dir)))
        train_files[corrupts[idx]] = [file for file in files if 'train_loss' in file]
        test_files[corrupts[idx]] = [file for file in files if 'test_loss' in file]
        train_error_files[corrupts[idx]] = [file for file in files if 'avg_error' in file]
        test_error_files[corrupts[idx]] = [file for file in files if 'avg_test_error' in file]

    for corr in train_files.keys(): # each iter different corruption
        train_losses = []
        test_losses = []
        train_errors = []
        test_errors = []
        for i in tqdm(range(len(train_files[corr]))):
            train = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs_{}_lr/train_loss_{}.pickle'.format(corr, bs, epochs, lr, i), 'rb'))
            test = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs_{}_lr/test_loss_{}.pickle'.format(corr, bs, epochs, lr, i), 'rb'))
            train_error = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs_{}_lr/avg_error_{}.pickle'.format(corr, bs, epochs, lr, i), 'rb'))
            test_error = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs_{}_lr/avg_test_error_{}.pickle'.format(corr, bs, epochs, lr, i), 'rb'))
            # convert tensors to float
            train = [float(i) for i in train]
            test = [float(i) for i in test]
            train_error = [float(i) for i in train_error]
            test_error = [float(i) for i in test_error]
            # append to iteration lists
            train_losses.append(train)
            test_losses.append(test)
            train_errors.append(train_error)
            test_errors.append(test_error)

        # convert to numpy for ease
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)
        train_errors = np.array(train_errors)
        test_errors = np.array(test_errors)

        # calc stats
        mean_train_losses = np.mean(train_losses, axis=0)
        mean_test_losses = np.mean(test_losses, axis=0)
        mean_train_errors = np.mean(train_errors, axis=0)
        mean_test_errors = np.mean(test_errors, axis=0)
        std_train_losses = np.std(train_losses, axis=0)
        std_test_losses = np.std(test_losses, axis=0)
        std_train_errors = np.std(train_errors, axis=0)
        std_test_errors = np.std(test_errors, axis=0)

        # create the utmost dictionary for each corruption
        train_files[corr] = {'mean': mean_train_losses, 'std': std_train_losses}
        test_files[corr] = {'mean': mean_test_losses, 'std': std_test_losses}
        train_error_files[corr] = {'mean': mean_train_errors, 'std': std_train_errors}
        test_error_files[corr] = {'mean': mean_test_errors, 'std': std_test_errors}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, corr in enumerate(list(train_files.keys())[::-1]):
        ax.plot(train_files[corr]['mean'], label='corruption={}'.format(corr), color=colors[idx])
        ax.fill_between(np.arange(epochs), train_files[corr]['mean'] - train_files[corr]['std'], train_files[corr]['mean'] + train_files[corr]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train loss')
    ax.set_ylim(-0.01, 0.41)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.grid(axis='y')
    plt.legend(loc='upper right', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/train_loss.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, corr in enumerate(list(test_files.keys())[::-1]):
        ax.plot(test_files[corr]['mean'], label='corr={}'.format(corr), color=colors[idx])
        ax.fill_between(np.arange(epochs), test_files[corr]['mean'] - test_files[corr]['std'], test_files[corr]['mean'] + test_files[corr]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test loss')
    ax.set_ylim(-0.01, 0.41)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/test_loss.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, corr in enumerate(list(train_error_files.keys())[::-1]):
        ax.plot(train_error_files[corr]['mean'], label='corr={}'.format(corr), color=colors[idx])
        ax.fill_between(np.arange(epochs), train_error_files[corr]['mean'] - train_error_files[corr]['std'],
                        train_error_files[corr]['mean'] + train_error_files[corr]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train error')
    ax.set_ylim(0, 0.5)
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/train_error.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, corr in enumerate(list(test_error_files.keys())[::-1]):
        ax.plot(test_error_files[corr]['mean'], label='corr={}'.format(corr), color=colors[idx])
        ax.fill_between(np.arange(epochs), test_error_files[corr]['mean'] - test_error_files[corr]['std'],
                        test_error_files[corr]['mean'] + test_error_files[corr]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test error')
    ax.set_ylim(0, 0.5)
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/test_error.pdf', format='pdf')
    plt.show()
# plot_corr_metrics([0.0, 0.2, 0.4, 0.6], 1, 100, 0.0025)


def plot_mnist_metrics(points, bs, epochs, lr):
    dirs = ['mnist_pickles/{}/metrics/mlp_0.0_corrupt_{}_bs_{}_epochs_{}_lr'.format(points, bs, epochs, lr), 'mnist_pickles/{}/metrics/cnn_0.0_corrupt_{}_bs_{}_epochs_{}_lr'.format(points, bs, epochs, lr)]

    train_files = {}
    test_files = {}
    total_trajectories = {}
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk("{}".format(dir)))
        if idx == 0:
            key = 'mlp'
        else:
            key = 'cnn'
        train_files[key] = [file for file in files if 'train_loss' in file]
        test_files[key] = [file for file in files if 'test_loss' in file]
        total_trajectories[key] = [file for file in files if 'trajectory' in file]

    for key in train_files.keys():  # each iter different corruption
        train_losses = []
        test_losses = []
        trajectories = []
        for i in tqdm(range(len(train_files[key]))):
            train = joblib.load(open('mnist_pickles/{}/metrics/{}_0.0_corrupt_{}_bs_{}_epochs_{}_lr/train_loss_{}.pickle'.format(points, key, bs, epochs, lr, i), 'rb'))
            test = joblib.load(open('mnist_pickles/{}/metrics/{}_0.0_corrupt_{}_bs_{}_epochs_{}_lr/test_loss_{}.pickle'.format(points, key, bs, epochs, lr, i), 'rb'))
            trajectory = joblib.load(open('mnist_pickles/{}/tracked_items/{}_0.0_corrupt_{}_bs_{}_epochs_{}_lr/trajectory_{}'.format(points, key, bs, epochs, lr, i), 'rb'))
            # convert tensors to float
            train = [float(i) for i in train]
            test = [float(i) for i in test]
            trajectory = [float(i) for i in trajectory]

            train = np.array(train)
            test = np.array(test)
            trajectory = np.array(trajectory)
            # append to iteration lists
            train_losses.append(train)
            test_losses.append(test)
            trajectories.append(trajectory)

        # convert to numpy for ease
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)
        trajectories = np.array(trajectories)

        # calc stats
        mean_train_losses = np.mean(train_losses, axis=0)
        mean_test_losses = np.mean(test_losses, axis=0)
        mean_trajectories = np.mean(trajectories, axis=0)
        std_train_losses = np.std(train_losses, axis=0)
        std_test_losses = np.std(test_losses, axis=0)
        std_trajectories = np.std(trajectories, axis=0)

        # create the utmost dictionary for each corruption
        train_files[key] = {'mean': mean_train_losses, 'std': std_train_losses}
        test_files[key] = {'mean': mean_test_losses, 'std': std_test_losses}
        total_trajectories[key] = {'mean': mean_trajectories, 'std': std_trajectories}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, key in enumerate(list(train_files.keys())[::-1]):
        ax.plot(train_files[key]['mean'], label='{}'.format(key), color=colors[idx])
        ax.fill_between(np.arange(len(train_files[key]['mean'])), train_files[key]['mean'] - train_files[key]['std'], train_files[key]['mean'] + train_files[key]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Train loss')
    # ax.set_ylim(-0.01, 0.15)
    # ax.set_yticks([0, 0.05, 0.1, 0.15])
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xticklabels([0, 50 * 100, 100 * 100, 150 * 100, 200 * 100])
    ax.grid(axis='y')
    plt.legend(loc='upper right', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_3/train_loss.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, key in enumerate(list(test_files.keys())[::-1]):
        ax.plot(test_files[key]['mean'], label='{}'.format(key), color=colors[idx])
        ax.fill_between(np.arange(len(test_files[key]['mean'])), test_files[key]['mean'] - test_files[key]['std'], test_files[key]['mean'] + test_files[key]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Test loss')
    # ax.set_ylim(-0.01, 0.15)
    # ax.set_yticks([0, 0.05, 0.1, 0.15])
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xticklabels([0, 50 * 100, 100 * 100, 150 * 100, 200 * 100])
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_3/test_loss.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, key in enumerate(list(total_trajectories.keys())[::-1]):
        ax.plot(total_trajectories[key]['mean'], label='{}'.format(key), color=colors[idx])
        ax.fill_between(np.arange(len(total_trajectories[key]['mean'])), total_trajectories[key]['mean'] - total_trajectories[key]['std'], total_trajectories[key]['mean'] + total_trajectories[key]['std'], color=colors[idx], alpha=0.3)
    ax.set_ylabel('Bias trajectory (per 100 iter)')
    ax.set_xlabel('Iteration')
    ax.set_ylim(-0.01, 0.09)
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xticklabels([0, 50 * 100, 100 * 100, 150 * 100, 200 * 100])
    ax.grid(axis='y')
    plt.legend(loc='upper right', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_3/trajectory.pdf', format='pdf')
    plt.show()
# plot_mnist_metrics(100, 1, 100, 0.002)


def plot_batch_metrics(freqs):

    freq_train = {}
    freq_test = {}
    freq_traj = {}
    for freq in freqs:
        train_files = {}
        test_files = {}
        dirs = ['pickles/experiment4/16/metrics/100_samples_{}_freq_3000_epochs'.format(freq), 'pickles/experiment4/32/metrics/100_samples_{}_freq_3000_epochs'.format(freq)]
        for idx, dir in enumerate(dirs):
            path, dirs, files = next(os.walk("{}".format(dir)))
            if idx == 0:
                key = '16'
            else:
                key = '32'
            train_files[key] = [file for file in files if 'train_loss' in file]
            test_files[key] = [file for file in files if 'test_loss' in file]

        bias_files = {}
        lr_files = {}
        error_files = {}
        dirs = ['pickles/experiment4/16/tracked_items/100_samples_{}_freq_3000_epochs'.format(freq), 'pickles/experiment4/32/tracked_items/100_samples_{}_freq_3000_epochs'.format(freq)]
        for idx, dir in enumerate(dirs):
            path, dirs, files = next(os.walk("{}".format(dir)))
            if idx == 0:
                key = '16'
            else:
                key = '32'
            bias_files[key] = [file for file in files if 'b1' in file]
            lr_files[key] = [file for file in files if 'lr' in file]
            error_files[key] = [file for file in files if 'error' in file]

        for key in train_files.keys():  # each iter different corruption
            train_losses = []
            test_losses = []
            biases = []
            lrs = []
            errors = []
            for i in tqdm(range(len(train_files[key]))):
                train = pickle.load(open('pickles/experiment4/{}/metrics/100_samples_{}_freq_3000_epochs/train_loss_{}.pickle'.format(key, freq, i), 'rb'))
                test = pickle.load(open('pickles/experiment4/{}/metrics/100_samples_{}_freq_3000_epochs/test_loss_{}.pickle'.format(key, freq, i), 'rb'))
                bias = pickle.load(open('pickles/experiment4/{}/tracked_items/100_samples_{}_freq_3000_epochs/b1_{}'.format(key, freq, i), 'rb'))
                lr = pickle.load(open('pickles/experiment4/{}/tracked_items/100_samples_{}_freq_3000_epochs/lr_{}'.format(key, freq, i), 'rb'))
                error = pickle.load(open('pickles/experiment4/{}/tracked_items/100_samples_{}_freq_3000_epochs/error_{}'.format(key, freq, i), 'rb'))

                # convert tensors to float
                train = [float(i) for i in train]
                test = [float(i) for i in test]

                train = np.array(train)
                test = np.array(test)

                # append to iteration lists
                train_losses.append(train)
                test_losses.append(test)
                biases.append(bias)
                lrs.append(lr)
                errors.append(error)

            # convert to numpy for ease
            train_losses = np.array(train_losses)
            test_losses = np.array(test_losses)
            biases = np.array(biases)
            lrs = np.array(lrs)
            errors = np.array(errors)

            # calc stats
            mean_train_losses = np.mean(train_losses, axis=0)
            mean_test_losses = np.mean(test_losses, axis=0)
            std_train_losses = np.std(train_losses, axis=0)
            std_test_losses = np.std(test_losses, axis=0)
            mean_biases = np.mean(biases, axis=0)
            mean_lrs = np.mean(lrs, axis=0)
            mean_errors = np.mean(errors, axis=0)

            mean_train_losses = np.mean(mean_train_losses.reshape((-1, 100)), axis=1)
            mean_train_losses = mean_train_losses.reshape(-1)
            std_train_losses = np.mean(std_train_losses.reshape((-1, 100)), axis=1)
            std_train_losses = std_train_losses.reshape(-1)
            mean_test_losses = np.mean(mean_test_losses.reshape((-1, 100)), axis=1)
            mean_test_losses = mean_test_losses.reshape(-1)
            std_test_losses = np.mean(std_test_losses.reshape((-1, 100)), axis=1)
            std_test_losses = std_test_losses.reshape(-1)

            # create the utmost dictionary for each corruption
            train_files[key] = {'mean': mean_train_losses, 'std': std_train_losses}
            test_files[key] = {'mean': mean_test_losses, 'std': std_test_losses}
            bias_files[key] = mean_biases
            lr_files[key] = mean_lrs
            error_files[key] = mean_errors

        freq_train[freq] = train_files
        freq_test[freq] = test_files
        # METRIC CALCULATION
        trajectory = {}
        min_iters = 9999999
        for key in bias_files.keys():
            biases = bias_files[key]
            lrs = lr_files[key]
            avg_errors = error_files[key]

            # trajectory
            flat_biases = biases.reshape(-1, biases.shape[2])
            flat_lrs = lrs.reshape(-1)[:-1]  # drop last one as diff calculates N-1 points
            flat_avg_errors = avg_errors.reshape(-1)[:-1]  # same as above
            traj = np.linalg.norm(np.diff(flat_biases, axis=0), axis=1) / (flat_lrs * flat_avg_errors)
            iter_trajectory = np.empty(traj.shape[0] + 1)
            iter_trajectory[:-1] = traj
            iter_trajectory[-1] = traj[-1]  # repeat last entry for the meaningless diff
            trajectory[key] = iter_trajectory

            if min_iters > flat_lrs.shape[0]:
                min_iters = flat_lrs.shape[0]

        for key in bias_files.keys():
            trajectory[key] = trajectory[key][:min_iters]
        freq_traj[freq] = trajectory

    plt.rcParams.update({'legend.fontsize': 12})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    i = 0
    for freq in freqs[::-1]:
        for key in list(train_files.keys())[::-1]:
            ax.plot(freq_train[freq][key]['mean'], label='freq={}, bs={}'.format(freq, key), color=colors[i])
            ax.fill_between(np.arange(len(freq_train[freq][key]['mean'])), freq_train[freq][key]['mean'] - freq_train[freq][key]['std'], freq_train[freq][key]['mean'] + freq_train[freq][key]['std'], color=colors[i], alpha=0.3)
            i += 1
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Train loss')
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax.grid(axis='y')
    plt.legend(loc='upper right', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_4/train_loss.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    i = 0
    for freq in freqs[::-1]:
        for key in list(test_files.keys())[::-1]:
            ax.plot(freq_test[freq][key]['mean'], label='freq={}, bs={}'.format(freq, key), color=colors[i])
            ax.fill_between(np.arange(len(freq_test[freq][key]['mean'])), freq_test[freq][key]['mean'] - freq_test[freq][key]['std'], freq_test[freq][key]['mean'] + freq_test[freq][key]['std'], color=colors[i], alpha=0.3)
            i += 1
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Test loss')
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_4/test_loss.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    i = 0
    for freq in freqs[::-1]:
        for key in list(trajectory.keys())[::-1]:
            smoothed = smooth(freq_traj[freq][key])
            ax.plot(smoothed, label='freq={}, bs={}'.format(freq, key), color=colors[i])
            i += 1
    ax.set_ylabel('Bias trajectory (per 100 iter)')
    ax.set_xlabel('Iteration')
    ax.grid(axis='y')
    plt.legend(loc='upper left', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_4/trajectory.pdf', format='pdf')
    plt.show()
# plot_batch_metrics([0.25, 0.75])


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def plot_trajectory_variance_distance_mlp(N, freqs, epochs):
    """
    for each epoch calculate the pairwise (pairs of iterations) sum of biases according to Theorem 1's formula
    plot these statistics for each frequency
    """
    dirs = []
    for freq in freqs:
        dirs.append('pickles/experiment1/tracked_items/{}_samples_{}_freq_{}_epochs'.format(N, freq, epochs))

    raw_bias = {}
    raw_lr = {}

    bias_files = {}
    lr_files = {}  # key = frequency value = list of files
    error_files = {}
    lipschitz_files = {}
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk("{}".format(dir)))
        bias_files[freqs[idx]] = [file for file in files if 'b1' in file]
        lr_files[freqs[idx]] = [file for file in files if 'lr' in file]
        error_files[freqs[idx]] = [file for file in files if 'error' in file]
        lipschitz_files[freqs[idx]] = [file for file in files if 'lipschitz' in file]

    for freq in bias_files.keys():  # each iter different freq
        biases = []
        lrs = []
        errors = []
        lipschitzs = []
        for i in tqdm(range(len(bias_files[freq]))):
            bias = pickle.load(open('pickles/experiment1/tracked_items/{}_samples_{}_freq_{}_epochs/b1_{}'.format(N, freq, epochs, i), 'rb'))
            lr = pickle.load(open('pickles/experiment1/tracked_items/{}_samples_{}_freq_{}_epochs/lr_{}'.format(N, freq, epochs, i), 'rb'))
            error = pickle.load(open('pickles/experiment1/tracked_items/{}_samples_{}_freq_{}_epochs/error_{}'.format(N, freq, epochs, i), 'rb'))
            lipschitz = pickle.load(open('pickles/experiment1/tracked_items/{}_samples_{}_freq_{}_epochs/lipschitz_{}'.format(N, freq, epochs, i), 'rb'))
            # append to iteration lists
            biases.append(bias)
            lrs.append(lr)
            errors.append(error)
            lipschitzs.append(lipschitz)
        # convert to numpy for ease
        biases = np.array(biases)
        lrs = np.array(lrs)
        errors = np.array(errors)
        lipschitzs = np.array(lipschitzs)

        # calc stats
        mean_biases = np.mean(biases, axis=0)
        mean_lrs = np.mean(lrs, axis=0)
        mean_errors = np.mean(errors, axis=0)
        mean_lipschitzs = np.mean(lipschitzs, axis=0)

        # create the utmost dictionary for each freq
        bias_files[freq] = mean_biases
        lr_files[freq] = mean_lrs
        error_files[freq] = mean_errors
        lipschitz_files[freq] = mean_lipschitzs

        raw_bias[freq] = biases
        raw_lr[freq] = lrs

    # METRIC CALCULATION
    trajectory = {}
    variance = {}
    distance = {}
    lip_const = {}
    integral_trajectory = {}
    for freq in freqs:
        biases = bias_files[freq]
        lrs = lr_files[freq]
        avg_errors = error_files[freq]

        # trajectory
        flat_biases = biases.reshape(-1, biases.shape[2])
        flat_lrs = lrs.reshape(-1)[:-1]  # drop last one as diff calculates N-1 points
        flat_avg_errors = avg_errors.reshape(-1)[:-1]  # same as above
        traj = np.linalg.norm(np.diff(flat_biases, axis=0), axis=1) / (flat_lrs*flat_avg_errors) / N
        iter_trajectory = np.empty(traj.shape[0] + 1)
        iter_trajectory[:-1] = traj
        iter_trajectory[-1] = traj[-1]  # repeat last entry for the meaningless diff
        epoch_trajectory = iter_trajectory.reshape((-1, N))
        epoch_trajectory = np.sum(epoch_trajectory, axis=1)
        trajectory[freq] = epoch_trajectory

        # integral trajectory
        integrals = []
        for epoch in range(epochs):
            integrals.append(np.sum(iter_trajectory[0:epoch*N]))
        integral_trajectory[freq] = np.array(integrals)

        # variance of corollary 1
        raw_biases = raw_bias[freq]
        raw_lrs = raw_lr[freq]
        mean_raw_biases = np.reshape(np.mean(raw_biases, axis=2), (raw_biases.shape[0], raw_biases.shape[1], 1, -1))  # average per window T
        norm_mean_raw_biases = np.power(np.linalg.norm((raw_biases - mean_raw_biases), axis=3), 2) / np.power(raw_lrs,2)
        freq_variance = np.mean(norm_mean_raw_biases, axis=2)
        freq_variance = freq_variance[:, -10:]  # keep only last k=10 epochs to plot
        freq_variance = np.reshape(freq_variance, -1)  # flatten
        variance[freq] = freq_variance

        # distance of corollary 2
        raw_biases = np.reshape(raw_biases, (raw_biases.shape[0], -1, raw_biases.shape[3]))  # flatten
        first_bias = np.reshape(raw_biases[:, 0], (raw_biases.shape[0], 1, -1))
        freq_distance = np.linalg.norm((raw_biases - first_bias), axis=2)
        freq_distance = freq_distance[:, -10*N:]  # keep only last k=10*N iterations to plot
        freq_distance = np.reshape(freq_distance, -1)  # flatten
        distance[freq] = freq_distance

        # lipschitz constant
        lipschitzs = lipschitz_files[freq]
        lipschitzs = np.reshape(lipschitzs, (-1))
        lip_const[freq] = lipschitzs

    # TRAJECTORY
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    # iterate with reverse order to plot smaller frequencies on top
    for idx, freq in enumerate(list(trajectory.keys())[::-1]):
        ax.plot(trajectory[freq], label='freq={}'.format(freq), color=colors[idx], alpha=0.3)
        smoothed = smooth(trajectory[freq])
        ax.plot(smoothed, color=colors[idx])
    ax.set_ylabel('Bias trajectory length (per epoch)')
    ax.set_xlabel('Epoch')
    ax.grid(axis='y')
    plt.legend(loc='upper left', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/trajectory.pdf', format='pdf')
    plt.show()

    # INTEGRAL TRAJECTORY
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    # iterate with reverse order to plot smaller frequencies on top
    for idx, freq in enumerate(list(integral_trajectory.keys())[::-1]):
        ax.plot(integral_trajectory[freq], label='freq={}'.format(freq), color=colors[idx])
    ax.set_ylabel('Bias trajectory length (total)')
    ax.set_xlabel('Epoch')
    ax.grid(axis='y')
    plt.legend(loc='upper left', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/integral_trajectory.pdf', format='pdf')
    plt.show()

    # VARIANCE
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    vplot = ax.violinplot(variance.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel('Variance of bias (normalized)')
    ax.set_xlabel('Frequency')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(1, len(list(variance.keys())) + 1))
    ax.set_xticklabels(list(variance.keys()))
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/variance.pdf', format='pdf')
    plt.show()

    # DISTANCE
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    vplot = ax.violinplot(distance.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel("Distance to initialization")
    ax.set_xlabel('Frequency')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(np.arange(1, len(list(distance.keys())) + 1))
    ax.set_xticklabels(list(distance.keys()))
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/distance.pdf', format='pdf')
    plt.show()

    # LIPSCHITZ
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, freq in enumerate(list(lip_const.keys())[::-1]):
        ax.plot(lip_const[freq], label='freq= {}'.format(freq), color=colors[idx], alpha=0.3)
        smoothed = smooth(lip_const[freq])
        ax.plot(smoothed, color=colors[idx])
    ax.set_ylabel('Lipschitz constant')
    ax.set_xticks([0, 50000, 100000, 150000, 200000, 250000, 300000])
    ax.set_xticklabels([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax.set_xlabel('Epoch')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/lipschitz.pdf', format='pdf')
    plt.show()
# plot_trajectory_variance_distance_mlp(100, [0.25, 0.5, 0.75, 1.0], 3000)


def plot_trajectory_variance_distance_cnn(N, corrupts, bs, epochs, lr):
    """
    for each epoch calculate the pairwise (pairs of iterations) sum of biases according to Theorem 1's formula
    plot these statistics for each frequency
    """
    dirs = []
    for corrupt in corrupts:
        dirs.append('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs_{}_lr'.format(corrupt, bs, epochs, lr))

    total_trajectories = {}
    total_variances = {}  # key = frequency value = list of files
    total_distances = {}
    total_integrals = {}
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk("{}".format(dir)))
        total_trajectories[corrupts[idx]] = [file for file in files if 'trajectory' in file]
        total_variances[corrupts[idx]] = [file for file in files if 'variance' in file]
        total_distances[corrupts[idx]] = [file for file in files if 'distance' in file]

    for corrupt in corrupts:  # each iter different freq
        trajectories = []
        variances = []
        distances = []
        for i in tqdm(range(len(total_trajectories[corrupt]))):
            trajectory = joblib.load(open('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs_{}_lr/trajectory_{}'.format(corrupt, bs, epochs, lr, i), 'rb'))
            variance = joblib.load(open('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs_{}_lr/variance_{}'.format(corrupt, bs, epochs, lr, i), 'rb'))
            distance = joblib.load(open('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs_{}_lr/distance_{}'.format(corrupt, bs, epochs, lr, i), 'rb'))

            # append to iteration lists
            trajectories.append(trajectory)
            variances.append(variance)
            distances.append(distance)

        trajectories = np.mean(np.array(trajectories), axis=0)
        variances = np.mean(np.array(variances), axis=0)
        distances = np.mean(np.array(distances), axis=0)

        total_trajectories[corrupt] = trajectories / N  # divide by the number of iterations in epoch
        total_variances[corrupt] = variances[-10:]
        total_distances[corrupt] = distances[-10*N:]

        # integral trajectory
        integrals = []
        for epoch in range(epochs):
            integrals.append(np.sum(trajectories[0:epoch]))
        total_integrals[corrupt] = np.array(integrals) / N

    # TRAJECTORY PLOT
    fig, ax = plt.subplots(nrows=1, ncols=1,  figsize=[7, 4.8])
    # iterate with reverse order to plot smaller corruptions on top
    for idx, corrupt in enumerate(corrupts[::-1]):
        ax.plot(list(total_trajectories[corrupt]), label='corruption={}'.format(corrupt), color=colors[idx])
    ax.set_ylabel('Bias trajectory length (per epoch)')
    ax.set_xlabel('Epoch')
    ax.set_ylim(-0.01, 0.71)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.grid(axis='y')
    plt.legend(loc='upper right', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/trajectory.pdf', format='pdf')
    plt.show()

    # TRAJECTORY PLOT
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    # iterate with reverse order to plot smaller corruptions on top
    for idx, corrupt in enumerate(corrupts[::-1]):
        ax.plot(list(total_integrals[corrupt]), label='corruption={}'.format(corrupt), color=colors[idx])
    ax.set_ylabel('Bias trajectory length (total)')
    ax.set_xlabel('Epoch')
    # ax.set_ylim(-0.01, 0.71)
    # ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.grid(axis='y')
    plt.legend(loc='upper left', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/integral_trajectory.pdf', format='pdf')
    plt.show()

    # VARIANCE PLOT
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    vplot = ax.violinplot(total_variances.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel('Variance of bias (normalized)')
    ax.set_xlabel('Corruption rate')
    # ax.set_yscale('log')
    ax.set_xticks(np.arange(1, len(corrupts) + 1))
    ax.set_xticklabels(corrupts)
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/variance.pdf', format='pdf')
    plt.show()

    # DISTANCE PLOT
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    vplot = ax.violinplot(total_distances.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel("Distance to initialization")
    ax.set_xlabel('Corruption rate')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticks(np.arange(1, len(corrupts) + 1))
    ax.set_xticklabels(corrupts)
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/distance.pdf', format='pdf')
    plt.show()
# plot_trajectory_variance_distance_cnn(10000, [0.0, 0.2, 0.4, 0.6], 1, 100, 0.0025)


def plot_separate(corrupts, bs, epochs):
    dirs = []
    for corr in corrupts:
        dirs.append('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs'.format(corr, bs, epochs))

    train_files = {}
    test_files = {} # key = corruption value = list of files
    train_error_files = {}
    test_error_files = {}
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk("{}".format(dir)))
        train_files[corrupts[idx]] = [file for file in files if 'train_loss' in file]
        test_files[corrupts[idx]] = [file for file in files if 'test_loss' in file]
        train_error_files[corrupts[idx]] = [file for file in files if 'avg_error' in file]
        test_error_files[corrupts[idx]] = [file for file in files if 'avg_test_error' in file]

    for corr in train_files.keys(): # each iter different corruption
        train_losses = []
        test_losses = []
        for i in tqdm(range(len(train_files[corr]))):
            train = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs/train_loss_{}.pickle'.format(corr, bs, epochs, i), 'rb'))
            test = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs/test_loss_{}.pickle'.format(corr, bs, epochs, i), 'rb'))

            # convert tensors to float
            train = [float(i) for i in train]
            test = [float(i) for i in test]
            # append to iteration lists
            train_losses.append(train)
            test_losses.append(test)

        # convert to numpy for ease
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        # create the utmost dictionary for each corruption
        train_files[corr] = train_losses
        test_files[corr] = test_losses

    for idx, corr in enumerate(list(train_files.keys())[::-1]):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
        for train_repeat in train_files[corr]:
            ax.plot(train_repeat)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train loss')
        plt.title('Corruption: {}'.format(corr))
        plt.tight_layout()
        plt.show()

    for idx, corr in enumerate(list(test_files.keys())[::-1]):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
        for test_repeat in test_files[corr]:
            ax.plot(test_repeat)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test loss')
        plt.title('Corruption: {}'.format(corr))
        plt.tight_layout()
        plt.show()
# plot_separate([0.0, 0.2, 0.4, 0.6], 1, 100)


def visualize_regions(N, freq):
    device = torch.device("cuda")

    x_train = pickle.load(open('pickles/data/200_samples_{}_freq_train_data.pickle'.format(freq), 'rb'))
    y_train = pickle.load(open('pickles/data/200_samples_{}_freq_train_labels.pickle'.format(freq), 'rb'))

    # generate a grid of high resolution
    x1 = np.linspace(-1, 1, N)
    x2 = np.linspace(-1, 1, N)

    mesh = np.meshgrid(x1, x2)
    xx1, xx2 = mesh

    # calculate labels
    yy = generate_cos_wave(freq, xx1, xx2)
    x_init = np.array(mesh).T.reshape((-1, 2))  # data is (100x100, 2)
    y = yy.reshape((-1))  # labels are (100x100)
    # project data to 10-d space
    rand_matrix = pickle.load(open('pickles/data/10_10_projection_matrix', 'rb'))
    zeros_x = np.zeros((x_init.shape[0], 10))
    zeros_x[:, :x_init.shape[1]] = x_init
    q, r = np.linalg.qr(rand_matrix, mode='complete')
    q = q * 10  # change the variance of q
    zeros_x = np.matmul(q, zeros_x.T).T
    x = zeros_x
    dataset = CustomDataset(x, y, device)

    # use sns cmap as it can be saturated manually and then turn to matplotlib cmap
    colormap = sns.color_palette('RdYlGn_r', n_colors=100)
    colormap = saturate(colormap, 0.2)
    colormap = cm.colors.ListedColormap(sns.color_palette(colormap).as_hex())

    # INPUT PLOT
    # min-max normalization
    y_norm = (y - y.min()) / (y.max() - y.min())
    y_norm = y_norm.reshape((N, N))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4, 4.8])
    cax = plt.contourf(xx1, xx2, y_norm, levels=np.linspace(0, 1, 100), cmap=colormap)
    ax.scatter(x_train[:, 0], x_train[:, 1], marker='.', color='blue')

    # remove ticks and frame. Also use the same x and y scaling
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.axis('equal')
    # adjust colorbar ticks to show unnormalized values
    cbar = fig.colorbar(cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(np.linspace(np.min(y), np.max(y), 5).round(1), fontsize=12)
    plt.savefig('images/experiment_2/ground_truth_{}_grid_{}_freq.png'.format(N, freq), format='png')
    plt.tight_layout()
    plt.show()

    # METRIC EXTRACTION

    model = MLP2(in_dim=10, n=32)
    model.to(device)
    model.load_state_dict(torch.load('pickles/experiment2/models/200_samples_{}_freq_1000_epochs_0.001_lr/0/model_state_999'.format(freq)))

    model.eval()
    # for each datapoint, a different row
    lamda_R_x = []
    outputs = []
    s = []
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in tqdm(enumerate(zip(dataset.x, dataset.y))):
            y_pred = model(x_batch).view(-1)

            x = x_batch
            kron = 1
            lipschitz = model.layers[0].weight.data.clone()
            for idx, layer in enumerate(model.layers):
                if idx % 2 == 0 and idx != 0:  # get activations of dense layers after the initial
                    s_l_x = torch.diag(torch.where(x > 0, 1.0, 0.0).view(-1))
                    kron = np.kron(torch.diagonal(s_l_x).cpu().numpy(), kron).astype(int)
                    W_l = layer.weight.data
                    lipschitz = torch.matmul(W_l, torch.matmul(s_l_x, lipschitz))  # W_l * S_l * ...
                x = layer(x)

            s.append(kron)
            lamda_R_x.append(torch.norm(lipschitz).cpu().numpy())
            outputs.append(y_pred.cpu().numpy())

    comparison = []  # used to assign points to regions
    for s_x in s:
        int_s_x = np.packbits(s_x).tolist()
        int_s_x = ''.join([str(i) for i in int_s_x])
        comparison.append(int_s_x)
    # group points with the same activation pattern. Points = indices
    groups = pd.Series(range(len(comparison))).groupby(comparison, sort=False).apply(list).tolist()


    # LIPSCHITZ PLOT

    lamda_R_x = np.array(lamda_R_x)
    # min-max normalization
    lamda_R_x_norm = (lamda_R_x - lamda_R_x.min()) / (lamda_R_x.max() - lamda_R_x.min())
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4, 4.8])
    for group in tqdm(groups): # one plot for each lipschitz constant
        z = np.zeros_like(lamda_R_x_norm)
        lipschitz = lamda_R_x_norm[group[0]]
        for idx in group:
            z[idx] = lipschitz
        z = z.reshape((N, N))
        cax = ax.contourf(xx1, xx2, z, levels=np.linspace(0.001, 1, 100), cmap=colormap, extend='neither')
        ax.contour(xx1, xx2, z, levels=0, colors=['black'], linewidths=0.5)
    ax.scatter(x_train[:, 0], x_train[:, 1], marker='.', color='blue')

    # remove ticks and frame. Also use the same x and y scaling
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.axis('equal')
    # adjust colorbar ticks to show unnormalized values
    cbar = fig.colorbar(cax, ticks=[0.001, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(np.linspace(np.min(lamda_R_x), np.max(lamda_R_x), 5).round(1), fontsize=12)
    plt.savefig('images/experiment_2/lipschitz_surface_{}_grid_{}_freq.png'.format(N, freq), format='png')
    plt.tight_layout()
    plt.show()

    # OUTPUT PLOT

    outputs = np.array(outputs)
    # min-max normalization
    outputs_norm = (outputs - outputs.min()) / (outputs.max() - outputs.min())
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4, 4.8])
    for group in tqdm(groups):  # one plot for each lipschitz constant
        z = np.zeros_like(outputs_norm)
        for idx in group:
            z[idx] = outputs_norm[idx]
        z = z.reshape((N, N))
        cax = ax.contourf(xx1, xx2, z, levels=np.linspace(0.001, 1, 100), cmap=colormap, extend='neither')
        ax.contour(xx1, xx2, z, levels=0, colors=['black'], linewidths=0.5)
    ax.scatter(x_train[:, 0], x_train[:, 1], marker='.', color='blue')

    # remove ticks and frame. Also use the same x and y scaling
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.axis('equal')
    # adjust colorbar ticks to show unnormalized values
    cbar = fig.colorbar(cax, ticks=[0.001, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(np.linspace(np.min(outputs), np.max(outputs), 5).round(1), fontsize=12)
    plt.savefig('images/experiment_2/output_surface_{}_grid_{}_freq.png'.format(N, freq), format='png')
    plt.tight_layout()
    plt.show()

    # COMBINATION PLOTS

    x = x_train
    y = y_train
    zeros_x = np.zeros((x.shape[0], rand_matrix.shape[0]))
    zeros_x[:, :x.shape[1]] = x
    q, r = np.linalg.qr(rand_matrix, mode='complete')
    q = q * 10  # change the variance of q
    zeros_x = np.matmul(q, zeros_x.T).T
    x = zeros_x
    train_dataset = CustomDataset(x, y, device)

    # create activation pattern matrix S_T

    S_T = np.empty((10 * 32, len(train_dataset.x)))  # 10 = 1st layer dim, 32 = 2nd layer dim
    lamda_R_x_train = np.empty((1, len(train_dataset.x)))
    s_train = []
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in tqdm(enumerate(zip(train_dataset.x, train_dataset.y))):
            x = x_batch
            kron = 1
            lipschitz = model.layers[0].weight.data.clone()
            for idx, layer in enumerate(model.layers):
                if idx % 2 == 0 and idx != 0:  # get activations of dense layers after the initial
                    s_l_x = torch.diag(torch.where(x > 0, 1.0, 0.0).view(-1))
                    kron = np.kron(torch.diagonal(s_l_x).cpu().numpy(), kron).astype(int)
                    W_l = layer.weight.data
                    lipschitz = torch.matmul(W_l, torch.matmul(s_l_x, lipschitz))  # W_l * S_l * ...
                x = layer(x)

            s_train.append(kron)
            lamda_R_x_train[:, batch_idx] = torch.norm(lipschitz).cpu().numpy()
            S_T[:, batch_idx] = kron

    comparison_train = []  # same as grid data. This has 200 values, the number of train points
    for s_x in s_train:
        int_s_x = np.packbits(s_x).tolist()
        int_s_x = ''.join([str(i) for i in int_s_x])
        comparison_train.append(int_s_x)
    res_points = []  # get the grid points that do not belong to any of the training point regions
    for i in range(len(comparison)):
        if comparison[i] not in comparison_train:
            res_points.append(i)

    distances = {}
    qs = {}
    opts = {}
    statuses = {}
    S_T, indices = np.unique(S_T, return_index=True, axis=1)  # keep only unique columns and indices of S_T to solve the linear system
    lamda_R_x_train = lamda_R_x_train[:, indices]  # keep the lipschitz constants of S_T's unique columns
    lamda_R_x_train = lamda_R_x_train.reshape(-1)
    for point in tqdm(res_points):
        s_x = s[point]
        q, opt, status = optimal_x_for_basis_pursuit(S_T, s_x.T, lamda_R_x_train)
        qs.update({point: q})
        opts.update({point: opt})
        statuses.update({point: status})
        k = np.count_nonzero(q)
        distances.update({point: k})
    print('With basis pursuit:{}'.format(np.unique(np.array(list(distances.values())))))

    # # ENABLE THE BELOW IF YOU HAVE SAVED THE LP'S RESULTS, AS IT IS QUITE SLOW
    # qs = pickle.load(open('pickles/lp_data/qs_from_lp_dict.pickle', 'rb'))
    # opts = pickle.load(open('pickles/lp_data/opts_from_lp_dict.pickle', 'rb'))
    # for point in res_points:
    #     q = qs[point]
    #     k = np.count_nonzero(q)
    #     distances.update({point: k})

    # DISTINCTNESS COMBINATION PLOT

    # create colors for each unique distance
    unique_k = np.unique(np.array(list(distances.values())))
    colormap = cm.get_cmap('tab20c', len(unique_k))
    colors = colormap.colors
    colors = np.append(colors, [[1., 1., 1., 1.]], axis=0)  # add red color for regions of training points
    unique_k = np.append(unique_k, -1)  # append dummy value for regions of training points

    # assign a color to each point
    point_colors = []
    for i, point in enumerate(dataset.x):
        if i in res_points:
            k = distances[i]
            idx = np.where(unique_k == k)
            point_colors.append(colors[idx])
        else:  # for training points
            point_colors.append(colors[-1])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4, 4.8])
    levels = np.arange(0, len(colors) + 1)
    for group in tqdm(groups):
        z = np.full(dataset.x.shape[0], -1)
        if group[0] in res_points:
            color_idx = np.where(np.all(colors == point_colors[group[0]], axis=1))[0]  # get index of color
        else:
            color_idx = np.where(np.all(colors == [1., 1., 1., 1.], axis=1))[0]  # distinct for the training points
        for idx in group:
            z[idx] = 1 + color_idx
        z = z.reshape((N, N))
        cax = ax.contourf(xx1, xx2, z, levels=levels, colors=colors, extend='neither')
        ax.contour(xx1, xx2, z, levels=0, colors=['black'], linewidths=0.5)

    # remove ticks and frame. Also use the same x and y scaling
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    unique_k = list(unique_k)
    unique_k[-1] = 'train'
    # adjust colorbar ticks to show unnormalized values
    cbar = fig.colorbar(cax, ticks=np.arange(len(colors)))
    cbar.ax.set_yticklabels(unique_k, fontsize=12)

    plt.axis('equal')
    plt.savefig('images/experiment_2/linear_combinations_distinct_{}_grid_{}_freq.png'.format(N, freq), format='png')
    plt.tight_layout()
    plt.show()

    # GRADIENT COMBINATION PLOT
    # min-max normalize opts dictionary values
    new_to_old_opts = {}  # track the unnormalized values for the colorbar later
    opt_vals = np.array(list(opts.values()))
    for k in opts:
        old = opts[k]
        opts[k] = (opts[k] - opt_vals.min()) / (opt_vals.max() - opt_vals.min())
        new_to_old_opts[opts[k]] = old
    # different levels based on the lp result. Contains infeasible points as well
    distinct_levels = np.unique(np.array(list(opts.values())))
    # create 100 colors from a colormap
    colors = []
    for lv in distinct_levels[1:]:  # select colors based on the actual distances of opt
        colors.append(colormap(lv))
    colors = np.append([[0.2, 0.2, 0.2, 1.]], colors, axis=0)  # add black color for regions of training points
    colors = np.append(colors, [[1., 1., 1., 1.]], axis=0)  # add white color for regions of training points
    distinct_levels = np.append(distinct_levels, -1)  # append dummy value for regions of training points

    # assign a color to each point
    point_colors = []
    for i, point in enumerate(dataset.x):
        point = point.cpu().numpy()
        if i in res_points:
            k = opts[i]
            idx = np.where(distinct_levels == k)
            point_colors.append(colors[idx])
        else:  # for training points
            point_colors.append(colors[-1])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4, 4.8])
    for group in tqdm(groups):
        z = np.full(dataset.x.shape[0], -1)
        if group[0] in res_points:
            opt = opts[group[0]]
            i = np.where(distinct_levels == opt)[0]
        else:
            i = distinct_levels.shape[0] - 1
        for idx in group:
            z[idx] = 1 + i
        z = z.reshape((N, N))
        cax = ax.contourf(xx1, xx2, z, levels=np.arange(0, len(distinct_levels)+1), colors=colors, extend='neither')
        ax.contour(xx1, xx2, z, levels=0, colors=['black'], linewidths=0.5)

    # remove ticks and frame. Also use the same x and y scaling
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # adjust colorbar ticks to show unnormalized values
    for idx, v in enumerate(distinct_levels[1:-1]):
        distinct_levels[idx+1] = new_to_old_opts[v] # idx + 1 because it starts from 1
    distinct_levels = np.round(distinct_levels, 2)  # round values for colorbar
    # rounding creates duplicates that overpopulate the colorbar. Find uniques without default sorting
    indexes = np.unique(distinct_levels, return_index=True)[1]
    unique = [distinct_levels[index] for index in sorted(indexes)]
    unique = list(unique)
    unique[0] = 'inf'
    unique[-1] = 'train'

    tick_labels = []
    for idx, val in enumerate(unique[:-1]):
        if idx % 3 == 0:
            tick_labels.append(val)
        else:
            tick_labels.append('')
    tick_labels.append(unique[-1])
    cbar = fig.colorbar(cax, ticks=np.linspace(0, len(distinct_levels), len(tick_labels)))
    cbar.ax.set_yticklabels(tick_labels, fontsize=12)

    plt.axis('equal')
    plt.savefig('images/experiment_2/linear_combinations_gradient_{}_grid_{}_freq.png'.format(N, freq), format='png')
    plt.tight_layout()
    plt.show()
# visualize_regions(200, 0.5)
