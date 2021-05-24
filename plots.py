import torch
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from models import MLP
from utils import smooth, generate_cos_wave, CustomDataset
import os
from tqdm import tqdm

plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'legend.fontsize': 16})

def plot_data(surface, samples, freq, type='imshow'):
    fig = plt.figure()
    if type == 'surface':

        ax = fig.gca(projection='3d')
        ax.plot_surface(surface['x'], surface['y'], surface['f'], rstride=1, cstride=1, alpha=0.3, zorder=0)
        ax.scatter(samples['x'], samples['y'], samples['z'], color='black', s=20, zorder=10)
        ax.set_zlabel('f')
        ax.set_zlim(-1, 1)
        ax.zaxis.set_major_locator(LinearLocator(5))
    else:
        ax = fig.add_subplot(111)
        ax.imshow(surface['f'], cmap=plt.cm.BrBG, interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])
        ax.scatter(samples['x'], samples['y'], color='black', s=20, zorder=10)
        cset = ax.contourf(surface['x'], surface['y'], surface['f'], 100, cmap=plt.cm.BrBG)
        plt.colorbar(cset)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_locator(LinearLocator(5))
    plt.tight_layout()
    plt.title('f=cos(2π{}x)+cos(2π{}y), num of samples={}'.format(freq, freq, len(samples['x'])))
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

    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']  # = # distinct frequencies

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, freq in enumerate(list(train_files.keys())[::-1]):
        ax.plot(train_files[freq]['mean'], label='freq={}'.format(freq), color=colors[idx])
        ax.fill_between(np.arange(epochs), train_files[freq]['mean'] - train_files[freq]['std'], train_files[freq]['mean'] + train_files[freq]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train loss')
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.set_ylim(-0.01, 0.175)
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
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/test_error.pdf', format='pdf')
    plt.show()
# plot_freq_metrics(100, [0.25, 0.5, 0.75, 1.0], 3000)


def plot_corr_metrics(corrupts, bs, epochs):
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
        train_errors = []
        test_errors = []
        for i in tqdm(range(len(train_files[corr]))):
            train = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs/train_loss_{}.pickle'.format(corr, bs, epochs, i), 'rb'))
            test = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs/test_loss_{}.pickle'.format(corr, bs, epochs, i), 'rb'))
            train_error = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs/avg_error_{}.pickle'.format(corr, bs, epochs, i), 'rb'))
            test_error = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs/avg_test_error_{}.pickle'.format(corr, bs, epochs, i), 'rb'))
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

    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']  # = # distinct frequencies

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, corr in enumerate(list(train_files.keys())[::-1]):
        ax.plot(train_files[corr]['mean'], label='corr={}'.format(corr), color=colors[idx])
        ax.fill_between(np.arange(epochs), train_files[corr]['mean'] - train_files[corr]['std'], train_files[corr]['mean'] + train_files[corr]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train loss')
    ax.set_ylim(0, 0.35)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/train_loss.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, corr in enumerate(list(test_files.keys())[::-1]):
        ax.plot(test_files[corr]['mean'], label='corr={}'.format(corr), color=colors[idx])
        ax.fill_between(np.arange(epochs), test_files[corr]['mean'] - test_files[corr]['std'], test_files[corr]['mean'] + test_files[corr]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test loss')
    ax.set_ylim(0, 0.35)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
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
    ax.set_ylim(0.5, 1.0)
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
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/test_error.pdf', format='pdf')
    plt.show()
# plot_corr_metrics([0.0, 0.2, 0.4, 0.6], 1, 100)


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
    for freq in freqs:
        biases = bias_files[freq]
        lrs = lr_files[freq]
        avg_errors = error_files[freq]

        epoch_trajectory = []
        for epoch in range(epochs):
            iter_trajectory = 0
            for i in range(len(biases[epoch]) - 1):
                iter_trajectory += np.linalg.norm((biases[epoch][i+1] - biases[epoch][i]))/(lrs[epoch][i]*avg_errors[epoch][i]) / len(biases[epoch])
            epoch_trajectory.append(iter_trajectory)
        trajectory[freq] = epoch_trajectory

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

        # PLOTS
    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])

    # iterate with reverse order to plot smaller frequencies on top
    for idx, freq in enumerate(list(trajectory.keys())[::-1]):
        ax.plot(trajectory[freq], label='Freq: {}'.format(freq), color=colors[idx], alpha=0.3)
        smoothed = smooth(trajectory[freq])
        ax.plot(smoothed, color=colors[idx])
    ax.set_ylabel('SGD trajectory')
    ax.set_xlabel('Epoch')
    # ax.set_yscale('log')
    plt.legend(loc='upper left', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/trajectory.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5.8, 4.8])

    vplot = ax.violinplot(variance.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel('Variance of bias (normalized)')
    ax.set_xlabel('Frequency')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(1, len(list(variance.keys())) + 1))
    ax.set_xticklabels(list(variance.keys()))
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/variance.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5.8, 4.8])
    vplot = ax.violinplot(distance.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel("Distance to initialization")
    ax.set_xlabel('Frequency')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(np.arange(1, len(list(distance.keys())) + 1))
    ax.set_xticklabels(list(distance.keys()))
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/distance.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, freq in enumerate(list(lip_const.keys())[::-1]):
        ax.plot(lip_const[freq], label='Freq: {}'.format(freq), color=colors[idx], alpha=0.3)
        smoothed = smooth(lip_const[freq])
        ax.plot(smoothed, color=colors[idx])
    ax.set_ylabel('Lipschitz constant')
    ax.set_xticks([0, 50000, 100000, 150000, 200000, 250000, 300000])
    ax.set_xticklabels([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('images/experiment_1/mlp/lipschitz.pdf', format='pdf')
    plt.show()
# plot_trajectory_variance_distance_mlp(100, [0.25, 0.5, 0.75, 1.0], 3000)


def plot_trajectory_variance_distance_cnn(N, corrupts, bs, epochs):
    """
    for each epoch calculate the pairwise (pairs of iterations) sum of biases according to Theorem 1's formula
    plot these statistics for each frequency
    """
    dirs = []
    for corrupt in corrupts:
        dirs.append('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs'.format(corrupt, bs, epochs))

    total_trajectories = {}
    total_variances = {}  # key = frequency value = list of files
    total_distances = {}
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
            trajectory = joblib.load(open('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs/trajectory_{}'.format(corrupt, bs, epochs, i), 'rb'))
            variance = joblib.load(open('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs/variance_{}'.format(corrupt, bs, epochs, i), 'rb'))
            distance = joblib.load(open('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs/distance_{}'.format(corrupt, bs, epochs, i), 'rb'))

            # append to iteration lists
            trajectories.append(trajectory)
            variances.append(variance)
            distances.append(distance)

        trajectories = np.mean(np.array(trajectories), axis=0)
        variances = np.mean(np.array(variances), axis=0)
        distances = np.mean(np.array(distances), axis=0)

        total_trajectories[corrupt] = trajectories / N  # divide by the number of iterations in epoch
        total_variances[corrupt] = variances
        total_distances[corrupt] = distances

        # PLOTS
    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']

    fig, ax = plt.subplots(nrows=1, ncols=1,  figsize=[7, 4.8])

    # iterate with reverse order to plot smaller corruptions on top
    for idx, corrupt in enumerate(corrupts[::-1]):
        # ax.plot(total_trajectories[corrupt], label='Corr: {}'.format(corrupt), color=colors[idx], alpha=0.3)
        ax.plot(list(total_trajectories[corrupt]), label='Corr: {}'.format(corrupt), color=colors[idx])
        # smoothed = smooth(total_trajectories[corrupt])
        # ax.plot(smoothed, color=colors[idx])
    ax.set_ylabel('SGD trajectory')
    ax.set_xlabel('Epoch')
    # ax.set_yscale('log')
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0.1), ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/trajectory.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5.8, 4.8])

    vplot = ax.violinplot(total_variances.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel('Variance of bias (normalized)')
    ax.set_xlabel('Corruption rate')
    # ax.set_yscale('log')
    ax.set_xticks(np.arange(1, len(corrupts) + 1))
    ax.set_xticklabels(corrupts)
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/variance.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5.8, 4.8])
    vplot = ax.violinplot(total_distances.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel("Distance to initialization")
    ax.set_xlabel('Corruption rate')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(np.arange(1, len(corrupts) + 1))
    ax.set_xticklabels(corrupts)
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    plt.tight_layout()
    plt.savefig('images/experiment_1/cnn/distance.pdf', format='pdf')
    plt.show()
# plot_trajectory_variance_distance_cnn(10000, [0.0, 0.2, 0.4, 0.6], 1, 100)


def visualize_regions(N, freq):
    device = torch.device("cuda")

    # generate a grid of high resolution
    x1 = np.linspace(-1, 1, N)
    x2 = np.linspace(-1, 1, N)

    mesh = np.meshgrid(x1, x2)
    xx1, xx2 = mesh

    # calculate labels
    yy = generate_cos_wave(freq, xx1, xx2)
    x_init = np.array(mesh).T.reshape((-1, 2))  # data is (100x100, 2)
    y = yy.reshape((-1))  # labels are (100x100)

    plt.contourf(xx1, xx2, yy)
    plt.show()

    # project data to 10-d space
    rand_matrix = pickle.load(open('pickles/data/10_10_projection_matrix', 'rb'))

    zeros_x = np.zeros((x_init.shape[0], 10))
    zeros_x[:, :x_init.shape[1]] = x_init
    q, r = np.linalg.qr(rand_matrix, mode='complete')
    q = q * 10  # change the variance of q
    zeros_x = np.matmul(q, zeros_x.T).T
    x = zeros_x

    dataset = CustomDataset(x, y, device)

    model = MLP(in_dim=10, n=32)
    model.to(device)
    model.load_state_dict(torch.load('pickles/experiment2/models/100_samples_{}_freq_1000_epochs_0.001_lr/0/model_state_999'.format(freq)))

    model.eval()
    # for each datapoint, a different row
    S_T = []
    lamda_R_x = []
    outputs = []

    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(zip(dataset.x, dataset.y))):
        y_pred = model(x_batch).view(-1)

        # create activation pattern matrix S_T
        x = x_batch
        W_1 = model.layers[0].weight.data.clone()
        lipschitz = W_1
        S = []
        for idx, layer in enumerate(model.layers):
            if idx % 2 == 0 and idx != 0:  # get activations of dense layers after the initial
                S_l = torch.diag(torch.where(x > 0, 1.0, 0.0).view(-1))
                diag_S_l = torch.diagonal(S_l, 0).cpu().numpy().astype(np.int32)
                # int_S_l = diag_S_l.dot(1 << np.arange(diag_S_l.size)[::-1]) # convert binary vector to integer

                int_S_l = np.packbits(diag_S_l)
                S.append(int_S_l)
                # S.append(S_l)
                W_l = layer.weight.data
                lipschitz = torch.matmul(W_l, torch.matmul(S_l, lipschitz))  # W_l * S_l * ...
            x = layer(x)

        S_T.append(S)
        lamda_R_x.append(torch.norm(lipschitz).cpu().numpy())
        outputs.append(y_pred.cpu().detach().numpy())
    S_T = np.array(S_T)
    for idx, el in enumerate(S_T): # S_T contains integer representations of the activations
        el[0] = ''.join([str(i) for i in el[0]]) # concat representations for the 1st layer
        el[1] = ''.join([str(i) for i in el[1]]) # concat representations for the 2nd layer
        S_T[idx] = int(el[0] + '' + el[1]) # concat representations for both layers
    S_T = S_T[:, 0] # drop 2nd dimension as it is duplicate. S_T contains plain integers now
    lamda_R_x = np.array(lamda_R_x)
    outputs = np.array(outputs)

    # min-max normalization
    lamda_R_x = (lamda_R_x - lamda_R_x.min()) / (lamda_R_x.max() - lamda_R_x.min())
    outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
    # group points with the same activation pattern. Points = indices of the S_T array
    groups = pd.Series(range(len(S_T))).groupby(S_T, sort=False).apply(list).tolist()

    # LIPSCHITZ PLOT
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4, 4.8])
    for group in tqdm(groups): # one plot for each lipschitz constant
        z = np.zeros_like(lamda_R_x)
        lipschitz = lamda_R_x[group[0]]
        for idx in group:
            z[idx] = lipschitz
            # z[idx] = 1
        z = z.reshape((N, N))
        im1 = ax.contourf(xx1, xx2, z, levels=np.linspace(0.001, 1, 50), cmap='coolwarm', extend='max')
        im2 = ax.contour(xx1, xx2, z, levels=0, colors=['black'], linewidths=0.5)
    # ax.scatter(xx1, xx2, marker='.', s=1, edgecolor='black', color='red')
    plt.axis('off')
    plt.savefig('images/experiment_2/lipschitz_surface_{}_grid_{}_freq.pdf'.format(N, freq), format='pdf')
    plt.tight_layout()
    plt.show()

    # OUTPUT PLOT
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4, 4.8])
    for group in tqdm(groups):  # one plot for each lipschitz constant
        z = np.zeros_like(lamda_R_x)
        for idx in group:
            z[idx] = outputs[idx]
            # z[idx] = 1
        z = z.reshape((N, N))
        im1 = ax.contourf(xx1, xx2, z, levels=np.linspace(0.001, 1, 50), cmap='coolwarm', extend='max')
        im2 = ax.contour(xx1, xx2, z, levels=0, colors=['black'], linewidths=0.5)
    # ax.scatter(xx1, xx2, marker='.', s=1, edgecolor='black', color='red')
    plt.axis('off')
    plt.savefig('images/experiment_2/output_surface_{}_grid_{}_freq.pdf'.format(N, freq), format='pdf')
    plt.tight_layout()
    plt.show()
visualize_regions(400, 0.75)

