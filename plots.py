import torch
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from models import MLP
from utils import smooth
from tqdm import tqdm
import os
import seaborn as sns

plt.rcParams.update({'font.size': 18})

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
        dirs.append('pickles/metrics/{}_samples_{}_freq_{}_epochs'.format(N, freq, epochs))

    train_files = {}
    test_files = {} # key = frequency value = list of files
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk("{}".format(dir)))
        train_files[freqs[idx]] = [file for file in files if 'train_loss' in file]
        test_files[freqs[idx]] = [file for file in files if 'test_loss' in file]

    for freq in train_files.keys(): # each iter different freq
        train_losses = []
        test_losses = []
        for i in range(len(train_files[freq])):
            train = pickle.load(open('pickles/metrics/{}_samples_{}_freq_{}_epochs/train_loss_{}.pickle'.format(N, freq, epochs, i), 'rb'))
            test = pickle.load(open('pickles/metrics/{}_samples_{}_freq_{}_epochs/test_loss_{}.pickle'.format(N, freq, epochs, i), 'rb'))
            # convert tensors to float
            train = [float(i) for i in train]
            test = [float(i) for i in test]
            # append to iteration lists
            train_losses.append(train)
            test_losses.append(test)

        # convert to numpy for ease
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        # calc stats
        mean_train_losses = np.mean(train_losses, axis=0)
        mean_test_losses = np.mean(test_losses, axis=0)
        std_train_losses = np.std(train_losses, axis=0)
        std_test_losses = np.std(test_losses, axis=0)

        # create the utmost dictionary for each freq
        train_files[freq] = {'mean': mean_train_losses, 'std': std_train_losses}
        test_files[freq] = {'mean': mean_test_losses, 'std': std_test_losses}


    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']  # = # distinct frequencies

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig.set_size_inches(12, 5)

    for idx, freq in enumerate(list(train_files.keys())[::-1]):
        ax1.plot(train_files[freq]['mean'], label='f:{}'.format(freq), color=colors[idx])
        ax1.fill_between(np.arange(epochs), train_files[freq]['mean'] - train_files[freq]['std'], train_files[freq]['mean'] + train_files[freq]['std'], color=colors[idx], alpha=0.3)
        ax2.plot(test_files[freq]['mean'], label='f:{}'.format(freq), color=colors[idx])
        ax2.fill_between(np.arange(epochs), test_files[freq]['mean'] - test_files[freq]['std'], test_files[freq]['mean'] + test_files[freq]['std'], color=colors[idx], alpha=0.3)
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    lines, labels = ax2.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fancybox=True, shadow=True)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
# plot_freq_metrics(100, [0.25, 0.5, 0.75, 1.0], 3000)


def plot_corr_metrics(N, corrupts, bs, epochs):
    dirs = []
    for corrupt in corrupts:
        dirs.append('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs'.format(corrupt, bs, epochs))

    train_files = {}
    test_files = {} # key = frequency value = list of files
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk("{}".format(dir)))
        train_files[corrupts[idx]] = [file for file in files if 'train_loss' in file]
        test_files[corrupts[idx]] = [file for file in files if 'test_loss' in file]

    for corrupt in train_files.keys(): # each iter different freq
        train_losses = []
        test_losses = []
        for i in range(len(train_files[corrupt])):
            train = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs/train_loss_{}.pickle'.format(corrupt, bs, epochs, i), 'rb'))
            test = joblib.load(open('cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs/test_loss_{}.pickle'.format(corrupt, bs, epochs, i), 'rb'))
            # convert tensors to float
            train = [float(i) for i in train]
            test = [float(i) for i in test]
            # append to iteration lists
            train_losses.append(train)
            test_losses.append(test)

        # convert to numpy for ease
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        # calc stats
        mean_train_losses = np.mean(train_losses, axis=0)
        mean_test_losses = np.mean(test_losses, axis=0)
        std_train_losses = np.std(train_losses, axis=0)
        std_test_losses = np.std(test_losses, axis=0)

        # create the utmost dictionary for each freq
        train_files[corrupt] = {'mean': mean_train_losses, 'std': std_train_losses}
        test_files[corrupt] = {'mean': mean_test_losses, 'std': std_test_losses}


    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']  # = # distinct frequencies

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig.set_size_inches(12, 5)

    for idx, corrupt in enumerate(list(train_files.keys())[::-1]):
        ax1.plot(train_files[corrupt]['mean'], label='corr:{}'.format(corrupt), color=colors[idx])
        ax1.fill_between(np.arange(epochs), train_files[corrupt]['mean'] - train_files[corrupt]['std'], train_files[corrupt]['mean'] + train_files[corrupt]['std'], color=colors[idx], alpha=0.3)
        ax2.plot(test_files[corrupt]['mean'], label='corr:{}'.format(corrupt), color=colors[idx])
        ax2.fill_between(np.arange(epochs), test_files[corrupt]['mean'] - test_files[corrupt]['std'], test_files[corrupt]['mean'] + test_files[corrupt]['std'], color=colors[idx], alpha=0.3)
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    lines, labels = ax1.get_legend_handles_labels()
    plt.legend(lines, labels, loc='best', ncol=2, frameon=False)
    # plt.yscale('log')
    plt.tight_layout()
    plt.show()
# plot_corr_metrics(100, [0.0, 0.25, 0.5, 0.75], 100, 40)


def plot_trajectory_variance_distance(N, freqs, epochs):
    """
    for each epoch calculate the pairwise (pairs of iterations) sum of biases according to Theorem 1's formula
    plot these statistics for each frequency
    """
    dirs = []
    for freq in freqs:
        dirs.append('pickles/tracked_items/{}_samples_{}_freq_{}_epochs'.format(N, freq, epochs))
    bias_files = {}
    lr_files = {}  # key = frequency value = list of files
    error_files = {}
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk("{}".format(dir)))
        bias_files[freqs[idx]] = [file for file in files if 'b1' in file]
        lr_files[freqs[idx]] = [file for file in files if 'lr' in file]
        error_files[freqs[idx]] = [file for file in files if 'error' in file]

    for freq in bias_files.keys():  # each iter different freq
        biases = []
        lrs = []
        errors = []
        for i in range(len(bias_files[freq])):
            bias = pickle.load(open('pickles/tracked_items/{}_samples_{}_freq_{}_epochs/b1_{}'.format(N, freq, epochs, i), 'rb'))
            lr = pickle.load(open('pickles/tracked_items/{}_samples_{}_freq_{}_epochs/lr_{}'.format(N, freq, epochs, i), 'rb'))
            error = pickle.load(open('pickles/tracked_items/{}_samples_{}_freq_{}_epochs/error_{}'.format(N, freq, epochs, i), 'rb'))

            # append to iteration lists
            biases.append(bias)
            lrs.append(lr)
            errors.append(error)

        # convert to numpy for ease
        biases = np.array(biases)
        lrs = np.array(lrs)
        errors = np.array(errors)

        # calc stats
        mean_biases = np.mean(biases, axis=0)
        mean_lrs = np.mean(lrs, axis=0)
        mean_errors = np.mean(errors, axis=0)
        std_biases = np.std(biases, axis=0)
        std_lrs = np.std(lrs, axis=0)
        std_errors = np.std(errors, axis=0)

        # create the utmost dictionary for each freq
        bias_files[freq] = {'mean': mean_biases, 'std': std_biases}
        lr_files[freq] = {'mean': mean_lrs, 'std': std_lrs}
        error_files[freq] = {'mean': mean_errors, 'std': std_errors}

    # METRIC CALCULATION
    trajectory = {}
    variance = {}
    std_variance = {}
    distance = {}
    std_distance = {}
    for freq in freqs:
        biases = bias_files[freq]['mean']
        std_biases = bias_files[freq]['std']
        lrs = lr_files[freq]['mean']
        avg_errors = error_files[freq]['mean']

        epoch_trajectory = []
        for epoch in range(epochs):
            iter_trajectory = 0
            for i in range(len(biases[epoch]) - 1):
                iter_trajectory += np.linalg.norm((biases[epoch][i+1] - biases[epoch][i])/(lrs[epoch][i]*avg_errors[epoch][i])) / len(biases[epoch])
            epoch_trajectory.append(iter_trajectory)
        trajectory[freq] = epoch_trajectory

        flatten_bias = np.array([iteration for epoch in biases for iteration in epoch][-10*N:])
        flatten_std_bias = np.array([iteration for epoch in std_biases for iteration in epoch][-10*N:])

        freq_variance = np.power(np.linalg.norm((flatten_bias - np.mean(flatten_bias)), axis=1), 2)
        std_freq_variance = np.power(np.linalg.norm((flatten_std_bias - np.mean(flatten_std_bias)), axis=1), 2)
        freq_distance = np.linalg.norm((flatten_bias - flatten_bias[0]), axis=1)
        std_freq_distance = np.linalg.norm((flatten_std_bias - flatten_std_bias[0]), axis=1)

        variance[freq] = np.array(freq_variance)
        std_variance[freq] = np.array(std_freq_variance)
        distance[freq] = freq_distance
        std_distance[freq] = std_freq_distance

    # PLOTS
    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']

    # iterate with reverse order to plot smaller frequencies on top
    for idx, freq in enumerate(list(trajectory.keys())[::-1]):
        plt.plot(trajectory[freq], label='Freq: {}'.format(freq), color=colors[idx], alpha=0.3)
        smoothed = smooth(trajectory[freq])
        plt.plot(smoothed, color=colors[idx])
    plt.ylabel("SGD Trajectory")
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('images/experiment_1/trajectory.pdf', format='pdf')
    plt.show()

    for idx, freq in enumerate(list(trajectory.keys())[::-1]):
        plt.plot(variance[freq], label='Freq: {}'.format(freq), color=colors[idx], alpha=0.7)
        plt.fill_between(np.arange(len(variance[freq])), variance[freq] - std_variance[freq], variance[freq] + std_variance[freq], color=colors[idx], alpha=0.3)
    plt.ylabel("Bias Variance")
    plt.xlabel('Epoch')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    plt.tight_layout()
    plt.savefig('images/experiment_1/variance.pdf', format='pdf')
    plt.show()

    for idx, freq in enumerate(list(trajectory.keys())[::-1]):
        plt.plot(distance[freq], label='Freq: {}'.format(freq), color=colors[idx], alpha=0.7)
        plt.fill_between(np.arange(len(distance[freq])),  distance[freq] - std_distance[freq], distance[freq] + std_distance[freq], color=colors[idx], alpha=0.3)
    plt.ylabel("Distance to Initialization")
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig('images/experiment_1/distance.pdf', format='pdf')
    plt.show()
# plot_trajectory_variance_distance(100, [0.25, 0.5, 0.75, 1.0], 3000)


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def plot_trajectory_variance_distance_alt(N, freqs, epochs):
    """
    for each epoch calculate the pairwise (pairs of iterations) sum of biases according to Theorem 1's formula
    plot these statistics for each frequency
    """
    dirs = []
    for freq in freqs:
        dirs.append('pickles/tracked_items/{}_samples_{}_freq_{}_epochs'.format(N, freq, epochs))

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
        for i in range(len(bias_files[freq])):
            bias = pickle.load(open('pickles/tracked_items/{}_samples_{}_freq_{}_epochs/b1_{}'.format(N, freq, epochs, i), 'rb'))
            lr = pickle.load(open('pickles/tracked_items/{}_samples_{}_freq_{}_epochs/lr_{}'.format(N, freq, epochs, i), 'rb'))
            error = pickle.load(open('pickles/tracked_items/{}_samples_{}_freq_{}_epochs/error_{}'.format(N, freq, epochs, i), 'rb'))
            lipschitz = pickle.load(open('pickles/tracked_items/{}_samples_{}_freq_{}_epochs/lipschitz_{}'.format(N, freq, epochs, i), 'rb'))
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
                iter_trajectory += np.linalg.norm((biases[epoch][i+1] - biases[epoch][i])/(lrs[epoch][i]*avg_errors[epoch][i])) / len(biases[epoch])
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
    ax.set_ylabel('SGD Trajectory')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    plt.legend(loc='lower right', ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_1/trajectory.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])

    vplot = ax.violinplot(variance.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel('Normalized Bias Variance')
    ax.set_xlabel('Frequency')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(1, len(list(variance.keys())) + 1))
    ax.set_xticklabels(list(variance.keys()))
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    plt.tight_layout()
    plt.savefig('images/experiment_1/variance.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    vplot = ax.violinplot(distance.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel("Distance to Initialization")
    ax.set_xlabel('Frequency')
    ax.set_xticks(np.arange(1, len(list(distance.keys())) + 1))
    ax.set_xticklabels(list(distance.keys()))
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    plt.tight_layout()
    plt.savefig('images/experiment_1/distance.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, freq in enumerate(list(lip_const.keys())[::-1]):
        ax.plot(lip_const[freq], label='Freq: {}'.format(freq), color=colors[idx], alpha=0.3)
        smoothed = smooth(lip_const[freq])
        ax.plot(smoothed, color=colors[idx])
    ax.set_ylabel('Lipschitz Constant')
    ax.set_xticks([0,50000, 100000, 150000, 200000, 250000, 300000])
    ax.set_xticklabels([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('images/experiment_1/lipschitz.pdf', format='pdf')
    plt.show()
# plot_trajectory_variance_distance_alt(100, [0.25, 0.5, 0.75, 1.0], 3000)


def plot_trajectory_variance_distance_cnn(N, corrupts, bs, epochs):
    """
    for each epoch calculate the pairwise (pairs of iterations) sum of biases according to Theorem 1's formula
    plot these statistics for each frequency
    """
    dirs = []
    for corrupt in corrupts:
        dirs.append('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs'.format(corrupt, bs, epochs))

    raw_bias = {}
    raw_lr = {}

    bias_files = {}
    lr_files = {}  # key = frequency value = list of files
    error_files = {}
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk("{}".format(dir)))
        bias_files[corrupts[idx]] = [file for file in files if 'b1' in file]
        lr_files[corrupts[idx]] = [file for file in files if 'lr' in file]
        error_files[corrupts[idx]] = [file for file in files if 'error' in file]

    for corrupt in bias_files.keys():  # each iter different freq
        biases = []
        lrs = []
        errors = []
        for i in range(len(bias_files[corrupt])):
            bias = joblib.load(open('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs/b1_{}'.format(corrupt, bs, epochs, i), 'rb'))
            lr = joblib.load(open('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs/lr_{}'.format(corrupt, bs, epochs, i), 'rb'))
            error = joblib.load(open('cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs/error_{}'.format(corrupt, bs, epochs, i), 'rb'))

            # append to iteration lists
            biases.append(bias)
            lrs.append(lr)
            errors.append(error)
        # convert to numpy for ease
        biases = np.array(biases)
        lrs = np.array(lrs)
        errors = np.array(errors)

        # calc stats
        mean_biases = np.mean(biases, axis=0)
        mean_lrs = np.mean(lrs, axis=0)
        mean_errors = np.mean(errors, axis=0)

        # create the utmost dictionary for each freq
        bias_files[corrupt] = mean_biases
        lr_files[corrupt] = mean_lrs
        error_files[corrupt] = mean_errors

        raw_bias[corrupt] = biases
        raw_lr[corrupt] = lrs

    # METRIC CALCULATION
    trajectory = {}
    variance = {}
    distance = {}
    for corrupt in corrupts:
        biases = bias_files[corrupt]
        lrs = lr_files[corrupt]
        avg_errors = error_files[corrupt]

        epoch_trajectory = []
        for epoch in range(epochs):
            iter_trajectory = 0
            for i in range(len(biases[epoch]) - 1):
                iter_trajectory += np.linalg.norm((biases[epoch][i+1] - biases[epoch][i])/(lrs[epoch][i]*avg_errors[epoch][i])) / len(biases[epoch])
            epoch_trajectory.append(iter_trajectory)
        trajectory[corrupt] = epoch_trajectory

        # variance of corollary 1
        raw_biases = raw_bias[corrupt]
        raw_lrs = raw_lr[corrupt]
        mean_raw_biases = np.reshape(np.mean(raw_biases, axis=2), (raw_biases.shape[0], raw_biases.shape[1], 1, -1))  # average per window T
        norm_mean_raw_biases = np.power(np.linalg.norm((raw_biases - mean_raw_biases), axis=3), 2) / np.power(raw_lrs,2)
        corrupt_variance = np.mean(norm_mean_raw_biases, axis=2)
        corrupt_variance = corrupt_variance[:, -10:]  # keep only last k=10 epochs to plot
        corrupt_variance = np.reshape(corrupt_variance, -1)  # flatten
        variance[corrupt] = corrupt_variance

        # distance of corollary 2
        raw_biases = np.reshape(raw_biases, (raw_biases.shape[0], -1, raw_biases.shape[3]))  # flatten
        first_bias = np.reshape(raw_biases[:, 0], (raw_biases.shape[0], 1, -1))
        corrupt_distance = np.linalg.norm((raw_biases - first_bias), axis=2)
        corrupt_distance = corrupt_distance[:, -10*N:]  # keep only last k=10*N iterations to plot
        corrupt_distance = np.reshape(corrupt_distance, -1)  # flatten
        distance[corrupt] = corrupt_distance

        # PLOTS
    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])

    # iterate with reverse order to plot smaller frequencies on top
    for idx, corrupt in enumerate(list(trajectory.keys())[::-1]):
        # ax.plot(trajectory[corrupt], label='Corruption: {}'.format(corrupt), color=colors[idx], alpha=0.3)
        ax.plot(trajectory[corrupt], label='Corr: {}'.format(corrupt), color=colors[idx])
        # smoothed = smooth(trajectory[corrupt])
        # ax.plot(smoothed, color=colors[idx])
    ax.set_ylabel('SGD Trajectory')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    plt.legend(loc='lower right', ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig('images/experiment_1/trajectory.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])

    vplot = ax.violinplot(variance.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel('Normalized Bias Variance')
    ax.set_xlabel('Corruption Rate')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(1, len(list(variance.keys())) + 1))
    ax.set_xticklabels(list(variance.keys()))
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    plt.tight_layout()
    plt.savefig('images/experiment_1/variance.pdf', format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    vplot = ax.violinplot(distance.values(), showmeans=False, showmedians=True, showextrema=False)
    ax.set_ylabel("Distance to Initialization")
    ax.set_xlabel('Corruption Rate')
    ax.set_xticks(np.arange(1, len(list(distance.keys())) + 1))
    ax.set_xticklabels(list(distance.keys()))
    for patch, color in zip(vplot['bodies'], colors[::-1]):
        patch.set_color(color)
    vplot['cmedians'].set_color('black')
    plt.tight_layout()
    plt.savefig('images/experiment_1/distance.pdf', format='pdf')
    plt.show()
# plot_trajectory_variance_distance_cnn(10000, [0.0, 0.25, 0.5, 0.75], 100, 40)
