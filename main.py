import os
import torch
import pickle
import joblib
import numpy as np
import torchvision.transforms as transforms
from utils import CustomDataset, CustomCIFAR10, parse_args, get_file_count, corrupt_labels
from models import MLP, CNN
from tqdm import tqdm


args = parse_args()
DATA_DIM = 10
N = args.N
freq = args.freq
device = torch.device("cuda")


def run_mlp():
    if args.load_data:
        # LOAD DATA FROM PICKLES
        x = pickle.load(open('pickles/data/{}_samples_{}_freq_train_data.pickle'.format(N, freq), 'rb'))
        y = pickle.load(open('pickles/data/{}_samples_{}_freq_train_labels.pickle'.format(N, freq), 'rb'))
        test_x = pickle.load(open('pickles/data/{}_freq_test_data.pickle'.format(freq), 'rb'))
        test_y = pickle.load(open('pickles/data/{}_freq_test_labels.pickle'.format(freq), 'rb'))
        rand_matrix = pickle.load(open('pickles/data/{}_{}_projection_matrix'.format(DATA_DIM, DATA_DIM), 'rb'))
    else:
        # CREATE DATA
        sample_x = np.random.uniform(-1, 1, N)
        sample_y = np.random.uniform(-1, 1, N)
        y = np.array([np.cos(2 * np.pi * freq * i) * np.cos(2 * np.pi * freq * j) for i, j in zip(sample_x, sample_y)])

        sample_test_x = np.random.uniform(-1, 1, int(np.floor(N * 0.3)))
        sample_test_y = np.random.uniform(-1, 1, int(np.floor(N * 0.3)))
        test_y = np.array([np.cos(2 * np.pi * freq * i) * np.cos(2 * np.pi * freq * j) for i, j in zip(sample_test_x, sample_test_y)])

        x = np.vstack((sample_x, sample_y)).T
        test_x = np.vstack((sample_test_x, sample_test_y)).T

        # rand_matrix = np.random.randn(DATA_DIM, DATA_DIM) / np.sqrt(DATA_DIM)
        # pickle.dump(rand_matrix, open('pickles/data/{}_{}_projection_matrix'.format(DATA_DIM, DATA_DIM), 'wb'))

        # WRITE DATA
        directory = 'pickles/data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(x, open('{}/{}_samples_{}_freq_train_data.pickle'.format(directory, N, freq), 'wb'))
        pickle.dump(y, open('{}/{}_samples_{}_freq_train_labels.pickle'.format(directory, N, freq), 'wb'))
        pickle.dump(test_x, open('{}/{}_freq_test_data.pickle'.format(directory, freq), 'wb'))
        pickle.dump(test_y, open('{}/{}_freq_test_labels.pickle'.format(directory, freq), 'wb'))

    # PROJECTION
    # Append zeros to expand 2d to DATA_DIM
    zeros_x = np.zeros((x.shape[0], DATA_DIM))
    zeros_x[:, :x.shape[1]] = x
    zeros_test_x = np.zeros((test_x.shape[0], DATA_DIM))
    zeros_test_x[:, :test_x.shape[1]] = test_x
    # get a random unitary matrix
    q, r = np.linalg.qr(rand_matrix, mode='complete')
    # project points on this matrix
    q = q * 10  # change the variance of q
    zeros_x = np.matmul(q, zeros_x.T).T
    zeros_test_x = np.matmul(q, zeros_test_x.T).T
    x = zeros_x
    test_x = zeros_test_x

    # DATASET CREATION
    dataset = CustomDataset(x, y, device)
    test_dataset = CustomDataset(test_x, test_y, device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    # MODEL
    model = MLP(in_dim=DATA_DIM, n=32)

    model.to(device)
    if args.load_model:
        # LOAD MODEL
        model.load_state_dict(torch.load('pickles/experiment2/models/{}_samples_{}_freq_{}_epochs_{}_lr/model_state_{}'.format(N, freq, args.epochs, args.lr, args.epochs-1)))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.MSELoss()

        # TRAIN-TEST
        mean_train_losses = []
        mean_test_losses = []
        avg_errors = []
        avg_test_errors = []

        # things to track for the experiments
        epoch_bias = []  # size = (num_of_epochs, num_of_iters_in_epoch)
        epoch_lr = []
        epoch_error = []
        test_epoch_error = []
        epoch_lipschitz = []

        # get file count. Find the last repeat of the same experiment to append number in the saved file name
        directory = 'pickles/experiment2/metrics/{}_samples_{}_freq_{}_epochs'.format(N, freq, args.epochs)
        file_count = get_file_count(directory, 'train_loss')

        for epoch in range(args.epochs):
            # TRAIN
            model.train()
            train_losses = []
            test_losses = []

            iter_bias = []
            iter_lr = []
            iter_error = []
            test_iter_error = []
            iter_lipschitz = []
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred = model(x_batch)
                y_pred = y_pred.view(-1)

                loss = loss_fn(y_pred, y_batch) / 2
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                error = torch.abs(y_batch - y_pred)

                with torch.no_grad():
                    x = x_batch
                    W_1 = model.layers[0].weight.data.clone()
                    lipschitz = W_1
                    for idx, layer in enumerate(model.layers):
                        if idx % 2 == 0 and idx != 0:  # append activations before relu as threshold is > 0 and relu crops negatives
                            S_l = torch.diag(torch.where(x > 0, 1.0, 0.0).view(-1))
                            W_l = layer.weight.data
                            lipschitz = torch.matmul(W_l, torch.matmul(S_l, lipschitz))  # W_l * S_l * ...
                        x = layer(x)
                    lipschitz = torch.norm(lipschitz)
                # for tracking metrics
                iter_bias.append(model.layers[0].bias.cpu().detach().numpy())
                iter_lr.append(optimizer.param_groups[0]['lr'])
                iter_error.append(error.cpu().detach().numpy()[0])
                iter_lipschitz.append(float(lipschitz.cpu().detach().numpy()))
            epoch_bias.append(iter_bias)
            epoch_lr.append(iter_lr)
            epoch_error.append(iter_error)
            epoch_lipschitz.append(iter_lipschitz)

            # TEST
            model.eval()
            with torch.no_grad():
                for batch_idx, (test_x_batch, test_y_batch) in enumerate(test_dataloader):
                    test_y_pred = model(test_x_batch)
                    test_y_pred = test_y_pred.view(-1)
                    loss = loss_fn(test_y_pred, test_y_batch) / 2
                    test_losses.append(loss.item())

                    test_error = torch.abs(test_y_pred - test_y_batch)
                    test_iter_error.append(test_error.cpu().detach().numpy()[0])
                test_epoch_error.append(test_iter_error)

            mean_train_losses.append(np.mean(train_losses))
            mean_test_losses.append(np.mean(test_losses))

            avg_errors.append(np.mean(epoch_error))
            avg_test_errors.append(np.mean(test_epoch_error))
            print('epoch : {}, train loss : {:.4f}, test loss : {:.4f}, Average Error : {}, Average Test Error : {}'.format(epoch + 1, mean_train_losses[-1], mean_test_losses[-1], avg_errors[-1], avg_test_errors[-1]))

            # SAVE MODEL STATE EVERY 100 EPOCHS
            if epoch % 100 == 0 or (epoch == args.epochs - 1):
                directory = 'pickles/experiment2/models/{}_samples_{}_freq_{}_epochs_{}_lr/{}'.format(N, freq, args.epochs, args.lr, file_count)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save(model.state_dict(), '{}/model_state_{}'.format(directory, epoch))

        directory = 'pickles/experiment2/tracked_items/{}_samples_{}_freq_{}_epochs'.format(N, freq, args.epochs)
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(epoch_bias, open('{}/b1_{}'.format(directory, file_count), 'wb'))
        pickle.dump(epoch_lr, open('{}/lr_{}'.format(directory, file_count), 'wb'))
        pickle.dump(epoch_error, open('{}/error_{}'.format(directory, file_count), 'wb'))
        pickle.dump(epoch_lipschitz, open('{}/lipschitz_{}'.format(directory, file_count), 'wb'))
        directory = 'pickles/experiment2/metrics/{}_samples_{}_freq_{}_epochs'.format(N, freq, args.epochs)
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(mean_train_losses, open('{}/train_loss_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(mean_test_losses, open('{}/test_loss_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(avg_errors, open('{}/avg_error_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(avg_test_errors, open('{}/avg_test_error_{}.pickle'.format(directory, file_count), 'wb'))


def run_cnn():
    # DATASET CREATION
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # select only 2 classes, airplane and dog
    custom_trainset = CustomCIFAR10(root='./data', train=True, download=True, transform=transform, exclude_list=[1, 2, 3, 4, 6, 7, 8, 9])
    custom_testset = CustomCIFAR10(root='./data', train=False, download=True, transform=transform, exclude_list=[1, 2, 3, 4, 6, 7, 8, 9])
    custom_trainset = corrupt_labels(custom_trainset, args.corrupt)  # corrupt data according to a corruption rate
    trainloader = torch.utils.data.DataLoader(custom_trainset, batch_size=args.bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(custom_testset, batch_size=args.bs, shuffle=False)

    # MODEL
    model = CNN()
    model.to(device)
    if args.load_model:
        # LOAD MODEL
        model.load_state_dict(torch.load('cnn_pickles/models/{}_corrupt_{}_bs_{}_epochs_{}_lr/model_state_{}'.format(args.corrupt, args.bs, args.epochs, args.lr, args.epochs - 1)))
    else:
        loss_fn = torch.nn.BCELoss()  # Binary cross entropy
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=10**(-1/args.decay))
        # TRAIN-TEST
        mean_train_losses = []
        mean_test_losses = []
        avg_epsilons = []
        avg_test_epsilons = []
        avg_errors = []
        avg_test_errors = []

        # things to track for the experiments
        epoch_bias = []  # size = (num_of_epochs, num_of_iters_in_epoch)
        epoch_lr = []
        epoch_epsilon = []
        test_epoch_epsilon = []
        epoch_error = []
        test_epoch_error = []

        # get file count. Find the last repeat of the same experiment to append number in the saved file name
        directory = 'cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs_{}_lr'.format(args.corrupt, args.bs, args.epochs, args.lr)
        file_count = get_file_count(directory, 'train_loss')

        epoch_trajectory = []
        epoch_variance = []
        distance = []
        first_bias = 0
        for epoch in range(args.epochs):
            # TRAIN
            model.train()
            train_losses = []
            test_losses = []

            iter_bias = []
            iter_lr = []
            iter_epsilon = []
            test_iter_epsilon = []
            iter_error = []
            test_iter_error = []

            iter_trajectory = 0
            prev_bias = 0
            prev_lr = 0
            prev_epsilon = 0
            for batch_idx, (x_batch, y_batch) in enumerate(trainloader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                y_pred = model(x_batch)
                y_pred = y_pred.view(-1)

                loss = loss_fn(y_pred, y_batch) / 2
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                error = torch.abs(y_pred - y_batch)
                epsilon = torch.abs(1 / 1 - y_pred - y_batch)  # epsilon is not error!

                # metrics tracking. In CNNs these metrics are tracked during training, as the raw bias files are too heavy to save
                bias = model.layers[1].bias.cpu().detach().numpy()
                if epoch == 0 and batch_idx == 0:  # set once for the whole training
                    first_bias = bias
                lr = optimizer.param_groups[0]['lr']
                error = error.cpu().detach().numpy()[0]
                epsilon = epsilon.cpu().detach().numpy()[0]

                if batch_idx > 0:
                    iter_trajectory += np.linalg.norm(bias - prev_bias) / (prev_lr * prev_epsilon)
                prev_bias = bias
                prev_lr = lr
                prev_epsilon = epsilon
                if epoch > 0 or (epoch == 0 and batch_idx > 0):
                    dist = np.linalg.norm((bias - first_bias), axis=0)
                    distance.append(dist)

                iter_bias.append(bias)
                iter_lr.append(lr)
                iter_error.append(error)
                iter_epsilon.append(epsilon)

            scheduler.step()  # apply exponential decay

            epoch_trajectory.append(iter_trajectory)
            np_iter_bias = np.array(iter_bias)
            mean_epoch_bias = np.reshape(np.mean(np_iter_bias, axis=0), (1, -1))  # average per epoch
            variance = np.mean(np.power(np.linalg.norm((np_iter_bias - mean_epoch_bias), axis=1), 2) / np.power(np.array(iter_lr), 2), axis=0)
            epoch_variance.append(variance)
            epoch_bias.append(iter_bias)
            epoch_lr.append(iter_lr)
            epoch_epsilon.append(iter_epsilon)
            epoch_error.append(iter_error)

            # TEST
            model.eval()
            with torch.no_grad():
                for batch_idx, (test_x_batch, test_y_batch) in enumerate(testloader):
                    test_x_batch = test_x_batch.to(device)
                    test_y_batch = test_y_batch.to(device, dtype=torch.float32)
                    test_y_pred = model(test_x_batch)
                    test_y_pred = test_y_pred.view(-1)

                    loss = loss_fn(test_y_pred, test_y_batch) / 2
                    test_losses.append(loss.item())

                    test_error = torch.abs(test_y_pred - test_y_batch)
                    test_epsilon = torch.abs(1 / 1 - test_y_pred - test_y_batch)
                    test_iter_error.append(test_error.cpu().detach().numpy()[0])
                    test_iter_epsilon.append(test_epsilon.cpu().detach().numpy()[0])
                test_epoch_error.append(test_iter_error)
                test_epoch_epsilon.append(test_iter_epsilon)

            mean_train_losses.append(np.mean(train_losses))
            mean_test_losses.append(np.mean(test_losses))
            avg_errors.append(np.mean(epoch_error))
            avg_test_errors.append(np.mean(test_epoch_error))
            avg_epsilons.append(np.mean(epoch_epsilon))
            avg_test_epsilons.append(np.mean(test_epoch_epsilon))
            print('epoch : {}, train loss : {:.4f}, test loss : {:.4f}, Average Error : {}, Average Test Error : {}'.format(epoch + 1, mean_train_losses[-1], mean_test_losses[-1], avg_errors[-1], avg_test_errors[-1]))

            # SAVE MODEL STATE EVERY 10 EPOCHS
            if epoch % 10 == 0 or (epoch == args.epochs - 1):
                directory = 'cnn_pickles/models/{}_corrupt_{}_bs_{}_epochs_{}_lr/{}'.format(args.corrupt, args.bs, args.epochs, args.lr, file_count)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save(model.state_dict(), '{}/model_state_{}'.format(directory, epoch))


        directory = 'cnn_pickles/tracked_items/{}_corrupt_{}_bs_{}_epochs_{}_lr'.format(args.corrupt, args.bs, args.epochs, args.lr)
        if not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(epoch_lr, open('{}/lr_{}'.format(directory, file_count), 'wb'), compress=True)
        joblib.dump(epoch_error, open('{}/error_{}'.format(directory, file_count), 'wb'), compress=True)
        joblib.dump(epoch_epsilon, open('{}/epsilon_{}'.format(directory, file_count), 'wb'), compress=True)
        joblib.dump(epoch_trajectory, open('{}/trajectory_{}'.format(directory, file_count), 'wb'), compress=True)
        joblib.dump(epoch_variance, open('{}/variance_{}'.format(directory, file_count), 'wb'), compress=True)
        joblib.dump(distance, open('{}/distance_{}'.format(directory, file_count), 'wb'), compress=True)
        directory = 'cnn_pickles/metrics/{}_corrupt_{}_bs_{}_epochs_{}_lr'.format(args.corrupt, args.bs, args.epochs, args.lr)
        if not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(mean_train_losses, open('{}/train_loss_{}.pickle'.format(directory, file_count), 'wb'), compress=True)
        joblib.dump(mean_test_losses, open('{}/test_loss_{}.pickle'.format(directory, file_count), 'wb'), compress=True)
        joblib.dump(avg_errors, open('{}/avg_error_{}.pickle'.format(directory, file_count), 'wb'), compress=True)
        joblib.dump(avg_test_errors, open('{}/avg_test_error_{}.pickle'.format(directory, file_count), 'wb'), compress=True)


if __name__ == '__main__':
    for i in tqdm(range(args.repeats)):
        if args.cnn:
            run_cnn()
        else:
            run_mlp()

