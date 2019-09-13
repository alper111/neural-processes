import argparse
import torch
import models
import utils
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="test a neural process.")
parser.add_argument("-in_dim", help="input dimension.", type=int, required=True)
parser.add_argument("-hidden_dim", help="hidden dimension.", type=int, required=True)
parser.add_argument("-query_dim", help="query dimension.", type=int, required=True)
parser.add_argument("-out_dim", help="output dimension.", type=int, required=True)
parser.add_argument("-en_layer", help="number of encoder layers.", type=int, required=True)
parser.add_argument("-de_layer", help="number of decoder layers.", type=int, required=True)
parser.add_argument("-nhead", help="number of attention heads.", type=int)
parser.add_argument("-out", help="output path.", type=str, required=True)
parser.add_argument("-att", help="whether to use self-attention.", type=int, default=1)

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if args.att == 1:
    model = models.ANPv2(
    in_dim=args.in_dim,
    hidden_dim=args.hidden_dim,
    query_dim=args.query_dim,
    out_dim=args.out_dim,
    en_layer=args.en_layer,
    dec_layer=args.de_layer,
    nhead=args.nhead
    )
else:
    model = models.CNP(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        query_dim=args.query_dim,
        out_dim=args.out_dim,
        en_layer=args.en_layer,
        dec_layer=args.de_layer
    )


model.to(device)
model.load_state_dict(torch.load(args.out))

# X = np.load("data/egg_demonstrations.npy")
X = torch.load("data/taskparam.pth")
query_dims = [0, 7, 8]
target_dims = [1, 2, 3, 4, 5, 6]

with torch.no_grad():
    model.eval().cpu()
    # test_set = [(0.35, -0.4), (-0.35, 0.4), (0.3, -0.35), (-0.3, 0.35)]
    for i in range(5):
        sample = X[np.random.randint(0, X.shape[0])]
        L = (sample[:, 0]-1).abs().argmin() + 1
        sample = sample[:L]
        R = torch.randperm(L)
        # context_index = R[:3]
        context_index = [0]
        out = model(context=sample[context_index], key=sample[context_index][:, query_dims], query=sample[:, query_dims])
        mean = out[:, :6]
        log_std = out[:, 6:]
        std = torch.nn.functional.softplus(log_std)
        
        fig, ax = plt.subplots(1, 6, figsize=(25, 4))
        for s in range(6):
            for j in range(X.shape[0]):
                L = (X[j, :, 0]-1).abs().argmin() + 1
                ax[s].plot(X[j, :L, 0], X[j, :L, target_dims[s]], c="b")
            ax[s].plot(sample[:, 0], sample[:, target_dims[s]], c="c")
            ax[s].plot(sample[:, 0], mean[:, s].numpy(), c='r')
            ax[s].fill_between(sample[:, 0], mean[:, s]-std[:, s], mean[:, s]+std[:, s], facecolor='r', alpha=0.1)
            ax[s].scatter(sample[context_index, 0], sample[context_index, target_dims[s]], marker='x', c='g')
            ax[s].set_title("Training samples")
        plt.show()

    test_set = [(0.5, 0.5), (0.5, 2.), (1., 2.5), (2.5, 3.5), (3., 4.), (3.5, 4.5)]
    for i in range(6):
        sample = X[np.random.randint(0, X.shape[0])]
        L = (sample[:, 0]-1).abs().argmin() + 1
        sample = sample[:L]
        R = torch.randperm(L)
        context_index = [0]
        sample[:, 7] = test_set[i][0]
        sample[:, 8] = test_set[i][1]

        out = model(context=sample[context_index], key=sample[context_index][:, query_dims], query=sample[:, query_dims])
        mean = out[:, :6]
        log_std = out[:, 6:]
        std = torch.nn.functional.softplus(log_std)
        
        fig, ax = plt.subplots(1, 6, figsize=(25, 4))
        for s in range(6):
            for j in range(X.shape[0]):
                L = (X[j, :, 0]-1).abs().argmin() + 1
                ax[s].plot(X[j, :L, 0], X[j, :L, target_dims[s]], c="b")
            # ax[s].plot(sample[:, 0], sample[:, target_dims[s]], c="c")
            ax[s].plot(sample[:, 0], mean[:, s].numpy(), c='r')
            ax[s].fill_between(sample[:, 0], mean[:, s]-std[:, s], mean[:, s]+std[:, s], facecolor='r', alpha=0.1)
            ax[s].scatter(sample[context_index, 0], sample[context_index, target_dims[s]], marker='x', c='g')
            ax[s].set_title("Test samples")
        plt.show()

    # for i in range(5):
    #     ind = np.random.randint(0, len(test_set))
    #     x_t = torch.linspace(0., 1., L).view(L, -1)
    #     y_t = utils.sample_gaussian(test_set[ind][0], test_set[ind][1], torch.linspace(-0.5, 0.5, L)).view(L, -1)
    #     # x_t = torch.linspace(-5., 5., L).view(L, -1)
    #     # y_t = utils.sample_tanh(x_t)
    #     xy_t = torch.cat([x_t, y_t], dim=1)
    #     R = torch.randperm(L)
    #     context_index = [0, 100, -1]    
        
    #     out = model(context=xy_t[context_index], key=x_t[context_index], query=x_t)
    #     mean = out[:, 0]
    #     log_std = out[:, 1]
    #     std = torch.nn.functional.softplus(log_std)
    #     for i in range(X.shape[0]):
    #         plt.plot(X[i, :, 0], X[i, :, 1], c="b")
    #     plt.plot(x_t.squeeze(0).numpy(), y_t.squeeze(0).numpy(), c="c")
    #     plt.plot(x_t.squeeze(0).numpy(), mean.numpy(), c='r')
    #     plt.fill_between(x_t.squeeze(1).numpy(), mean-std, mean+std, facecolor='r', alpha=0.1)
    #     plt.scatter(xy_t[context_index, 0], xy_t[context_index, 1], marker='x', c='g')
    #     plt.title("Test samples")
    #     plt.show()

    # for i in range(5):
    #     ind = np.random.randint(0, len(test_set))
    #     sample = X[np.random.randint(0, X.shape[0])]
    #     out = model(context=sample, key=sample[:, 0], query=sample[:, 0])
    #     mean = out[:, 0]
    #     log_std = out[:, 1]
    #     std = torch.nn.functional.softplus(log_std)
        
    #     for i in range(X.shape[0]):
    #         plt.plot(X[i, :, 0], X[i, :, 1], c="b")
    #     plt.plot(sample[:, 0].numpy(), sample[:, 1].numpy(), c="c")
    #     plt.plot(sample[:, 0].numpy(), mean.numpy(), c='r')
    #     plt.fill_between(sample[:, 0].numpy(), mean-std, mean+std, facecolor='r', alpha=0.1)
    #     plt.title("All context")
    #     plt.show()
