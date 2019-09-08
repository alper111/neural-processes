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
    model = models.ANP(
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

X = np.load("data/egg_demonstrations.npy")
L = X.shape[1]

with torch.no_grad():
    model.eval().cpu()
    test_set = [(0.35, -0.5), (-0.35, 0.5), (0.3, -0.55), (-0.3, 0.55)]
    for i in range(5):
        x_t = torch.linspace(0., 1., L).view(L, -1)
        sample = X[np.random.randint(0, X.shape[0])]
        y_t = torch.tensor(sample, dtype=torch.float).view(L, -1)
        # x_t = torch.linspace(-5., 5., L).view(L, -1)
        # y_t = utils.sample_tanh(x_t)
        xy_t = torch.cat([x_t, y_t], dim=1)
        R = torch.randperm(L)
        context_index = np.random.permutation(200)[:3]
        out = model(context=xy_t[context_index], key=x_t[context_index], query=x_t)
        mean = out[:, 0]
        log_std = out[:, 1]
        std = torch.nn.functional.softplus(log_std)
        
        for i in range(X.shape[0]):
            plt.plot(np.linspace(0., 1., L), X[i], c="b")
        plt.plot(np.linspace(0, 1, L), y_t.squeeze(0).numpy(), c="c")
        plt.plot(np.linspace(0, 1, L), mean.numpy(), c='r')
        plt.fill_between(np.linspace(0, 1, L), mean-std, mean+std, facecolor='r', alpha=0.1)
        plt.scatter(xy_t[context_index, 0], xy_t[context_index, 1], marker='x', c='g')
        plt.title("Training samples")
        plt.show()

    for i in range(5):
        ind = np.random.randint(0, len(test_set))
        x_t = torch.linspace(0., 1., L).view(L, -1)
        y_t = utils.sample_gaussian(test_set[ind][0], test_set[ind][1], torch.linspace(-0.5, 0.5, L)).view(L, -1)
        # x_t = torch.linspace(-5., 5., L).view(L, -1)
        # y_t = utils.sample_tanh(x_t)
        xy_t = torch.cat([x_t, y_t], dim=1)
        R = torch.randperm(L)
        context_index = [0, 100, -1]    
        
        out = model(context=xy_t[context_index], key=x_t[context_index], query=x_t)
        mean = out[:, 0]
        log_std = out[:, 1]
        std = torch.nn.functional.softplus(log_std)
        
        for i in range(X.shape[0]):
            plt.plot(np.linspace(0., 1., L), X[i], c="b")
        plt.plot(np.linspace(0., 1., L), y_t.squeeze(0).numpy(), c="c")
        plt.plot(np.linspace(0, 1, L), mean.numpy(), c='r')
        plt.fill_between(np.linspace(0, 1, L), mean-std, mean+std, facecolor='r', alpha=0.1)
        plt.scatter(xy_t[context_index, 0], xy_t[context_index, 1], marker='x', c='g')
        plt.title("Test samples")
        plt.show()

    for i in range(5):
        ind = np.random.randint(0, len(test_set))
        x_t = torch.linspace(0., 1., L).view(L, -1)
        sample = X[np.random.randint(0, X.shape[0])]
        y_t = torch.tensor(sample, dtype=torch.float).view(L, -1)
        # y_t = utils.sample_gaussian(test_set[ind][0], test_set[ind][1], torch.linspace(-0.5, 0.5, L)).view(L, -1)
        # x_t = torch.linspace(-5., 5., L).view(L, -1)
        # y_t = utils.sample_tanh(x_t)
        xy_t = torch.cat([x_t, y_t], dim=1)
        R = torch.randperm(L)
        
        out = model(context=xy_t, key=x_t, query=x_t)
        mean = out[:, 0]
        log_std = out[:, 1]
        std = torch.nn.functional.softplus(log_std)
        
        for i in range(X.shape[0]):
            plt.plot(np.linspace(0., 1., L), X[i], c="b")
        plt.plot(np.linspace(0., 1., L), y_t.squeeze(0).numpy(), c="c")
        plt.plot(np.linspace(0, 1, L), mean.numpy(), c='r')
        plt.fill_between(np.linspace(0, 1, L), mean-std, mean+std, facecolor='r', alpha=0.1)
        # plt.scatter(xy_t[context_index, 0], xy_t[context_index, 1], marker='x', c='g')
        plt.title("All context")
        plt.show()
