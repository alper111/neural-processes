import torch
import models
import utils
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

IN_DIM = 2
HIDDEN_DIM = 256
QUERY_DIM = 1
OUTPUT_DIM = 2

model = models.CNP(
    in_dim=IN_DIM,
    hidden_dim=HIDDEN_DIM,
    query_dim=QUERY_DIM,
    out_dim=OUTPUT_DIM,
    en_layer=2,
    dec_layer=2
)

# model = models.ANP(
#     in_dim=IN_DIM,
#     hidden_dim=HIDDEN_DIM,
#     query_dim=QUERY_DIM,
#     out_dim=OUTPUT_DIM,
#     en_layer=3,
#     dec_layer=3,
#     nhead=32
# )

model.to(device)
model.load_state_dict(torch.load("out/deneme/model.ckpt"))

X = np.load("data/egg_demonstrations.npy")
L = X.shape[1]

with torch.no_grad():
    model.eval().cpu()
    test_set = [(0.35, -0.4), (-0.35, 0.4), (0.3, -0.3), (-0.3, 0.3)]
    for i in range(5):
        # sample = np.random.multivariate_normal(mu_p, cov_p, 1)
        sample = X[np.random.randint(0, X.shape[0])]
        # ind = np.random.randint(0, len(test_set))
        # sample = utils.sample_gaussian(test_set[ind][0], test_set[ind][1], torch.linspace(-0.5, 0.5, L))
        x_t = torch.linspace(0., 1., L).view(L, -1)
        y_t = torch.tensor(sample, dtype=torch.float).view(L, -1)
        xy_t = torch.cat([x_t, y_t], dim=1)
        R = torch.randperm(L)
        context_index = [0, 100, -1]
        out = model(xy_t[context_index], x_t)
        mean = out[:, 0]
        log_std = out[:, 1]
        std = 0.1 + 0.9 * torch.nn.functional.softplus(log_std)
        
        plt.plot(np.linspace(0., 1., L), y_t.squeeze(0).numpy(), c="c")
        for i in range(X.shape[0]):
            plt.plot(np.linspace(0., 1., L), X[i], c="b")
        plt.plot(np.linspace(0, 1, L), mean.numpy(), c='r')
        plt.fill_between(np.linspace(0, 1, L), mean-std, mean+std, facecolor='r', alpha=0.1)
        plt.scatter(xy_t[context_index, 0], xy_t[context_index, 1], marker='x', c='g')
        plt.title("Training samples")
        plt.show()

    for i in range(5):
        # sample = np.random.multivariate_normal(mu_p, cov_p, 1)
        # sample = X[np.random.randint(0, X.shape[0])]
        ind = np.random.randint(0, len(test_set))
        sample = utils.sample_gaussian(test_set[ind][0], test_set[ind][1], torch.linspace(-0.5, 0.5, L))
        x_t = torch.linspace(0., 1., L).view(L, -1)
        y_t = sample.clone().view(L, -1)
        xy_t = torch.cat([x_t, y_t], dim=1)
        R = torch.randperm(L)
        context_index = [0, 100, -1]
        out = model(xy_t[context_index], x_t)
        mean = out[:, 0]
        log_std = out[:, 1]
        std = 0.1 + 0.9 * torch.nn.functional.softplus(log_std)
        
        plt.plot(np.linspace(0., 1., L), y_t.squeeze(0).numpy(), c="c")
        for i in range(X.shape[0]):
            plt.plot(np.linspace(0., 1., L), X[i], c="b")
        plt.plot(np.linspace(0, 1, L), mean.numpy(), c='r')
        plt.fill_between(np.linspace(0, 1, L), mean-std, mean+std, facecolor='r', alpha=0.1)
        plt.scatter(xy_t[context_index, 0], xy_t[context_index, 1], marker='x', c='g')
        plt.title("Test samples")
        plt.show()