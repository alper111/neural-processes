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
HIDDEN_DIM = 128
QUERY_DIM = 1
OUTPUT_DIM = 2
NUM_ITER = 200000

# model = models.CNP(
#     in_dim=IN_DIM,
#     hidden_dim=HIDDEN_DIM,
#     query_dim=QUERY_DIM,
#     out_dim=OUTPUT_DIM,
#     en_layer=4,
#     dec_layer=3
# )

model = models.ANP(
    in_dim=IN_DIM,
    hidden_dim=HIDDEN_DIM,
    query_dim=QUERY_DIM,
    out_dim=OUTPUT_DIM,
    en_layer=4,
    dec_layer=3,
    nhead=8
)

model.to(device)
print(model)

optimizer = torch.optim.Adam(
    lr=1e-4,
    params=model.parameters(),
    betas=(0.9, 0.999),
    amsgrad=True
)

# sample from a GP
x = np.linspace(-5., 5.).reshape(-1, 1)
mu_p = np.zeros(x.shape[0])
cov_p = models.kernel(x, x)

avg_loss = 0.0
for i in range(NUM_ITER):
    optimizer.zero_grad()
    sample = np.random.multivariate_normal(mu_p, cov_p, 1)
    x_t = torch.linspace(-5., 5., 50, device=device).view(1, -1)
    y_t = torch.tensor(sample, dtype=torch.float, device=device)
    xy_t = torch.cat([x_t, y_t], dim=0).t()

    R = torch.randperm(50)
    num_of_context = torch.randint(3, 10, (1,))
    num_of_target = torch.randint(2, 10, (1,))
    context_index = R[:num_of_context]
    # predict both context and target. they say it's better
    target_index = R[:(num_of_context+num_of_target)]

    mu, log_std = model(xy_t, context_index, target_index)
    # instead of predicting \sigma, predict its log for stability. cuz non-positive \sigma is pointless.
    std = 0.1 + 0.9 * torch.nn.functional.softplus(log_std)
    dists = torch.distributions.Normal(mu, std)
    # calculate loss
    loss = -dists.log_prob(xy_t[target_index, 1]).mean()
    loss.backward()
    optimizer.step()
    
    avg_loss += loss.item()
    if (i+1) % 1000 == 0:
        print("iter: %d, loss: %.4f" % (i+1, avg_loss*1e-3))
        avg_loss = 0.0

with torch.no_grad():
    model.eval().cpu()
    for i in range(5):
        sample = np.random.multivariate_normal(mu_p, cov_p, 1)
        x_t = torch.linspace(-5., 5., 50).view(1, -1)
        y_t = torch.tensor(sample, dtype=torch.float)
        xy_t = torch.cat([x_t, y_t], dim=0).t()
        R = torch.randperm(50)
        context_index = R[:5]
        mean, log_std = model(xy_t, context_index, R)
        std = 0.1 + 0.9 * torch.nn.functional.softplus(log_std)
        
        plt.plot(xy_t[:, 0].numpy(), xy_t[:, 1].numpy(), c='c')
        plt.plot(xy_t[:, 0].numpy(), mean.numpy(), c='r')
        plt.fill_between(xy_t[:, 0].numpy(), mean-std, mean+std, facecolor='r', alpha=0.1)
        
        mu_s, cov_s = models.posterior_predictive(x_t.t().numpy(), xy_t[context_index, 0].view(-1,1).numpy() , xy_t[context_index, 1].view(-1,1).numpy(), sigma_f=1.0, sigma_y=0.01)
        mu_s = mu_s.reshape(-1)
        cov_s = np.sqrt(np.diag(cov_s))
        plt.plot(xy_t[:, 0].numpy(), mu_s, c='b')
        plt.fill_between(xy_t[:, 0].numpy(), mu_s-cov_s, mu_s+cov_s, facecolor='b', alpha=0.1)
        plt.scatter(xy_t[context_index, 0], xy_t[context_index, 1], marker='x', c='g')
        plt.show()


torch.save(model.eval().cpu().state_dict(), "model.pth")
