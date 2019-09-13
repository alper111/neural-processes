import argparse
import os
import time
import torch
import models
import utils
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="train a neural process.")
parser.add_argument("-in_dim", help="input dimension.", type=int, required=True)
parser.add_argument("-hidden_dim", help="hidden dimension.", type=int, required=True)
parser.add_argument("-query_dim", help="query dimension.", type=int, required=True)
parser.add_argument("-out_dim", help="output dimension.", type=int, required=True)
parser.add_argument("-en_layer", help="number of encoder layers.", type=int, required=True)
parser.add_argument("-de_layer", help="number of decoder layers.", type=int, required=True)
parser.add_argument("-nhead", help="number of attention heads.", type=int)
parser.add_argument("-iter", help="number of iterations.", type=int, required=True)
parser.add_argument("-lr", help="learning rate.", type=float, default=1e-3)
parser.add_argument("-out", help="output path.", type=str, required=True)
parser.add_argument("-seed", help="seed number", type=float, default=None)
parser.add_argument("-att", help="whether to use self-attention.", type=int, default=1)

args = parser.parse_args()

ATT = True if args.att == 1 else False

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if args.seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)

if not os.path.exists(args.out):
    os.makedirs(args.out)
os.chdir(args.out)
arg_dict = vars(args)
for key in arg_dict.keys():
    print("%s: %s" % (key, arg_dict[key]))
    print("%s: %s" % (key, arg_dict[key]), file=open("args.txt", "a"))
print("date: %s" % time.asctime(time.localtime(time.time())))
print("date: %s" % time.asctime(time.localtime(time.time())), file=(open("args.txt", "a")))

if args.att:
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
print(model)

optimizer = torch.optim.Adam(
    lr=args.lr,
    params=model.parameters(),
    betas=(0.9, 0.999),
    amsgrad=True
)

# X = torch.load("../../data/egg.pth")
X = torch.load("../../data/taskparam.pth")
query_dims = [0, 7, 8]
target_dims = [1, 2, 3, 4, 5, 6]

avg_loss = 0.0
for i in range(args.iter):
    optimizer.zero_grad()
    sample = X[np.random.randint(0, X.shape[0])].to(device)
    L = (sample[:, 0]-1).abs().argmin() + 1
    sample = sample[:L]

    R = torch.randperm(L)
    num_of_context = torch.randint(1, 8, (1,))
    num_of_target = torch.randint(2, 10, (1,))
    context_index = R[:num_of_context]
    # context_index = [0]
    # predict both context and target. they say it's better
    target_index = R[:(num_of_context+num_of_target)]

    out = model(context=sample[context_index, :], key=sample[context_index][:, query_dims], query=sample[target_index][:, query_dims])
    mu = out[:, :6]
    log_std = out[:, 6:]

    std = torch.nn.functional.softplus(log_std)
    dists = torch.distributions.Normal(mu, std)
    # calculate loss
    loss = -dists.log_prob(sample[target_index][:, target_dims]).mean()
    loss.backward()
    optimizer.step()
    
    avg_loss += loss.item()
    if (i+1) % 1000 == 0:
        print("iter: %d, loss: %.4f" % (i+1, avg_loss*1e-3))
        avg_loss = 0.0
        torch.save(model.eval().cpu().state_dict(), "model.ckpt")
        model.train().to(device)
