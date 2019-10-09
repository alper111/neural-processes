import argparse
import pickle
import torch
import models
import utils
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="test a neural process.")
parser.add_argument("-hidden_dim", help="hidden dimension.", type=int, required=True)
parser.add_argument("-en_layer", help="number of encoder layers.", type=int, required=True)
parser.add_argument("-de_layer", help="number of decoder layers.", type=int, required=True)
parser.add_argument("-nhead", help="number of attention heads.", type=int)
parser.add_argument("-model", help="model path.", type=str, required=True)
parser.add_argument("-data", help="data path.", type=str, required=True)
parser.add_argument("-att", help="whether to use self-attention.", type=int, default=1)

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

data_dict = pickle.load(open(args.data, "rb"))
X = data_dict["data"]
query_dims = data_dict["query_dims"]
target_dims = data_dict["target_dims"]

if args.att == 1:
    model = models.ANPv2(
    in_dim=len(query_dims)+len(target_dims),
    hidden_dim=args.hidden_dim,
    query_dim=len(query_dims),
    out_dim=len(target_dims)*2,
    en_layer=args.en_layer,
    dec_layer=args.de_layer,
    nhead=args.nhead
    )
else:
    model = models.CNP(
        in_dim=len(query_dims)+len(target_dims),
        hidden_dim=args.hidden_dim,
        query_dim=len(query_dims),
        out_dim=len(target_dims)*2,
        en_layer=args.en_layer,
        dec_layer=args.de_layer
    )

model.to(device)
model.load_state_dict(torch.load(args.model))

with torch.no_grad():
    model.eval().cpu()
    for i in range(5):
        sample = X[np.random.randint(0, X.shape[0])]
        L = (sample[:, 0]-1).abs().argmin() + 1
        sample = sample[:L]
        R = torch.randperm(L)
        context_index = R[:3]
        out = model(context=sample[context_index], key=sample[context_index][:, query_dims], query=sample[:, query_dims])
        mean = out[:, :len(target_dims)]
        log_std = out[:, len(target_dims):]
        std = torch.nn.functional.softplus(log_std)
        
        if len(target_dims) == 1:
            plt.figure(figsize=(5, 4))
            for j in range(X.shape[0]):
                plt.plot(X[j, :, 0], X[j, :, target_dims], c="b")
            plt.plot(sample[:, 0], sample[:, target_dims], c="c")
            plt.plot(sample[:, 0], mean[:, s].numpy(), c='r')
            plt.fill_between(sample[:, 0], mean[:, 0]-std[:, 0], mean[:, 0]+std[:, 0], facecolor='r', alpha=0.1)
            plt.scatter(sample[context_index, 0], sample[context_index, target_dims], marker='x', c='g')
            plt.title("Training samples")
            plt.show()
        else:
            fig, ax = plt.subplots(1, len(target_dims), figsize=(len(target_dims)*5, 4))
            for s in range(len(target_dims)):
                for j in range(X.shape[0]):
                    ax[s].plot(X[j, :, 0], X[j, :, target_dims[s]], c="b")
                ax[s].plot(sample[:, 0], mean[:, s].numpy(), c="r")
                ax[s].fill_between(sample[:, 0], mean[:, s]-std[:, s], mean[:, s]+std[:, s], facecolor="r", alpha=0.1)
                ax[s].scatter(sample[context_index, 0], sample[context_index, target_dims[s]], marker="x", c="g")
                ax[s].set_title("Training samples")
            plt.show()
                    
