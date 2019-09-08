import torch
import numpy as np


# source: https://github.com/krasserm/bayesian-machine-learning/blob/master/gaussian_processes.ipynb
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    ''' Isotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2. Args: X1: Array of m points (m x d). X2: Array of n points (n x d). Returns: Covariance matrix (m x n). '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
    
# source: # https://github.com/krasserm/bayesian-machine-learning/blob/master/gaussian_processes.ipynb
def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    ''' Computes the sufficient statistics of the GP posterior predictive distribution from m training data X_train and Y_train and n new inputs X_s. Args: X_s: New input locations (n x d). X_train: Training locations (m x d). Y_train: Training targets (m x 1). l: Kernel length parameter. sigma_f: Kernel vertical variation parameter. sigma_y: Noise parameter. Returns: Posterior mean vector (n x d) and covariance matrix (n x n). '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_s, X_train, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + sigma_y**2 * np.eye(len(X_s))
    
    mu_s = np.matmul(K_s, np.linalg.solve(K, Y_train))
    cov_s = K_ss - np.matmul(K_s, np.linalg.solve(K, K_s.T))
    
    return mu_s, cov_s

class CNP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, query_dim, out_dim, en_layer, dec_layer):
        super(CNP, self).__init__()
        if en_layer == 1:
            self.encoder = torch.nn.Linear(in_dim, hidden_dim)
        else:
            self.encoder = [
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU()
            ]
            for i in range(en_layer-2):
                self.encoder.append(torch.nn.Linear(hidden_dim, hidden_dim))
                self.encoder.append(torch.nn.ReLU())
            self.encoder.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.encoder = torch.nn.Sequential(*self.encoder)
        
        if dec_layer == 1:
            self.decoder = torch.nn.Linear(hidden_dim+query_dim, out_dim)
        else:
            self.decoder = [
                torch.nn.Linear(hidden_dim+query_dim, hidden_dim),
                torch.nn.ReLU()
            ]
            for i in range(dec_layer-2):
                self.decoder.append(torch.nn.Linear(hidden_dim, hidden_dim))
                self.decoder.append(torch.nn.ReLU())
            self.decoder.append(torch.nn.Linear(hidden_dim, out_dim))
            self.decoder = torch.nn.Sequential(*self.decoder)
        
    def forward(self, context, query):
        query = query.view(query.shape[0], -1)
        # encode
        h = self.encoder(context)
        # aggregate
        h = h.mean(dim=0)
        h = torch.stack([h]*(query.shape[0]), dim=0)
        r = torch.cat([h, query], dim=1)
        # predict
        out = self.decoder(r)
        return out


class ANP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, query_dim, out_dim, en_layer, dec_layer, nhead):
        super(ANP, self).__init__()
        if en_layer == 1:
            self.encoder = torch.nn.Linear(in_dim, hidden_dim)
        else:
            self.encoder = [
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU()
            ]
            for i in range(en_layer-2):
                self.encoder.append(torch.nn.Linear(hidden_dim, hidden_dim))
                self.encoder.append(torch.nn.ReLU())
            self.encoder.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.encoder = torch.nn.Sequential(*self.encoder)
        
        if dec_layer == 1:
            self.decoder = torch.nn.Linear(hidden_dim, out_dim)
        else:
            self.decoder = [
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            ]
            for i in range(dec_layer-2):
                self.decoder.append(torch.nn.Linear(hidden_dim, hidden_dim))
                self.decoder.append(torch.nn.ReLU())
            self.decoder.append(torch.nn.Linear(hidden_dim, out_dim))
            self.decoder = torch.nn.Sequential(*self.decoder)
        self.projector = torch.nn.Linear(query_dim, hidden_dim)
        self.attention = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead)


    def forward(self, context, key, query):
        query = query.view(query.shape[0], -1)
        key = key.view(key.shape[0], -1)
        # encode
        h = self.encoder(context)
        h.unsqueeze_(1)
        # aggregate
        q_t = self.projector(query)
        k_t = self.projector(key)
        q_t.unsqueeze_(1)
        k_t.unsqueeze_(1)
        h, _ = self.attention(query=q_t, key=k_t, value=h)
        h.squeeze_(1)
        # predict
        pred = self.decoder(h)
        return pred
