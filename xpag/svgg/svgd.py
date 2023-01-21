import math
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.distributions as dist
import traceback



# RBF Kernel
class RBF(torch.nn.Module):
    def __init__(
        self, 
        sigma=None,
        gamma=None,
        sig_mult=1
        ):
        super(RBF, self).__init__()

        self.sigma = sigma
        self.gamma = gamma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
            self.sig_median=sigma
        else:
            sigma = self.sigma
            
        if self.gamma is None:
            gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        else:
            gamma = self.gamma
        K_XY = (-gamma * dnorm2).exp()
  
        return K_XY


# Stein Variational Gradient Descent
class SVGD:
    def __init__(
        self, 
        P, 
        K, 
        optimizer, 
        epoch,
        ):

        self.P = P
        self.K = K
        self.optim = optimizer
        self.T = epoch
        
        assert  torch.cuda.is_available()
        self.device = torch.device("cuda")

    def phi(self, X,ann=1):
        X = X.detach().requires_grad_(True).to(self.device)
        
        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob.sum(), X, retain_graph=True)[0]

        if score_func.isnan().any():
            score_func = torch.nan_to_num(score_func)
            
        K_XX = self.K(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]
        #if grad_K.isnan().any():
        #    import ipdb;ipdb.set_trace()

        phi = (K_XX.detach().matmul(score_func)*ann + grad_K) / X.size(0)
        if phi.isnan().any():
            phi = torch.nan_to_num(phi)            
        return phi

    def step(self, X,ann=1):
        self.optim.zero_grad()
        X.grad = -self.phi(X,ann)
        if X.grad.isnan().any():
            import ipdb;ipdb.set_trace()
        self.optim.step()


    def annealed(self,t,period,T=1e6,p=5, mode=2, C=4):
        if mode == 1:
            return np.tanh((self.slope*t/T)**p)
        elif mode == 2:
            t = t % period
            return np.tanh((self.slope*t/period)**p)
        elif mode==3:
            return int(t > (T//2))