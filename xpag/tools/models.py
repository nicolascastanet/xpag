import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import os
import math
from typing import Callable
from abc import ABC, abstractmethod
from sklearn.linear_model import SGDClassifier



def layer_init(layer, w_scale=1.0):
  if hasattr(layer, 'weight') and len(layer.weight.shape) > 1:
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
  return layer

class GELU(nn.Module):
  def forward(self, input):
    return F.gelu(input)

class DROPOUT(nn.Module):
  """
  Use in MCdropout for dropout in train and test time
  """
  def __init__(self,proba):
    super().__init__()
    self.p = proba
  def forward(self, input):
    return F.dropout(input, p=self.p)



def train_torch_model(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        criterion,
        nb_steps=100
        ):

        for _ in range(nb_steps):
            for x, y in dataloader:
                
                output = model(x)
                loss = criterion(output, y.reshape(-1,1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class MLP(nn.Module):
    """Standard feedforward network.
    Args:
      input_size (int): number of input features
      output_size (int): number of output features
      hidden_sizes (int tuple): sizes of hidden layers

      norm: pre-activation module (e.g., nn.LayerNorm)
      activ: activation module (e.g., GELU, nn.ReLU)
      drop_prob: dropout probability to apply between layers (not applied to input)
    """
    def __init__(
        self, 
        input_size, 
        output_size=1, 
        layers=(256, 256), 
        norm=nn.Identity, 
        activ=nn.ReLU, 
        drop_prob=0.
    ):
        super().__init__()
        self.output_size = output_size

        layer_sizes = (input_size, ) + tuple(layers) + (output_size, )
        if len(layer_sizes) == 2:
            layers = [nn.Linear(layer_sizes[0], layer_sizes[1], bias=False)]
        else:
            layers = []
            for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:]):
                layers.append(nn.Linear(dim_in, dim_out))
                if norm not in [None, nn.Identity]:
                    layers.append(norm(dim_out))
                layers.append(activ())
                if drop_prob > 0.:
                    layers.append(nn.Dropout(p=drop_prob))
            layers = layers[:-(1 + (norm not in [None, nn.Identity]) + (drop_prob > 0))]
            layers = list(map(layer_init, layers))
        self.f = nn.Sequential(*layers)

    def forward(self, x, sig=True):
        if sig:
            return torch.sigmoid(self.f(x))
        else:
            return self.f(x)





class OCSVM(nn.Module):
    def __init__(self, kernel,sk_model):
        super(OCSVM, self).__init__()
        self.kernel = kernel
        self.sk_model = sk_model
        self.rand_nb_scale = 1000
        self.penalty = 0.0001
        self.device = torch.device("cuda")

    def fit(self,X:np.array):
        """input : np.array"""
        
        self.sk_model.fit(X)
        self.X_supp = torch.from_numpy(self.sk_model.support_vectors_).type(torch.float).to(self.device) # support vectors
        self.A = torch.from_numpy(self.sk_model.dual_coef_).type(torch.float).to(self.device) # alpha dual coef
        self.B = torch.from_numpy(self.sk_model.intercept_).type(torch.float).to(self.device) # decision biais
        self.calibration(X)


    def calibration(self, x):
        x_min, x_max = np.min(x,axis=0), np.max(x,axis=0)
        x_range = x_max - x_min
        while True:
            rd_X = np.random.uniform(low=x_min - x_range,
                                     high=x_min + x_range, 
                                     size=(self.rand_nb_scale,x.shape[1])
                                    )
            y = self.sk_model.predict(rd_X).reshape(-1,1)
            if y.sum() > -len(y) and y.sum() < len(y) : # check if there is 2 classes
                break
        X = self.sk_model.decision_function(rd_X).reshape(-1,1)
        
        clf = SGDClassifier(loss='modified_huber',alpha=self.penalty)
        clf.fit(X,y) 
        self.w = torch.from_numpy(clf.coef_).type(torch.float).to(self.device)
        self.b = torch.from_numpy(clf.intercept_).type(torch.float).to(self.device)

    def logistic(self, x):
        return torch.sigmoid(torch.mm(x, self.w)+ self.b)
        

    def forward(self, x):
        """input : torch.tensor"""

        if self.sk_model.fit_status_:
            raise NameError('Sklearn model not fitted !')
        
        K = self.kernel(x,self.X_supp) # kernel matrix
        scores = torch.mm(K,self.A.reshape(self.X_supp.shape[0],-1)) + self.B # decision function
        return scores


    def log_prob(self,x,log=True):
        if log:
            return torch.log(self.logistic(self.forward(x)))
        return self.logistic(self.forward(x))




class LogisticRegression(torch.nn.Module):
    """ A Logistic Regression Model with sigmoid output in Pytorch"""
    def __init__(self, input_size):
        super().__init__()
        self.in_size = input_size
        self.w = torch.nn.Parameter(torch.randn((1, input_size)))
        self.b = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        if self.in_size > 1:
            s = torch.mm(self.w, x.T) + self.b 
        else:
            s = self.w*x + self.b
        return torch.sigmoid(s)

    def reset_weights(self):
        self.w = torch.nn.Parameter(torch.randn((1, self.in_size)))
        self.b = torch.nn.Parameter(torch.randn(1))


class OCSVM_0(torch.nn.Module):
    """
    PyTorch implementation on One class SVM with probability calibration
    """
    def __init__(self, kernel,sk_model):
        super(OCSVM_0, self).__init__()
        self.kernel = kernel
        self.sk_model = sk_model
        assert  torch.cuda.is_available()
        self.device = torch.device("cuda")
        self.logistic = LogisticRegression(input_size=1)
        self.logistic.to(self.device)
        self.fit_status = False

    def fit(self,X):
        """input : np.array"""

        # Fit One class SVM
        
        self.sk_model.fit(X)
        self.X_supp = torch.from_numpy(self.sk_model.support_vectors_).type(torch.float).to(self.device) # support vectors
        self.A = torch.from_numpy(self.sk_model.dual_coef_).type(torch.float).to(self.device) # alpha dual coef
        self.B = torch.from_numpy(self.sk_model._intercept_).type(torch.float).to(self.device) # decision biais
        self.n = int(self.sk_model.n_support_)

        # Fit logistic regression for calibration

        # prepare data with random oversample
        oversample = RandomOverSampler()
        X_train, y_train = self.sk_model.decision_function(X), self.sk_model.predict(X)
        X_train, y_train = oversample.fit_resample(X_train.reshape(-1,1), y_train.reshape(-1,1))
        y_train = (y_train+1)/2
        X_train, y_train = torch.from_numpy(X_train).to(self.device), torch.from_numpy(y_train).to(self.device)
        X_train, y_train = X_train.type(torch.float), y_train.type(torch.float)
        torch_train_dataset = data.TensorDataset(X_train,y_train)
        train_dataloader = data.DataLoader(torch_train_dataset, batch_size=len(torch_train_dataset))

        # Fit logistic regression model
        self.logistic.reset_weights()
        self.logistic.to(self.device)
        self.logistic.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.logistic.parameters(), lr=0.1)

        for _ in range(200):
            for x, y in train_dataloader:

                output = self.logistic(x)
                loss = criterion(output, y.reshape(-1,1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.fit_status = True

    def forward(self, x):
        """input : torch.tensor"""

        #import ipdb;ipdb.set_trace()

        #if self.sk_model.fit_status_:
        #    raise NameError('Sklean model not fitted !')
        
        K = self.kernel(x,self.X_supp) # kernel matrix
        scores = torch.mm(K,self.A.reshape(self.n,-1)) + self.B # decision function
        return torch.sign(scores), scores

    def log_prob(self,x,log=True):
        
        if log:
            prob = torch.log(self.logistic(self.forward(x)[1]))
        else:
            prob = self.logistic(self.forward(x)[1])
        return prob



class MultivariateGeneralizedGaussian():

    def __init__(self,mean=0,input_shape=1,alpha=1,beta=2):
        self.input_shape = input_shape
        self.mean = mean
        self.alpha = alpha
        self.beta = beta
    
    def log_prob(self, x, beta=None, alpha=None, log=True):

        if beta is None:
            beta = self.beta
        if alpha is None:
            alpha = self.alpha

        if self.input_shape == 1:
            g = torch.lgamma(torch.tensor(1/beta))
            norm = beta/(2*alpha*g)
            
            return norm*torch.exp(-(torch.abs(x-self.mean)/alpha)**beta)

        else:
            
            in_s = self.input_shape

            # Norm calcul
            cov = alpha*torch.eye(in_s)
            g_1 = torch.exp((torch.lgamma(torch.tensor(in_s/2))))
            g_2 = torch.exp((torch.lgamma(torch.tensor(in_s/(2*beta)))))
            det = torch.det(cov)**(1/2)

            n_1 = g_1/((math.pi**(in_s/2))*g_2*2**(in_s/(2*beta)))
            n_2 = beta/det

            norm = torch.log(n_1*n_2+1e-7)

            # Batch Kernel distance
            bs = x.shape[0]
            x = x.unsqueeze(1)
            mean = self.mean.unsqueeze(0)
            cov = cov.repeat(bs,1,1)
            res_1 = torch.bmm((x-mean),torch.inverse(cov))
            res_2 = torch.bmm(res_1,torch.permute(x-mean, (0, 2, 1)))

            #prob = torch.exp(-1/2*res_2**beta)
            prob = -1/2*res_2**beta
            
            if log == False:
                return torch.exp((norm + prob).squeeze(1))
            else:
                return (norm + prob).squeeze(1)