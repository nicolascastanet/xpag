import matplotlib.pyplot as plt
import torchvision
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
import seaborn



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
                
                
def train_vae_model(
            vae: nn.Module,
            optimizer: torch.optim.Optimizer,
            dataloader: torch.utils.data.DataLoader,
            nb_steps=100,
            print_loss=True
            ):
    # Set train mode for both the encoder and the decoder
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    vae.train()
    loss_list = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for epoch in range(nb_steps):
        train_loss = 0.0
        for x in dataloader:
            # Move tensor to the proper device
            x = x[0].to(device)
            x_hat = vae(x)
            
            #loss = F.mse_loss(x_hat, x, size_average=False) + 0.5*vae.encoder.kl
            # Evaluate loss
            loss = F.binary_cross_entropy(x_hat,x, size_average=False) + vae.encoder.kl
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            
        
        batch_loss = train_loss / len(dataloader.dataset)
        loss_list.append(batch_loss)
        
        if print_loss:
            print(f'epoch {epoch}:',batch_loss)
            
        if batch_loss < 20:
            return loss_list
            
    return loss_list
    #return train_loss / len(dataloader.dataset)
    
    
def plot_ae_outputs(epoch, encoder, decoder, path, dataset, t_idx, n=10):
    
    plt.figure(figsize=(16,4.5))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      #img = dataset[t_idx[i]][0].unsqueeze(0).to(device) # for torch input
      img = torch.from_numpy(dataset[t_idx[i]]).unsqueeze(0).to(device).type(torch.float) # for numpy input
      
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      
      plt.imshow(np.transpose(img.cpu().squeeze().numpy()), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      
      plt.imshow(np.transpose(rec_img.cpu().squeeze().numpy()), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
         
    plt.title(f'Epoch {epoch}', fontweight="bold", fontsize=20)
    plt.savefig(path + f'/epoch_{epoch}.png', format='png', dpi=400)
    
   
def show_image(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg))   
 
def ae_plot_gen(step, plot_step, vae, path, writer=None):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    vae.eval()
    d = vae.latent_dim

    with torch.no_grad():
        # sample latent vectors from the normal distribution
        latent = torch.randn(128, d, device=device)

        # reconstruct images from the latent vectors
        img_recon = vae.decoder(latent)
        img_recon = img_recon.cpu()

        fig, ax = plt.subplots(figsize=(20, 8.5))
        recons_img = torchvision.utils.make_grid(img_recon.data[:100],10,5)
        show_image(recons_img)
        plt.axis("off")
        plt.show()
        plt.savefig(path + f'/step_{step}.png', format='png', dpi=400)
        
        if writer is not None:
            
            #writer.add_image(f'images/recons/step_{step}', 
            #                 torchvision.transforms.functional.rotate(recons_img,180))
            writer.add_image(f'VAE/goal_generation', torch.transpose(recons_img,1,2), plot_step)
            


def compare_vae_obs(step,
                    plot_step,
                    real_obs, 
                    pixel_obs, 
                    encoder, 
                    decoder, 
                    path,
                    writer=None,
                    dist_threshold=0.1,
                    plot_images_similarity=False
                ):
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    encoder.eval()
    d = encoder.latent_dim

    latent_gen = torch.randn(200, d)#, device=device)
    
    real_obs_close = []
    pixel_obs_close = []
    idxs = []
    
    with torch.no_grad():
        # Latent code of real images
        latent_encode = encoder(pixel_obs)
        
        # Generated decoded images from latent space sampling 
        img_gen = decoder(latent_gen.to(device))
        img_gen = img_gen.cpu().numpy()
        
    for lg in latent_gen:
        # Compute L2 norm between latent codes
        diff = np.linalg.norm((latent_encode - lg).numpy(), axis=-1)
        argmin = np.argmin(diff)
        #if diff[argmin] > dist_threshold:
        #    continue
        #else:
        real_obs_close.append(real_obs[argmin])
        pixel_obs_close.append(pixel_obs[argmin].numpy())
        idxs.append(argmin)
        
    real_obs = np.array(real_obs_close)
    
    if plot_images_similarity:
        n = 10
        fig = plt.figure(figsize=(16,4.5))
        for i in range(n):
            ax = plt.subplot(2,n,i+1)
            real_img = pixel_obs_close[i]
            gen_img = img_gen[i]

            plt.imshow(np.transpose(gen_img), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('generated images')
            ax = plt.subplot(2, n, i + 1 + n)

            plt.imshow(np.transpose(real_img), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Corresponding real images')


        plt.title(f'Step {step}', fontweight="bold", fontsize=20)
        plt.savefig(path + f'/step_{step}.png', format='png', dpi=400)
        
        if writer is not None:
            writer.add_figure('VAE/real_goal_from_latent', fig, plot_step)
        
    return real_obs
    


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