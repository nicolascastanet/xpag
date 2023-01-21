import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from pstats import SortKey
import time
import copy




class GoalGanTrainer():
  """Goal-GAN training module"""

  def __init__(
            self,
            gan_discriminator,
            gan_generator,
            init_state,
            dist_init,
            batch_size = 200, 
            history_length = 1000, 
            optimize_every=2000, 
            k_steps=100, 
            noise_dim=4,
            log_every=5000 
        ):
    assert  torch.cuda.is_available()
    self.device = torch.device("cuda")


    self.gan_discriminator = gan_discriminator
    self.gan_generator = gan_generator
    self.init_state = init_state
    self.dist_init = dist_init

    # Setup GANs optimizers
    self.optimizerD = torch.optim.Adam(self.gan_discriminator.parameters())
    self.optimizerG = torch.optim.Adam(self.gan_generator.parameters())
    
    # Misc. variables
    self.log_every = log_every
    self.batch_size = batch_size
    self.history_length = history_length
    self.optimize_every = optimize_every
    self.opt_steps = 0
    self.is_opt = 0
    self.k_steps = k_steps
    self.k_batch = True
    self.sampling_mode = 'over' # in {'over', 'random', 'balanced', 'smote', 'under_smote', 'MEP'}
    self.n_batch = int(history_length / batch_size)

    self.p_min = 0.2
    self.p_max = 0.8
    self.noise_dim = noise_dim
    self.ready = False
    self.log = True


  def torch(self, x):
    return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
    
  def optimize(self, goal_success_pred, buffer):

    self.opt_steps += 1
    i=0
    history = self.history_length
    while True:
        print("loop i =",i)
        ##############################################
        ### Over sampling of previous trajectories ###
        ##############################################

        buffers = copy.deepcopy(buffer.sample_recent(self.batch_size, history)) # Train on recent data
        desired_g = buffers["observation.desired_goal"][:,0,:] # If same goal at every step of an episode
        init_ag = buffers["observation.achieved_goal"][:,0,:] # Init achieved goal

        states = np.concatenate((init_ag, desired_g),axis=1)
        y = buffers["is_success"].max(axis = 1)

        ################################
        ### Label GOIDs in real data ###
        ################################

        probas = goal_success_pred(self.torch(states)) # are passed through sigmoid
        y_g = (probas > self.p_min) & (probas < self.p_max) # goid or not

        i+=1

        # Check if there is more than 1 class
        if y_g.sum() > 0 and len(y_g) > y_g.sum():
            break
        if i > 5:
            history*=2
        if i > 10:
          break


    if y_g.sum() > 0 and len(y_g) > y_g.sum():
      # Naive random over sampling
      oversample = RandomOverSampler()
      X, y = oversample.fit_resample(states, y_g.cpu())
    else:
      X, y = np.copy(states), np.copy(y_g.cpu())

    inputs = self.torch(X)
    behav_goals = self.torch(X[:,2:])
    y_g = self.torch(np.expand_dims(y,1))

    #print("Begin GAN training")
    #start = time.time()

    for _ in range(self.k_steps):
        #######################
        ### train  GAN disc ###
        #######################

        L_real_goids = (y_g * (self.gan_discriminator(inputs, sig=False) - 1)**2).mean()
        L_real_not_goids = (torch.logical_not(y_g) * (self.gan_discriminator(inputs, sig=False) + 1)**2).mean()

        noise = torch.randn(inputs.shape[0], self.noise_dim).to(self.device)
        
        # Sample init state to concat with gen goals
        obs_init = self.init_state.repeat(inputs.shape[0],1).to(self.device)
        gen_goals = torch.cat((obs_init, self.gan_generator(noise, sig=False)), 1)
        L_fake = ((self.gan_discriminator(gen_goals, sig=False) + 1)**2).mean()

        Loss_D = L_real_goids + L_real_not_goids + L_fake

        self.optimizerD.zero_grad()
        Loss_D.backward()
        self.optimizerD.step()

        ############################
        ### train  GAN generator ###
        ############################

        for i in range(2):
            noise = torch.randn(inputs.shape[0], self.noise_dim).to(self.device)
            gen_goals = torch.cat((obs_init, self.gan_generator(noise, sig=False)), 1)
            Loss_G = (self.gan_discriminator(gen_goals, sig=False)**2).mean()

            self.optimizerG.zero_grad()
            Loss_G.backward()
            self.optimizerG.step()
        
            self.ready = True
        

    #stop = time.time()
    #print("End GAN training, time (s) : ",round(stop-start,1))