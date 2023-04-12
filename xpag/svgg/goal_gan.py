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
            dist_init = None,
            batch_size = 200, 
            history_length = 1000, 
            optimize_every=2000, 
            k_steps=100, 
            noise_dim=4,
            fetch_table=False,
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
    self.fetch_table=fetch_table
    self.p_valid = None


  def torch(self, x):
    return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
  
  def relabel_fetch_goals(self, goals):
    table_z_pos_fetch = torch.full((goals.shape[0],1),0.42469975 # Table height
                                    ).type(torch.FloatTensor
                                    ).to(self.device)
                                
    return torch.cat((goals, table_z_pos_fetch),dim=1)      
    
  def optimize(self, goal_success_pred, buffer, p_valid=None):

    self.opt_steps += 1
    i=0
    history = self.history_length
    while True:
        print("loop i =",i)
        ##############################################
        ### Over sampling of previous trajectories ###
        ##############################################
        
        buffers = buffer.pre_sample()
        rollout_batch_size = buffers["episode_length"].shape[0] # buffer current size
        #history_length = np.arange(max(rollout_batch_size - 100, 0), rollout_batch_size)
        
        episode_idxs = np.random.choice(
            rollout_batch_size,
            size=self.batch_size,
            replace=True,
            p=buffers["episode_length"][:, 0, 0]
            / buffers["episode_length"][:, 0, 0].sum(),
                    )


        t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
        t_samples = np.random.randint(t_max_episodes)

        transitions_dg = {
                key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
            }
        
        transitions_init = {
                key: buffers[key][episode_idxs, 0] for key in buffers.keys()
            }
        
        #if self.opt_steps < 5:
        #  desired_g = transitions_dg["next_observation.achieved_goal"]
        #else:
        #  desired_g = transitions_dg["next_observation.desired_goal"]
        desired_g = transitions_dg["next_observation.achieved_goal"]
        init_ag = transitions_init["observation.achieved_goal"]
      
        #buffers = copy.deepcopy(buffer.sample_recent(self.batch_size, history)) # Train on recent data
        #desired_g = buffers["observation.desired_goal"][:,0,:] # If same goal at every step of an episode
        
        #desired_g = buffers["observation.achieved_goal"][:,-1,:]
        #init_ag = buffers["observation.achieved_goal"][:,0,:] # Init achieved goal
        
        #import ipdb;ipdb.set_trace()
      

        states = np.concatenate((init_ag, desired_g),axis=1)
        #y = buffers["is_success"].max(axis = 1)

        ################################
        ### Label GOIDs in real data ###
        ################################

        probas = goal_success_pred(self.torch(states)) # are passed through sigmoid
        y_g = (probas > self.p_min) & (probas < self.p_max) # goid or not
        
        if p_valid is not None:
          validity_proba = p_valid.log_prob(self.torch(desired_g), log=False)
          y_g = y_g & (validity_proba > 0.5)
          
            
        #import ipdb;ipdb.set_trace()

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
      
    if len(y.shape) == 1:
      y = y.reshape(-1,1)
      
    if self.fetch_table:
      X = np.delete(X, (2,5), 1)
    #print("Begin GAN training")
    inputs = self.torch(X)
    y_g = self.torch(y)
    #start = time.time()

    for _ in range(self.k_steps):
        #######################
        ### train  GAN disc ###
        #######################

        L_real_goids = (y_g * (self.gan_discriminator(inputs, sig=False) - 1)**2).mean()
        L_real_not_goids = (torch.logical_not(y_g) * (self.gan_discriminator(inputs, sig=False) + 1)**2).mean()

        noise = torch.randn(inputs.shape[0], self.noise_dim).to(self.device)
        
        # Sample init state to concat with gen goals
        if self.dist_init is not None:
          obs_init = self.dist_init.sample((inputs.shape[0],)).to(self.device)
        else:
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