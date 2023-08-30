# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import random
import cProfile
import re
import pstats, io
from pstats import SortKey
from abc import ABC, abstractmethod
from typing import Tuple, Any, Iterable
from types import MethodType
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils import data
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler

# SVGG import
from xpag.svgg.svgd import RBF, SVGD
from xpag.svgg.misc import softmax
from xpag.tools.models import train_torch_model
from xpag.plotting.plotting import update_csv, plot_decision_boundary, plot_particles, plot_prior



class Setter(ABC):
    def __init__(self, name: str):
        self.name = name
        self.sparse_reward = True

    @abstractmethod
    def reset(self, env, observation, info, eval_mode=False) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def reset_done(
        self, env, observation, info, done, eval_mode=False
    ) -> Tuple[Any, Any, Any]:
        pass

    @abstractmethod
    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode: bool = False,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        pass

    @abstractmethod
    def write_config(self, output_file: str):
        pass

    @abstractmethod
    def save(self, directory: str):
        pass

    @abstractmethod
    def load(self, directory: str):
        pass

    def process_transition(self, goals, obs, info, terminated):
        # Relabel transition w.r.t. the new desired goals

        obs["desired_goal"] = np.copy(goals)
        
        info["is_success"] = self.eval_env.is_success(obs["achieved_goal"], goals
                ).reshape((self.num_envs, 1))

        if self.terminated:
            terminated = np.copy(info["is_success"]).reshape((self.num_envs, 1))

        if self.sparse_reward:
            reward = self.eval_env.compute_reward(obs["achieved_goal"], goals, info
            ).reshape((self.num_envs, 1))
        
        else:
            reward = -np.linalg.norm(obs['achieved_goal'] - goals, axis=-1
                                    ).astype(np.float32
                                    ).reshape((self.num_envs, 1))
                                    
        return obs, reward, terminated, info



class RigSetter(Setter, ABC):
    def __init__(self, num_envs, eval_env, input_dim):
        super().__init__("RigSetter")
        self.num_envs = num_envs
        self.eval_env = eval_env
        self.current_goals = None
        self.terminated = False
        self.d = input_dim
        self.current_goals = None

    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):

        if not eval_mode:
            #print("first reset done !")
            self.first_reset_done = True
            # Replace Goals when episode is done
            new_goals = np.random.randn(self.num_envs, self.d)
            
            # Update current goals
            if self.current_goals is None:
                self.current_goals = np.copy(new_goals)
            else:
                self.current_goals = np.where(done == 1, new_goals, self.current_goals)
                
            observation["desired_goal"] = np.copy(self.current_goals)

        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        if not eval_mode and self.current_goals is not None:
            new_observation, reward, terminated, info = self.process_transition(    
                                                                            self.current_goals,
                                                                            new_observation, 
                                                                            info, 
                                                                            terminated
                                                                            )

        return new_observation, reward, terminated, truncated, info

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass


class DefaultSetter(Setter, ABC):
    def __init__(self):
        super().__init__("DefaultSetter")

    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        return new_observation, reward, terminated, truncated, info

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass



class RandomSetter(Setter, ABC):
    def __init__(self, buffer, num_envs, eval_env):
        super().__init__("RandomSetter")
        self.buffer = buffer
        self.num_envs = num_envs
        self.current_goals = None
        self.eval_env = eval_env
        self.terminated = False

    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):

        if not eval_mode:
            # sample random achieved goals
            buffers = self.buffer.pre_sample()
            current_size = buffers["episode_length"].shape[0]
            
            episode_idxs = np.random.choice(
                current_size,
                size=self.num_envs,
                replace=True,
                p=buffers["episode_length"][:, 0, 0]
                / buffers["episode_length"][:, 0, 0].sum(),
                    )

            t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
            t_samples = np.random.randint(t_max_episodes)

            transitions = {
                    key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
                }

            ag_candidates = transitions["next_observation.achieved_goal"]
            self.current_goals = ag_candidates


        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        if not eval_mode and self.current_goals is not None:
            new_observation, reward, terminated, info = self.process_transition(    
                                                                            self.current_goals,
                                                                            new_observation, 
                                                                            info, 
                                                                            terminated
                                                                            )

        return new_observation, reward, terminated, truncated, info

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass





class UniformSetter(Setter, ABC):
    def __init__(self, eval_env, num_envs):
        super().__init__("UniformSetter")
        self.current_goals = None
        self.eval_env = eval_env
        self.num_envs = num_envs
        self.terminated = False
        self.maze_size = int(eval_env.size_max[0])+1

    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        
        if not eval_mode:
            
            new_goals = (np.random.rand(len(done), 2) * self.maze_size) - np.array([0.5,0.5]) # TODO: Remove hard cpoded shape 
            # Update current goals
            if self.current_goals is None:
                self.current_goals = new_goals
            else:
                self.current_goals = np.where(done == 1, new_goals, self.current_goals)
            observation["desired_goal"] = np.copy(self.current_goals)

        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):
        if not eval_mode:
            if self.current_goals is not None:
                new_observation, reward, terminated, info = self.process_transition(   
                                                                            self.current_goals,
                                                                            new_observation, 
                                                                            info, 
                                                                            terminated
                                                                            )

        return new_observation, reward, terminated, truncated, info

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass
    
    
    
class SvggBufferSetter(Setter, ABC):
    def __init__(
        self, 
        num_envs, 
        eval_env,
        buffer, 
        criterion, 
        model,
        model_optimizer,
        plot,
        save_dir=None
    ):
        super().__init__("SvggBufferSetter")
        self.eval_env = eval_env
        self.num_envs = num_envs
        self.buffer = buffer
        self.criterion = criterion
        self.model = model
        self.model_optimizer = model_optimizer
        self.save_dir = save_dir

        self.steps = 1
        assert  torch.cuda.is_available()
        self.device = torch.device("cuda")

        self.current_goals = None
        self.num_ag_candidate = 100
        
        # Model param
        self.model_oe = 100 // num_envs
        self.model_bs = 100
        self.model_hl = 300
        self.model_k_steps = 100
        self.model_opt_steps = 0
        self.model_ready = False

        # Misc.
        self.model_plot_freq = 1
        self.plot = plot
        self.first_reset_done = False
        self.terminated = True
        self.table_fetch = False


    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        
        if not eval_mode and self.model_ready:

            # sample random achieved goals
            buffers = self.buffer.pre_sample()
            current_size = buffers["episode_length"].shape[0]
            
            episode_idxs = np.random.choice(
                current_size,
                size=self.num_ag_candidate * self.num_envs,
                replace=True,
                p=buffers["episode_length"][:, 0, 0]
                / buffers["episode_length"][:, 0, 0].sum(),
                    )

            t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
            t_samples = np.random.randint(t_max_episodes)

            transitions = {
                    key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
                }

            ag_candidates = transitions["next_observation.achieved_goal"]
            
            scores = self.criterion.log_prob(torch.from_numpy(ag_candidates)
                                             .type(torch.float)
                                             .to(self.device)
                                            )
            
            probas = F.softmax(scores, dim=0)
            distrib = Categorical(probas.squeeze())
            chosen_idx = distrib.sample((self.num_envs,)).cpu().numpy()
            #import ipdb;ipdb.set_trace()

            #goals = ag_candidates[chosen_idx]
            #scores = scores.reshape((self.num_ag_candidate,-1))
            #chosen_idx = torch.argmax(scores, axis=0).cpu().numpy()
            new_goals = ag_candidates[chosen_idx]
            
            
            if self.current_goals is None:
                self.current_goals = np.copy(new_goals)
            else:
                self.current_goals = np.where(done == 1, np.copy(new_goals), self.current_goals)      

        return observation, info, done


    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        # Count agent steps and optimize the different modules
        self.steps += (1-eval_mode)
        
        if not eval_mode:
            # We have to change the desired goal at every step because the env.step
            # set the goal himself with the env._sample_goal function
            # Also we have to recompute the reward / success of the trajectory

            # Relabel if there is some current goals
            if self.current_goals is not None:
                new_observation, reward, terminated, info = self.process_transition(self.current_goals,
                                                                                    new_observation, 
                                                                                    info, 
                                                                                    terminated
                                                                                    )

            # Optimize model
            if not self.steps % max(self.model_oe, 1):

                buffers = copy.deepcopy(self.buffer.sample_recent(self.model_bs, self.model_hl)) # Train on recent data
                desired_g = buffers["observation.desired_goal"][:,0,:] # If same goal at every step of an episode
                init_ag = buffers["observation.achieved_goal"][:,0,:] # Init achieved goal

                X = np.concatenate((init_ag, desired_g),axis=1)
                y = buffers["is_success"].max(axis = 1)
                states = np.copy(X); successes = np.copy(y)

                # Check if there is more than 1 class
                if y.sum() > 0 and len(y) > y.sum():
                    oversample = RandomOverSampler()
                    X, y = oversample.fit_resample(X, y)

                X, y = torch.from_numpy(X), torch.from_numpy(y)
                X, y = X.type(torch.float).to(self.device), y.type(torch.float).to(self.device)
                torch_train_dataset = data.TensorDataset(X,y)
                train_dataloader = data.DataLoader(torch_train_dataset, 
                                                    batch_size=len(torch_train_dataset))

                train_torch_model(
                                self.model, 
                                self.model_optimizer, 
                                train_dataloader, 
                                nn.BCELoss(), 
                                self.model_k_steps
                            )

                self.model_opt_steps += 1
                #if self.plot:
                #self.plot_model(X, y, env)
                self.model_ready = True
                with torch.no_grad():
                    states = torch.from_numpy(states).type(torch.FloatTensor).to(self.device)
                    successes = torch.from_numpy(successes).type(torch.FloatTensor).to(self.device)
                    outputs = self.model(states)
                    acc = ((outputs > 0.5).float() == successes.reshape(-1,1)).float().mean()

                update_csv("nn_acc", float(acc), self.steps * self.num_envs, self.save_dir)


        return new_observation, reward, terminated, truncated, info



    def plot_prior(self,env, achieved_g):
        if not self.prior_opt_steps % max(self.prior_plot_freq, 1):
            plot_prior(
                    achieved_g, 
                    env,
                    self.steps*self.num_envs,
                    self.save_dir,
                    self.prior
                )


    def plot_model(self, X, y, env):
        if not self.model_opt_steps % max(self.model_plot_freq, 1):
            plot_decision_boundary(
                                self.model, 
                                X, 
                                y,
                                env,
                                self.steps * self.num_envs,
                                self.save_dir
                            )
    
    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass



class SvggSetter(Setter, ABC):
    def __init__(
        self, 
        num_envs, 
        eval_env,
        buffer, 
        particles,
        svgd,
        criterion, 
        model,
        model_optimizer,
        prior,
        plot,
        part_to_goals=lambda x: x.cpu().numpy(),
        save_dir=None
    ):
        super().__init__("SvggSetter")
        self.eval_env = eval_env
        self.num_envs = num_envs
        self.buffer = buffer
        self.particles = particles
        self.svgd = svgd
        self.criterion = criterion
        self.model = model
        self.model_optimizer = model_optimizer
        self.prior = prior
        self.save_dir = save_dir

        self.steps = 1
        assert  torch.cuda.is_available()
        self.device = torch.device("cuda")

        # Particles param
        self.particles_oe = 50 // num_envs
        self.particles_opt_steps = 0
        self.particles_period = 5000 // num_envs
        self.n = self.particles.shape[0]

        # Init current goals
        self.part_to_goals = part_to_goals
        init_goal_idxs = np.random.choice(self.n,self.num_envs)
        self.current_goals = self.part_to_goals(self.particles)[init_goal_idxs]
        
        # Prior param
        self.prior_oe = 1000 // num_envs
        self.prior_batch_size = 10_000
        self.prior_opt_steps = 0
        self.prior_ready = False

        # Model param
        self.model_oe = 100 // num_envs
        self.model_bs = 100
        self.model_hl = 300
        self.model_k_steps = 100
        self.model_opt_steps = 0
        self.model_ready = False

        # Misc.
        self.particles_plot_freq = 500
        self.model_plot_freq = 1
        self.prior_plot_freq = 1
        self.plot = plot
        self.annealed_freq = 5
        self.first_reset_done = False
        self.terminated = True
        self.table_fetch = False
        
        plot_particles(
                    self.particles, 
                    self.criterion, 
                    self.steps*self.num_envs,
                    self.save_dir,
                )
        


    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        #if not eval_mode:
        #    # Replace Goals when episode is done
        #    goal_idxs = np.random.choice(self.n,self.num_envs)
        #    goal_particles = self.particles.cpu()[goal_idxs]
    #
        #    goal = np.where(done == 1, goal_particles, env.goal)
        #    env.set_goal(np.copy(goal)) # This is wrong, don't do it in the env, but in the setter !
        #    observation["desired_goal"] = np.copy(goal)

        if not eval_mode:
            #print("first reset done !")
            self.first_reset_done = True
            # Replace Goals when episode is done
            new_goal_idxs = np.random.choice(self.n,self.num_envs)
            goals = self.part_to_goals(self.particles)
            new_goals = goals[new_goal_idxs]
            # Update current goals
            self.current_goals = np.where(done == 1, new_goals, self.current_goals)
            observation["desired_goal"] = np.copy(self.current_goals)

        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        # Count agent steps and optimize the different modules
        self.steps += (1-eval_mode)
        
        if not eval_mode:
            # We have to change the desired goal at every step because the env.step
            # set the goal himself with the env._sample_goal function
            # Also we have to recompute the reward / success of the trajectory

            #import ipdb;ipdb.set_trace()
            new_observation, reward, terminated, info = self.process_transition(    self.current_goals,
                                                                                    new_observation, 
                                                                                    info, 
                                                                                    terminated
                                                                                    )

            
            # Optimize particles
            if not self.steps % max(self.particles_oe, 1) and self.model_ready and self.prior_ready:    
                
                if (self.steps % self.particles_period) < self.particles_period / self.annealed_freq:
                    annealed = 0
                else:
                    annealed = 1

                #for _ in range(20):
                self.svgd.step(self.particles, annealed)
                self.particles_opt_steps += 1
                if self.plot:
                    self.plot_particles(env)
        
            # Optimize model
            if not self.steps % max(self.model_oe, 1):

                buffers = copy.deepcopy(self.buffer.sample_recent(self.model_bs, self.model_hl)) # Train on recent data
                desired_g = buffers["observation.desired_goal"][:,0,:] # If same goal at every step of an episode
                init_ag = buffers["observation.achieved_goal"][:,0,:] # Init achieved goal

                X = np.concatenate((init_ag, desired_g),axis=1)
                y = buffers["is_success"].max(axis = 1)
                states = np.copy(X); successes = np.copy(y)

                # Check if there is more than 1 class
                if y.sum() > 0 and len(y) > y.sum():
                    oversample = RandomOverSampler()
                    X, y = oversample.fit_resample(X, y)

                X, y = torch.from_numpy(X), torch.from_numpy(y)
                X, y = X.type(torch.float).to(self.device), y.type(torch.float).to(self.device)
                torch_train_dataset = data.TensorDataset(X,y)
                train_dataloader = data.DataLoader(torch_train_dataset, 
                                                    batch_size=len(torch_train_dataset))

                train_torch_model(
                                self.model, 
                                self.model_optimizer, 
                                train_dataloader, 
                                nn.BCELoss(), 
                                self.model_k_steps
                            )

                self.model_opt_steps += 1
                #if self.plot:
                #self.plot_model(X, y, env)
                self.model_ready = True
                with torch.no_grad():
                    states = torch.from_numpy(states).type(torch.FloatTensor).to(self.device)
                    successes = torch.from_numpy(successes).type(torch.FloatTensor).to(self.device)
                    outputs = self.model(states)
                    acc = ((outputs > 0.5).float() == successes.reshape(-1,1)).float().mean()

                update_csv("nn_acc", float(acc), self.steps * self.num_envs, self.save_dir)


            # Optimize prior
            if not self.steps % max(self.prior_oe, 1):
                
                buffers = self.buffer.pre_sample()
                rollout_batch_size = buffers["episode_length"].shape[0] # buffer current size
                #history_length = np.arange(max(rollout_batch_size - 100, 0), rollout_batch_size)
                
                episode_idxs = np.random.choice(
                    rollout_batch_size,
                    size=self.prior_batch_size,
                    replace=True,
                    p=buffers["episode_length"][:, 0, 0]
                    / buffers["episode_length"][:, 0, 0].sum(),
                            )


                t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
                t_samples = np.random.randint(t_max_episodes)

                transitions = {
                        key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
                    }
                

                achieved_g = transitions["next_observation.achieved_goal"]
                if self.table_fetch:
                    achieved_g = achieved_g[:,:2] #TODO Remove hard_coded shape ?
                        
                self.prior.fit(achieved_g)
                self.prior_ready = True
                #if self.plot:
                #    self.plot_prior(env, achieved_g)


        return new_observation, reward, terminated, truncated, info



    def plot_prior(self,env, achieved_g):
        if not self.prior_opt_steps % max(self.prior_plot_freq, 1):
            plot_prior(
                    achieved_g, 
                    env,
                    self.steps*self.num_envs,
                    self.save_dir,
                    self.prior
                )


    def plot_particles(self, env):
        if not self.particles_opt_steps % max(self.particles_plot_freq, 1):
            plot_particles(
                    self.particles, 
                    self.criterion, 
                    self.steps*self.num_envs,
                    self.save_dir,
                    env
                )

    def plot_model(self, X, y, env):
        if not self.model_opt_steps % max(self.model_plot_freq, 1):
            plot_decision_boundary(
                                self.model, 
                                X, 
                                y,
                                env,
                                self.steps * self.num_envs,
                                self.save_dir
                            )
    
    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass


class SvggMCMCSetter(Setter, ABC):
    def __init__(
        self, 
        num_envs, 
        eval_env,
        buffer, 
        criterion, 
        model,
        model_optimizer,
        prior,
        plot,
        part_to_goals=lambda x: x.cpu().numpy(),
        save_dir=None
    ):
        super().__init__("SvggMCMCSetter")
        self.eval_env = eval_env
        self.num_envs = num_envs
        self.buffer = buffer
        self.criterion = criterion
        self.model = model
        self.model_optimizer = model_optimizer
        self.prior = prior
        self.save_dir = save_dir

        self.steps = 1
        assert  torch.cuda.is_available()
        self.device = torch.device("cuda")

        self.current_goals = None
        self.current_mcmc_samples = None
        self.maze_size = int(eval_env.size_max[0])+1
        
        # Prior param
        self.prior_oe = 1000 // num_envs
        self.prior_batch_size = 10_000
        self.prior_opt_steps = 0
        self.prior_ready = False

        # Model param
        self.model_oe = 100 // num_envs
        self.model_bs = 100
        self.model_hl = 300
        self.model_k_steps = 100
        self.model_opt_steps = 0
        self.model_ready = False

        # Misc.
        self.particles_plot_freq = 500
        self.model_plot_freq = 1
        self.prior_plot_freq = 1
        self.plot = plot
        self.annealed_freq = 5
        self.first_reset_done = False
        self.terminated = True
        self.table_fetch = False
        


    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        if not eval_mode and self.current_mcmc_samples is not None:
            self.first_reset_done = True
            # Replace Goals when episode is done
            new_goal_idxs = np.random.choice(len(self.current_mcmc_samples),self.num_envs)
            new_goals = self.current_mcmc_samples[new_goal_idxs]
            # Update current goals
            if self.current_goals is None:
                self.current_goals = np.copy(new_goals)
            else:
                self.current_goals = np.where(done == 1, new_goals, self.current_goals)
            observation["desired_goal"] = np.copy(self.current_goals)

        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        # Count agent steps and optimize the different modules
        self.steps += (1-eval_mode)
        
        if not eval_mode:
            # We have to change the desired goal at every step because the env.step
            # set the goal himself with the env._sample_goal function
            # Also we have to recompute the reward / success of the trajectory

            if self.current_goals is not None:
                new_observation, reward, terminated, info = self.process_transition(    self.current_goals,
                                                                                    new_observation, 
                                                                                    info, 
                                                                                    terminated
                                                                                    )

            # Optimize model and MCMC samples
            if not self.steps % max(self.model_oe, 1):

                buffers = copy.deepcopy(self.buffer.sample_recent(self.model_bs, self.model_hl)) # Train on recent data
                desired_g = buffers["observation.desired_goal"][:,0,:] # If same goal at every step of an episode
                init_ag = buffers["observation.achieved_goal"][:,0,:] # Init achieved goal

                X = np.concatenate((init_ag, desired_g),axis=1)
                y = buffers["is_success"].max(axis = 1)
                states = np.copy(X); successes = np.copy(y)

                # Check if there is more than 1 class
                if y.sum() > 0 and len(y) > y.sum():
                    oversample = RandomOverSampler()
                    X, y = oversample.fit_resample(X, y)

                X, y = torch.from_numpy(X), torch.from_numpy(y)
                X, y = X.type(torch.float).to(self.device), y.type(torch.float).to(self.device)
                torch_train_dataset = data.TensorDataset(X,y)
                train_dataloader = data.DataLoader(torch_train_dataset, 
                                                    batch_size=len(torch_train_dataset))

                train_torch_model(
                                self.model, 
                                self.model_optimizer, 
                                train_dataloader, 
                                nn.BCELoss(), 
                                self.model_k_steps
                            )

                self.model_opt_steps += 1
                self.model_ready = True
                with torch.no_grad():
                    states = torch.from_numpy(states).type(torch.FloatTensor).to(self.device)
                    successes = torch.from_numpy(successes).type(torch.FloatTensor).to(self.device)
                    outputs = self.model(states)
                    acc = ((outputs > 0.5).float() == successes.reshape(-1,1)).float().mean()

                update_csv("nn_acc", float(acc), self.steps * self.num_envs, self.save_dir)
                
                if self.prior_ready:
                    new_samples = self.mcmc_samples(10_000, self.criterion)
                    self.current_mcmc_samples = new_samples.cpu().numpy()
                



            # Optimize prior
            if not self.steps % max(self.prior_oe, 1):
                
                buffers = self.buffer.pre_sample()
                rollout_batch_size = buffers["episode_length"].shape[0] # buffer current size
                #history_length = np.arange(max(rollout_batch_size - 100, 0), rollout_batch_size)
                
                episode_idxs = np.random.choice(
                    rollout_batch_size,
                    size=self.prior_batch_size,
                    replace=True,
                    p=buffers["episode_length"][:, 0, 0]
                    / buffers["episode_length"][:, 0, 0].sum(),
                            )


                t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
                t_samples = np.random.randint(t_max_episodes)

                transitions = {
                        key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
                    }
                

                achieved_g = transitions["next_observation.achieved_goal"]
                if self.table_fetch:
                    achieved_g = achieved_g[:,:2] #TODO Remove hard_coded shape ?
                        
                self.prior.fit(achieved_g)
                self.prior_ready = True
                #if self.plot:
                #    self.plot_prior(env, achieved_g)


        return new_observation, reward, terminated, truncated, info
    
    
    def mcmc_samples(self, nb_steps, distrib):
        states = []
        burn_in = int(nb_steps*0.2)
        current = torch.FloatTensor((torch.rand(2) * self.maze_size) - torch.tensor([0.5,0.5])).to(self.device)
        
        for _ in range(nb_steps):
            states.append(current)
            movement = current + torch.cuda.FloatTensor(2).normal_()/2
            
            curr_prob = torch.exp(distrib.log_prob(current.unsqueeze(dim=0)))
            move_prob = torch.exp(distrib.log_prob(movement.unsqueeze(dim=0)))
                
            acceptance = min(float(move_prob/curr_prob),1)
            if self.random_coin(acceptance):
                current = movement
                       
        return torch.stack(states[burn_in:])
    
    def random_coin(self, p):
        unif = random.uniform(0,1)
        if unif>=p:
            return False
        else:
            return True


    def plot_prior(self,env, achieved_g):
        if not self.prior_opt_steps % max(self.prior_plot_freq, 1):
            plot_prior(
                    achieved_g, 
                    env,
                    self.steps*self.num_envs,
                    self.save_dir,
                    self.prior
                )


    def plot_particles(self, env):
        if not self.particles_opt_steps % max(self.particles_plot_freq, 1):
            plot_particles(
                    self.particles, 
                    self.criterion, 
                    self.steps*self.num_envs,
                    self.save_dir,
                    env
                )

    def plot_model(self, X, y, env):
        if not self.model_opt_steps % max(self.model_plot_freq, 1):
            plot_decision_boundary(
                                self.model, 
                                X, 
                                y,
                                env,
                                self.steps * self.num_envs,
                                self.save_dir
                            )
    
    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass


class GoalGanSetter(Setter, ABC):
    def __init__(
        self, 
        num_envs, 
        eval_env,
        buffer, 
        model,
        model_optimizer,
        gan_trainer,
        gan_gen,
        gan_disc,
        plot,
        part_to_goals=lambda x: x.cpu().numpy()
    ):
        super().__init__("GoalGanSetter")
        self.eval_env = eval_env
        self.num_envs = num_envs
        self.buffer = buffer        
        self.success_pred = model
        self.success_pred_optimizer = model_optimizer
        self.gan_trainer = gan_trainer
        self.gan_generator = gan_gen
        self.gan_discriminator = gan_disc
        self.steps = 1
        assert  torch.cuda.is_available()
        self.device = torch.device("cuda")

        # Init current goals
        self.current_goals = None

        # Model param
        self.model_oe = 5000 // num_envs
        self.model_bs = 100
        self.model_hl = 300
        self.model_k_steps = 100
        self.model_opt_steps = 0
        self.model_ready = False

        # Gan params
        self.gan_oe = 2000 // num_envs
        self.gan_warmup = 10_000 // num_envs
        self.noise_dim = 4
        self.gan_ready = False
        self.part_to_goals = part_to_goals
        
        # Prior param
        self.prior_oe = 1000 // num_envs
        self.prior_batch_size = 10_000
        self.prior_opt_steps = 0
        self.prior_ready = False

        # Misc.
        self.prior = None
        self.save_dir = None
        self.plot = plot
        self.first_reset_done = False
        self.terminated = True

    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        
        if not eval_mode and self.gan_ready:
            # Replace Goals when episode is done
            noise = torch.randn(self.num_envs, self.noise_dim).to(self.device)
            with torch.no_grad():
                new_goals = self.part_to_goals(self.gan_generator(noise, sig=False))
                new_goals+=np.random.randn(self.num_envs, new_goals.shape[1])/5
                
                
            # Update current goals
            if self.current_goals is None:
                self.current_goals = new_goals
            else:
                self.current_goals = np.where(done == 1, new_goals, self.current_goals)
            observation["desired_goal"] = np.copy(self.current_goals)

        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        # Count agent steps and optimize the different modules
        self.steps += (1-eval_mode)
        
        if not eval_mode:
            
            if self.current_goals is not None:
                new_observation, reward, terminated, info = self.process_transition(   
                                                                            self.current_goals,
                                                                            new_observation, 
                                                                            info, 
                                                                            terminated
                                                                            )
            # Optimize GAN
            if not self.steps % max(self.gan_oe, 1) and self.model_ready and self.steps > self.gan_warmup:
                self.gan_trainer.optimize(
                                        self.success_pred,
                                        self.buffer,
                                        self.prior            
                                        )
                self.gan_ready = True



            # Optimize model
            if not self.steps % max(self.model_oe, 1):
                
                buffers = copy.deepcopy(self.buffer.sample_recent(self.model_bs, self.model_hl)) # Train on recent data
                desired_g = buffers["observation.desired_goal"][:,0,:] # If same goal at every step of an episode
                init_ag = buffers["observation.achieved_goal"][:,0,:] # Init achieved goal

                X = np.concatenate((init_ag, desired_g),axis=1)
                y = buffers["is_success"].max(axis = 1)

                states = np.copy(X); successes = np.copy(y)

                # Check if there is more than 1 class
                if y.sum() > 0 and len(y) > y.sum():
                    oversample = RandomOverSampler()
                    X, y = oversample.fit_resample(X, y)

                X, y = torch.from_numpy(X), torch.from_numpy(y)
                X, y = X.type(torch.float).to(self.device), y.type(torch.float).to(self.device)
                torch_train_dataset = data.TensorDataset(X,y)
                train_dataloader = data.DataLoader(torch_train_dataset, 
                                                    batch_size=len(torch_train_dataset))

                train_torch_model(
                                self.success_pred, 
                                self.success_pred_optimizer, 
                                train_dataloader, 
                                nn.BCELoss(), 
                                self.model_k_steps
                            )

                self.model_opt_steps += 1
                self.model_ready = True

                with torch.no_grad():
                    states = torch.from_numpy(states).type(torch.FloatTensor).to(self.device)
                    successes = torch.from_numpy(successes).type(torch.FloatTensor).to(self.device)
                    outputs = self.success_pred(states)
                    acc = ((outputs > 0.5).float() == successes.reshape(-1,1)).float().mean()

                update_csv("nn_acc", float(acc), self.steps * self.num_envs, self.save_dir)
                
                
                
            # Optimize prior
            if self.prior is not None and not self.steps % max(self.prior_oe, 1):
                
                buffers = self.buffer.pre_sample()
                rollout_batch_size = buffers["episode_length"].shape[0] # buffer current size
                #history_length = np.arange(max(rollout_batch_size - 100, 0), rollout_batch_size)
                
                episode_idxs = np.random.choice(
                    rollout_batch_size,
                    size=self.prior_batch_size,
                    replace=True,
                    p=buffers["episode_length"][:, 0, 0]
                    / buffers["episode_length"][:, 0, 0].sum(),
                            )


                t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
                t_samples = np.random.randint(t_max_episodes)

                transitions = {
                        key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
                    }
                

                achieved_g = transitions["next_observation.achieved_goal"]            
                self.prior.fit(achieved_g)
                self.prior_ready = True

        return new_observation, reward, terminated, truncated, info
        


    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass




class DensitySetter(Setter, ABC):
    def __init__(self, num_envs, eval_env, buffer, kde):
        super().__init__("DensitySetter")
        self.num_envs = num_envs
        self.buffer = buffer
        self.kde = kde
        self.density_oe = 5000 // num_envs
        self.steps = 0
        self.num_ag_candidate = 500
        self.density_batch_size = 10000
        self.density_opt_steps = 0
        self.fitted_kde = None
        self.kde_ready = False
        self.kde_sample_mean = 0.
        self.kde_sample_std = 1.
        self.save_dir = None
        self.warmup = 1000 // num_envs

        self.terminated = False
        self.eval_env = eval_env
        self.current_goals = None
        self.min_ags = None
        self.randomize = False
        self.alpha = -1.


    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):

        if not eval_mode and self.kde_ready:
            
            if self.current_goals is None:
                self.current_goals = np.copy(self.min_ags)
            else:
                self.current_goals = np.where(done == 1, np.copy(self.min_ags), self.current_goals)      

        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        # Count agent steps and optimize the different modules
        self.steps += (1-eval_mode)

        if not eval_mode:
            # Relabel if there is some current goals
            if self.current_goals is not None:
                    new_observation, reward, terminated, info = self.process_transition(    
                                                                            self.current_goals,
                                                                            new_observation, 
                                                                            info, 
                                                                            terminated
                                                                            )

            # Optimize Density Model
            if not self.steps % max(self.density_oe, 1) and self.steps > self.warmup:

                self.density_opt_steps+=1
                self.kde_ready = True

                # sample batch
                buffers = self.buffer.pre_sample()
                current_size = buffers["episode_length"].shape[0] # buffer current size

                episode_idxs = np.random.choice(
                    current_size,
                    size=self.density_batch_size,
                    replace=True,
                    p=buffers["episode_length"][:, 0, 0]
                    / buffers["episode_length"][:, 0, 0].sum(),
                        )

                
                t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
                t_samples = np.random.randint(t_max_episodes)

                transitions = {key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()}

                kde_samples = transitions["next_observation.achieved_goal"]
                #if not self.density_opt_steps % max(1, 1):
                #    plot_prior(
                #        kde_samples[::10],
                #        env,
                #        self.steps*self.num_envs,
                #        self.save_dir
                #    )

                # Normalize samples
                self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
                self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
                norm_kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std

                self.fitted_kde = self.kde.fit(norm_kde_samples)
                
                
                # Sample min ags
                rand_idx = np.random.permutation(len(kde_samples))
                ag_candidates = kde_samples[rand_idx]

                scores_flat = self.fitted_kde.score_samples((ag_candidates  - self.kde_sample_mean) / self.kde_sample_std )
                scores = scores_flat.reshape(self.num_envs, -1)
                normalized_inverse_densities = softmax(scores * self.alpha)
                normalized_inverse_densities *= -1.  # make negative / reverse order so that lower is better.

                if self.randomize:  # sample proportional to the absolute score
                    abs_goal_values = np.abs(normalized_inverse_densities)
                    normalized_values = abs_goal_values / np.sum(abs_goal_values, axis=1, keepdims=True)
                    chosen_idx = (normalized_values.cumsum(1) > np.random.rand(normalized_values.shape[0])[:, None]).argmax(1)

                else:
                    #chosen_idx = np.random.permutation(np.argsort(scores_flat)[:self.num_envs])
                    candidates_idx = np.argsort(scores_flat)[:100]
                    chosen_idx = np.random.choice(candidates_idx, self.num_envs)
                    chosen_idx_1 = np.argmin(scores, axis=1)
                    chosen_idx_2 = np.argmin(normalized_inverse_densities, axis=1)

                self.min_ags = np.copy(ag_candidates[chosen_idx])
                
        return new_observation, reward, terminated, truncated, info        


    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass




class CompositeSetter(Setter, ABC):
    def __init__(self, setter1: Setter, setter2: Setter):
        super().__init__("CompositeSetter")
        self.setter1 = setter1
        self.setter2 = setter2

    def reset(self, env, observation, info, eval_mode=False):
        obs_, info_ = self.setter1.reset(env, observation, info, eval_mode)
        return self.setter2.reset(env, obs_, info_, eval_mode)

    def reset_done(self, env, observation, info, done, eval_mode=False):
        obs_, info_, done_ = self.setter1.reset_done(
            env, observation, info, done, eval_mode
        )
        return self.setter2.reset_done(env, obs_, info_, done_, eval_mode)

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):
        new_obs_, reward_, terminated_, truncated_, info_ = self.setter1.step(
            env,
            observation,
            action,
            action_info,
            new_observation,
            reward,
            terminated,
            truncated,
            info,
            eval_mode,
        )
        return self.setter2.step(
            env,
            observation,
            action,
            action_info,
            new_obs_,
            reward_,
            terminated_,
            truncated_,
            info_,
            eval_mode,
        )

    def write_config(self, output_file: str):
        self.setter1.write_config(output_file + ".1")
        self.setter2.write_config(output_file + ".2")

    def save(self, directory: str):
        self.setter1.save(os.path.join(directory, "1"))
        self.setter2.save(os.path.join(directory, "2"))

    def load(self, directory: str):
        self.setter1.load(os.path.join(directory, "1"))
        self.setter2.load(os.path.join(directory, "2"))
