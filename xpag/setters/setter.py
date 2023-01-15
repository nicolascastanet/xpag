# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

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
from torch.utils import data
import numpy as np

# SVGG import
from xpag.svgg.svgd import RBF, SVGD
from xpag.tools.models import train_torch_model
from xpag.plotting.plotting import plot_decision_boundary, plot_particles, plot_prior



class Setter(ABC):
    def __init__(self, name: str):
        self.name = name

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

        reward = self.eval_env.compute_reward(obs["achieved_goal"], goals, info
        ).reshape((self.num_envs, 1))

        return obs, reward, terminated, info




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





class UniformSetter(Setter, ABC):
    def __init__(self):
        super().__init__("UniformSetter")

    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        
        goal = np.where(done == 1, env.sample_uniform_goal(), env.goal)
        env.set_goal(np.copy(goal))
        observation["desired_goal"] = np.copy(goal)

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
        plot
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

        self.steps = 1
        assert  torch.cuda.is_available()
        self.device = torch.device("cuda")

        # Particles param
        self.particles_oe = 50 // num_envs
        self.particles_opt_steps = 0
        self.particles_period = 5000 // num_envs
        self.n = self.particles.shape[0]

        # Init current goals
        init_goal_idxs = np.random.choice(self.n,self.num_envs)
        self.current_goals = self.particles.cpu()[init_goal_idxs]

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
        self.save_dir = None
        self.particles_plot_freq = 250
        self.model_plot_freq = 1
        self.prior_plot_freq = 1
        self.plot = plot
        self.annealed_freq = 5
        self.first_reset_done = False
        self.terminated = True


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
            new_goals = self.particles.cpu()[new_goal_idxs]

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

            #import ipdb;ipdb.set_trace()
            
            # Optimize particles
            if not self.steps % max(self.particles_oe, 1) and self.model_ready and self.prior_ready:    
                
                if (self.steps % self.particles_period) < self.particles_period / self.annealed_freq:
                    annealed = 0
                else:
                    annealed = 1

                self.svgd.step(self.particles, annealed)
                self.particles_opt_steps += 1
                #if self.plot:
                #self.plot_particles(env)
        
            # Optimize model
            if not self.steps % max(self.model_oe, 1):

                buffers = copy.deepcopy(self.buffer.sample_recent(self.model_bs, self.model_hl)) # Train on recent data
                desired_g = buffers["observation.desired_goal"][:,0,:] # If same goal at every step of an episode
                init_ag = buffers["observation.achieved_goal"][:,0,:] # Init achieved goal

                X = np.concatenate((init_ag, desired_g),axis=1)
                y = buffers["is_success"].max(axis = 1)

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
                self.prior.fit(achieved_g)
                self.prior_ready = True
                #if self.plot:
                #self.plot_prior(env, achieved_g)


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
                    env,
                    self.steps*self.num_envs,
                    self.save_dir
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




class DensitySetter(Setter, ABC):
    def __init__(self, num_envs, buffer, kde):
        super().__init__("SvggSetter")
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



    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):

        if not eval_mode and self.kde_ready:

            # sample random achieved goals
            buffers = self.buffer.pre_sample()
            current_size = buffers["episode_length"].shape[0]
            
            episode_idxs = np.random.choice(
                current_size,
                size=self.num_ag_candidate,
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

            scores = self.fitted_kde.score_samples((ag_candidates  - self.kde_sample_mean) / self.kde_sample_std )

            min_density_idx = np.random.permutation(np.argsort(scores)[:self.num_envs])
            min_density_ags = ag_candidates[min_density_idx]

            goal = np.where(done == 1, min_density_ags, env.goal)
            env.set_goal(np.copy(goal))
            observation["desired_goal"] = np.copy(goal)

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
            # Optimize Density Model
            if not self.steps % max(self.density_oe, 1) and self.steps > self.warmup:
                
                #pr = cProfile.Profile()
                #pr.enable()
                
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
                kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std

                self.fitted_kde = self.kde.fit(kde_samples)

                #pr.disable()
                #s = io.StringIO()
                #sortby = SortKey.CUMULATIVE
                #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                #ps.print_stats()
                #print(s.getvalue())


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
