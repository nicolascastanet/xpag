from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta
import numpy as np


class Criterion(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def write_config(self, output_file: str):
        pass

    @abstractmethod
    def save(self, directory: str):
        pass

    @abstractmethod
    def load(self, directory: str):
        pass



class AlphaBetaDifficulty(Criterion, ABC):
    def __init__(
        self, 
        alpha, 
        beta, 
        model,
        prior, 
        init_state, 
        temp=20
    ):
        super().__init__("AlphaBetaDifficulty")

        self.device = torch.device("cuda")

        # Distribution parameters
        self.alpha = torch.tensor(alpha).to(self.device)
        self.beta = torch.tensor(beta).to(self.device)
        self.beta_distrib = Beta(self.alpha,self.beta)
        self.temp = temp

        # We need the model of the agent's skills and the environment to compute the criterion
        self.model = model
        self.init_state = init_state
        self.prior = prior
        self.only_prior = False


    def log_prob(self, goals, eval_mode=False, log=True):  

        if self.only_prior:
            return self.prior.log_prob(goals)

        init_obs_batch = self.init_state.repeat(goals.shape[0],1).to(self.device)

        goals_init_obs = torch.cat((init_obs_batch,goals),dim=1)
        probas = self.model(goals_init_obs)

        criterion = torch.exp(self.beta_distrib.log_prob(probas))

        return self.temp * criterion + self.prior.log_prob(goals)


    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass


class MinDensity(Criterion, ABC):
    def __init__(self, density, env):
        super().__init__("MinDensity")

        assert  torch.cuda.is_available()
        self.device = torch.device("cuda")

        self.density = density
        self.env = env

    def score_goals(self, goals, eval_mode=False):

        num_envs, num_sampled_ags = goals.shape[:2]

        # score the goals to get log densities, and exponentiate to get densities
        flattened_sampled_ags = goals.reshape(num_envs * num_sampled_ags, -1)
        sampled_ag_scores = density_module.evaluate_log_density(flattened_sampled_ags)
        sampled_ag_scores = sampled_ag_scores.reshape(num_envs, num_sampled_ags)  # these are log densities

        # Take softmax of the alpha * log density.
        # If alpha = -1, this gives us normalized inverse densities (higher is rarer)
        # If alpha < -1, this skews the density to give us low density samples
        normalized_inverse_densities = softmax(sampled_ag_scores * self.alpha)
        normalized_inverse_densities *= -1.

        return criterion


    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass

