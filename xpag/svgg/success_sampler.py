import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod


class GoalSuccessSampler(ABC):
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





class NN_predictor(GoalSuccessSampler):
    def __init__(
        self,
        name,
        model,
        buffer,
        history_length = 200, 
        optimize_every=250
    ):
        super().__init__("nn_predictor")
        self.history_length = history_length
        self.buffer = buffer


    def sample_trajectories(self):
        buffers = self.buffer.pre_sample()
        episode_max = buffers["episode_length"].shape[0]
        episode_range = self.history_length
        
        episode_idxs = np.arange(episode_max - episode_range, episode_max)
        t_max_episodes = buffers["episode_length"][episode_idxs, 0]
        t_max_episodes = t_max_episodes.flatten().astype(int)
        
        trajectories = {
            key: buffers[key][episode_idxs, t_max_episodes-1] for key in buffers.keys()
            }

        return None, None , None
        # TODO
