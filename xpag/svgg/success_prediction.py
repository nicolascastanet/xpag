import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod


class GoalSuccessPredictor(ABC):
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





class NN_predictor(GoalSuccessPredictor):
    def __init__(
        self,
        name,
        model,
        buffer,
        batch_size = 50, 
        history_length = 200, 
        optimize_every=250
    ):
        super().__init__("nn_predictor")
        self.batch_size = batch_size
        self.history_length = history_length
        self.optimize_every = optimize_every
        self.buffer = buffer

        assert  torch.cuda.is_available()
        self.device = torch.device("cuda")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())


    def optimize(self):
        inputs, targets, behav_goals = self.sample_trajectories()
        self.optimize_model(inputs, targets, behav_goals)


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

        pass
        # TODO

    def optimize_model(
        self, 
        inputs, 
        targets, 
        ):
  
        for _ in range(self.k_steps):

        # Shuffle data at each epoch
            perm = np.random.permutation(self.history_length)
            Xtrain = inputs[perm]
            ytrain = targets[perm]

            for j in range(self.history_length // self.n_batch):

                # Get mini batch
                indsBatch = range(j * self.n_batch, (j+1) * self.n_batch)
                X = Xtrain[indsBatch]
                y = ytrain[indsBatch]

                # outputs here have not been passed through sigmoid
                outputs = self.model(X)

                loss = F.binary_cross_entropy_with_logits(outputs, y)

                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def __call__(self, *states_and_maybe_goals):
        """Input / output are numpy arrays"""
        states = np.concatenate(states_and_maybe_goals, -1)
        return self.numpy(torch.sigmoid(self.model(self.torch(states))))
