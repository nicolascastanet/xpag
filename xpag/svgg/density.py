from abc import ABC, abstractmethod

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.special import entr
from xpag.svgg.svgd import RBF, SVGD

class Density(ABC):
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


class RawKernelDensity(Density):
    """
    A KDE-based density model for raw items in the replay buffer (e.g., states/goals).
    """
    def __init__(self, item, optimize_every=10, samples=10000, kernel='gaussian', bandwidth=0.1, normalize=True, 
        log_entropy=False, tag='', buffer_name='replay_buffer'):


        super().__init__("KernelDensity")
        self.step = 0
        self.item = item
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        self.optimize_every = optimize_every
        self.samples = samples
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.kde_sample_mean = 0.
        self.kde_sample_std = 1.
        self.fitted_kde = None
        self.ready = False
        self.log_entropy = log_entropy
        self.buffer_name = buffer_name


    def _optimize(self, force=False):
        buffer = getattr(self, self.buffer_name).buffer.BUFF['buffer_' + self.item]
        self.step +=1

        if force or (self.step % self.optimize_every == 0 and len(buffer)):
            self.ready = True
            sample_idxs = np.random.randint(len(buffer), size=self.samples)
            kde_samples = buffer.get_batch(sample_idxs)
            #og_kde_samples = kde_samples

        if self.normalize:
            self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
            self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
            kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std

        self.fitted_kde = self.kde.fit(kde_samples)

      # Now also log the entropy
        if self.log_entropy and hasattr(self, 'logger') and self.step % 250 == 0:
            # Scoring samples is a bit expensive, so just use 1000 points
            num_samples = 1000
            s = self.fitted_kde.sample(num_samples)
            entropy = -self.fitted_kde.score(s)/num_samples + np.log(self.kde_sample_std).sum()
            self.logger.add_scalar('Explore/{}_entropy'.format(self.module_name), entropy, log_every=500)

    def evaluate_log_density(self, samples):
        assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
        return self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )

    def evaluate_elementwise_entropy(self, samples, beta=0.):
        """ Given an array of samples, compute elementwise function of entropy of the form:

            elem_entropy = - (p(samples) + beta)*log(p(samples) + beta)

        Args:
          samples: 1-D array of size N
          beta: float, offset entropy calculation

        Returns:
          elem_entropy: 1-D array of size N, elementwise entropy with beta offset
        """
        assert self.ready, "ENSURE READY BEFORE EVALUATING ELEMENT-WISE ENTROPY"
        log_px = self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )
        px = np.exp(log_px)
        elem_entropy = entr(px + beta)
        return elem_entropy
