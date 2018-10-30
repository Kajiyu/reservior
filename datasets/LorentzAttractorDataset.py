# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import collections


# LorentzAttractorDataset
class LorentzAttractorDataset(Dataset):
    """
    Mackey Glass dataset
    """

    # Constructor
    def __init__(self, sample_len, n_samples, tau=17, seed=None):
        """
        Constructor
        :param sample_len: Length of the time-series in time steps.
        :param n_samples: Number of samples to generate.
        :param tau: Delay of the MG with commonly used value of tau=17 (mild chaos) and tau=30 is moderate chaos.
        :param seed: Seed of random number generator.
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.tau = tau
        self.delta_t = 10
        self.timeseries = 1.2
        self.history_len = tau * self.delta_t

        # Init seed if needed
        if seed is not None:
            torch.manual_seed(seed)
        # end if
    # end __init__

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return self.n_samples
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # History
        history = collections.deque((torch.rand(self.history_len, 3)))

        # Preallocate tensor for time-serie
        inp = torch.zeros(self.sample_len, 3)

        # For each time step
        for timestep in range(self.sample_len):
            for _ in range(self.delta_t):
                xtau = history.popleft()
                history.append(self.timeseries)
                x_dot, y_dot, z_dot = self.lorenz(xs[i], ys[i], zs[i])
                self.timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - 0.1 * history[-1]) / self.delta_t
                self.timeseries[]
            # end for
            inp[timestep] = self.timeseries
        # end for

        # Squash timeseries through tanh
        return torch.tan(inp - 1)
    # end __getitem__

    def lorenz(x, y, z, s=10, r=28, b=2.667):
        '''
        Given:
        x, y, z: a point of interest in three dimensional space
        s, r, b: parameters defining the lorenz attractor
        Returns:
        x_dot, y_dot, z_dot: values of the lorenz attractor's partial
        derivatives at the point x, y, z
        '''
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot

# end LorentzAttractorDataset