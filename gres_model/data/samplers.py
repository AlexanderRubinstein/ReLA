import torch
from torch.utils.data.sampler import Sampler
import random

class RandomSubsetTrainingSampler(Sampler):
    """
    Randomly sample a subset of the dataset.
    """
    def __init__(self, size, ratio):
        """
        Args:
            size (int): the total number of data of the underlying dataset
            ratio (float): the ratio of data to sample
        """
        self.size = size
        self.ratio = ratio
        self.num_samples = int(size * ratio)
        self.indices = list(range(size))
        random.shuffle(self.indices)
        self.indices = self.indices[:self.num_samples]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples
