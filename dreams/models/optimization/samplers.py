import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, BatchSampler
from typing import Iterator


class _MaxVarSampler(DistributedSampler):
    """
    NOTE: Should be used only in conjunction with MaxVarBatchSampler.
    NOTE: Implements drop_last behaviour by default.
    TODO: Generalize to non-distributed version?
    TODO: Docstrings.
    """
    def __init__(self, dataset: Dataset, max_var_features: np.ndarray, batch_size: int, swap_width=5):
        assert len(dataset) == len(max_var_features)
        super().__init__(dataset, shuffle=True, drop_last=True)

        self.max_var_features = max_var_features
        self.batch_size = batch_size
        self.swap_width = swap_width

        # Sort indices wrt feature values
        self.indices = torch.from_numpy(np.argsort(max_var_features))

    def __iter__(self) -> Iterator:

        # Deterministically shuffle indices based on epoch and seed by swapping elements with up to `swap_width`
        # position distances
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        swap_idx1 = torch.arange(0, len(self.indices), self.swap_width)
        swap_idx2 = swap_idx1 + torch.randint(self.swap_width, swap_idx1.shape, generator=g)
        self.indices[swap_idx1], self.indices[swap_idx2] = self.indices[swap_idx2], self.indices[swap_idx1]

        # Remove the  tail of data to make it evenly divisible by the num. of devices
        indices = self.indices[:self.total_size]
        assert len(indices) == self.total_size

        # Subset indices based on device rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # Remove the tail of data to make it evenly divisible by the num. of batches
        num_batches = self.num_samples // self.batch_size
        indices = indices[:self.batch_size * num_batches]

        # Split indices to batches maximizing the distances between their positions in `indices` with .t()
        indices = indices.view(self.batch_size, num_batches).t()

        # Deterministically permute batches
        indices = indices[torch.randperm(num_batches, generator=g)]

        # Flatten batches
        indices = indices.flatten().tolist()

        # Return iterator over reordered sample indices
        return iter(indices)


class MaxVarBatchSampler(BatchSampler):
    def __init__(self, dataset: Dataset, max_var_features: np.ndarray, batch_size: int, swap_width=5):
        sampler = _MaxVarSampler(dataset, max_var_features, batch_size, swap_width)
        super().__init__(sampler=sampler, batch_size=batch_size, drop_last=True)
