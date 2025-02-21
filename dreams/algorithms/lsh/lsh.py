import numpy as np
import hashlib
from tqdm import tqdm
from dreams.utils.spectra import bin_peak_list, bin_peak_lists


class RandomProjection:
    def __init__(self, n_elems: int, n_hyperplanes: int, seed=3):
        np.random.seed(seed)
        self.H = np.random.randn(n_hyperplanes, n_elems)

    def __arr_to_str_hash(self, arr: np.array) -> str:
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def compute(self, x: np.array, as_str=True, batched=False):
        if batched:
            if x.ndim != 2:
                raise ValueError(f'x must be 2D array, got {x.ndim}D array')
            proj_signs = np.einsum('ij,kj->ki', self.H, x) >= 0
            if as_str:
                # Convert each row of the boolean array to bytes and hash using SHA-256
                proj_signs = np.apply_along_axis(self.__arr_to_str_hash, 1, proj_signs).astype('S64')
        else:
            if x.ndim != 1:
                raise ValueError(f'x must be 1D array, got {x.ndim}D array')
            proj_signs = self.H @ x >= 0
            if as_str:
                proj_signs = self.__arr_to_str_hash(proj_signs)

        return proj_signs


class PeakListRandomProjection:
    def __init__(self, bin_step=1, max_mz=1000., n_hyperplanes=50, seed=3):
        assert (max_mz / bin_step) % 1 == 0
        self.bin_step = bin_step
        self.max_mz = max_mz
        self.rand_projection = RandomProjection(n_elems=int(max_mz / bin_step), n_hyperplanes=n_hyperplanes, seed=seed)

    def compute(self, peak_list: np.array, as_str=True):
        if peak_list.ndim != 2:
            raise ValueError(f'peak_list must be 2D array, got {peak_list.ndim}D array')
        bpl = bin_peak_list(peak_list, self.max_mz, self.bin_step)
        return self.rand_projection.compute(bpl, as_str=as_str)


class BatchedPeakListRandomProjection(PeakListRandomProjection):
    """
    Uses numba code to compute hashes for peak_lists given in a shape (num_peak_lists, 2, num_peaks).
    If subbatch_size is specified additionally splits peak lists into subbatch_size (should be used when dataset is
    large).
    """

    # TODO: change default parameters
    def __init__(self, bin_step=1, max_mz=1000., n_hyperplanes=50, subbatch_size=32, seed=3):
        super().__init__(bin_step, max_mz, n_hyperplanes, seed)
        self.subbatch_size = subbatch_size

    def __compute_batch(self, peak_lists: np.array, as_str: bool):
        bpls = bin_peak_lists(peak_lists, self.max_mz, self.bin_step)
        return self.rand_projection.compute(bpls, as_str=as_str, batched=True)

    def compute(self, peak_lists: np.array, as_str=True, logger=None, progress_bar=True):
        if peak_lists.ndim != 3:
            raise ValueError(f'peak_lists must be 3D array, got {peak_lists.ndim}D array')

        n = peak_lists.shape[0]
        
        if not self.subbatch_size or self.subbatch_size >= n:
            return self.__compute_batch(peak_lists, as_str=as_str)

        lshs = []
        batch_idx = range(0, n, self.subbatch_size)
        
        with tqdm(total=n, disable=not progress_bar, desc='Computing LSHs') as pbar:
            for i in batch_idx:
                if logger:
                    logger.info(f'Computing LSH for batch [{i}:{i+self.subbatch_size}] (out of {n})...')
                lshs.append(self.__compute_batch(peak_lists[i:i+self.subbatch_size, ...], as_str=as_str))
                pbar.update(min(self.subbatch_size, n - i))
        
        return np.concatenate(lshs)

