import numpy as np
import hashlib
from tqdm import tqdm
from dreams.utils.spectra import bin_peak_list, bin_peak_lists


class RandomProjection:
    def __init__(self, n_elems: int, n_hyperplanes: int, seed=3):
        np.random.seed(seed)
        # if (n_hyperplanes % 64) != 0: 
        #     n_hyperplanes_ = round(n_hyperplanes / 64) * 64 if n_hyperplanes > 64 else 64
        #     print(f"n_hyperplanes ({n_hyperplanes}) must be positive and divisible by 64. rounding it to {n_hyperplanes_}.")
        #     n_hyperplanes = n_hyperplanes_

        self.H = np.random.randn(n_hyperplanes, n_elems)

    def compute(self, x: np.array, as_str=True, batched=False):
        if batched:
            proj_signs = np.einsum('ij,kj->ki', self.H, x) >= 0
        else:
            proj_signs = self.H @ x >= 0

        if as_str:
            # Convert each row of the boolean array to bytes and hash using SHA-256
            proj_signs_hashed = np.apply_along_axis(
                lambda row: hashlib.sha256(row.tobytes()).hexdigest(), 1, proj_signs
            )
            return proj_signs_hashed
            
        return proj_signs

        # if as_int:
        #     # Binary representation of a boolean array as integer 
        #     print('-'*50)   
        #     proj_signs_u8 = proj_signs.astype(np.uint8) # N, n_hyp
        #     print(proj_signs_u8.shape)
        #     proj_signs_u64 = np.packbits(proj_signs_u8, axis=1).view(np.uint64) # N, (n_hyp//8)
        #     print(proj_signs_u64.shape)
        #     print(proj_signs_u64[0])
        #     proj_signs = np.bitwise_xor.reduce(proj_signs_u64, axis=1) # N,
        #     print(proj_signs[0])

        return proj_signs


class PeakListRandomProjection:
    def __init__(self, bin_step=0.5, max_mz=1000., n_hyperplanes=64, seed=3):
        assert (max_mz / bin_step) % 1 == 0
        self.bin_step = bin_step
        self.max_mz = max_mz
        self.rand_projection = RandomProjection(n_elems=int(max_mz / bin_step), n_hyperplanes=n_hyperplanes, seed=seed)

    def compute(self, peak_list: np.array, as_int=True):
        bpl = bin_peak_list(peak_list, self.max_mz, self.bin_step)
        return self.rand_projection.compute(bpl, as_int=as_int)


class BatchedPeakListRandomProjection(PeakListRandomProjection):
    """
    Uses numba code to compute hashes for peak_lists given in a shape (num_peak_lists, 2, num_peaks).
    If subbatch_size is specified additionally splits peak lists into subbatch_size (should be used when dataset is
    large).
    """

    def __init__(self, bin_step=0.5, max_mz=1000., n_hyperplanes=64, subbatch_size=32, seed=3):
        super().__init__(bin_step, max_mz, n_hyperplanes, seed)
        self.subbatch_size = subbatch_size

    def __compute_batch(self, peak_lists: np.array, as_int: bool):
        bpls = bin_peak_lists(peak_lists, self.max_mz, self.bin_step)
        return self.rand_projection.compute(bpls, as_int=as_int, batched=True)

    def compute(self, peak_lists: np.array, as_int=True, logger=None, progress_bar=True):
        n = peak_lists.shape[0]
        
        if not self.subbatch_size or self.subbatch_size >= n:
            return self.__compute_batch(peak_lists, as_int=as_int)

        lshs = []
        batch_idx = range(0, n, self.subbatch_size)
        
        with tqdm(total=n, disable=not progress_bar, desc='Computing LSHs') as pbar:
            for i in batch_idx:
                if logger:
                    logger.info(f'Computing LSH for batch [{i}:{i+self.subbatch_size}] (out of {n})...')
                lshs.append(self.__compute_batch(peak_lists[i:i+self.subbatch_size, ...], as_int=as_int))
                pbar.update(min(self.subbatch_size, n - i))
        
        return np.concatenate(lshs)

