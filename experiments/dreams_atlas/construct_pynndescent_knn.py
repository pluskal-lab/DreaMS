import h5py
import numpy as np
import pynndescent
from pathlib import Path
from msml.utils.io import setup_logger
from msml.utils.data import CSRKNN
from msml.definitions import *


def main():

    k = 3
    out_dir = Path('/storage/plzen1/home/romanb/DreaMS_Atlas')
    lib_pth = Path('/auto/brno2/home/romanb/msml/msml/data/merged/datasets/nist20_mona_clean_merged_spectra_dreams.hdf5')
    gems_pth = Path('/storage/plzen1/home/romanb/msvn_C/msvn_C_H1000_KK1.merged.hdf5')

    d = 40_000_000
    out_pth = out_dir / f'DreaMS_Atlas_{k}NN_after{d}.npz'
    logger = setup_logger(out_pth.with_suffix('.log'))

    # Load spectral library embeddings
    # logger.info(f'Loading embeddings from {lib_pth}.')
    # f = h5py.File(lib_pth, 'r')
    # embs_lib = f[DREAMS_EMBEDDING][:]
    # embs_lib = embs_lib.astype(np.float32)
    # f.close()
    # logger.info(f'Loaded {embs_lib.shape[0]} embeddings.')

    # Load GeMS embeddings
    logger.info(f'Loading embeddings from {gems_pth}.')
    f = h5py.File(gems_pth, 'r')
    embs_gems = f[DREAMS_EMBEDDING]
    embs_gems = embs_gems[d:]
    embs_gems = embs_gems.astype(np.float32)
    f.close()
    logger.info(f'Loaded {embs_gems.shape[0]} embeddings.')

    # Concatenate embeddings
    logger.info('Concatenating embeddings.')
    # all_embs = np.vstack([embs_lib, embs_gems])
    # del embs_lib, embs_gems
    all_embs = embs_gems
    del embs_gems

    # Create PyNNDescent index
    logger.info('Creating PyNNDescent index.')
    pynn_knn = pynndescent.PyNNDescentTransformer(
        metric='cosine', n_neighbors=k, search_epsilon=0.25, n_jobs=1, low_memory=True, verbose=True
    ).fit_transform(all_embs)

    logger.info('Initializing CSRKNN object.')
    knn = CSRKNN(pynn_knn)

    logger.info(f'Saving k-NN to {out_pth}.')
    knn.to_npz(out_pth)

    logger.info('Done.')


if __name__ == '__main__':
    main()
