import h5py
import numpy as np
import seaborn as sns
import pandas as pd
# import nndescent
# import pynndescent
import ngtpy
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
from msml.utils.io import setup_logger, TqdmToLogger
from msml.utils.data import CSRKNN


def load_gems_embs(pth, logger):
    # TODO: This is a temporary fix, we need to get rid of blank samples in a better way
    with h5py.File(pth, 'r') as f:
        # Get rid of "blank" samples
        gemc_C_names = pd.Series(f['name'][:]).astype(str)
        logger.info(f'Num. of spectra: {gemc_C_names.shape}')
        blank_substrs = ['blank', 'wash', 'no_inj', 'noinj', 'empty', 'solvent']
        gemc_C_names = gemc_C_names[gemc_C_names.apply(lambda n: all([s not in n.lower() for s in blank_substrs]))]
        idx_gems = gemc_C_names.index.tolist()
        logger.info(f'Num. of non-blank spectra: {gemc_C_names.shape}')
        # Get embeddings
        gemc_C_embs = f['DreaMS_Embedding'][:][idx_gems]
    return gemc_C_embs


def main():

    k = 3
    # lib_pth = Path('/auto/brno2/home/romanb/msml/msml/data/merged/datasets/nist20_mona_clean_A_merged_spectra_dreams.pkl')
    lib_pth = Path('/auto/brno2/home/romanb/msml/msml/data/merged/datasets/nist20_mona_clean_merged_spectra_dreams.hdf5')
    gems_dir = Path('/storage/plzen1/home/romanb/msvn_C')
    out_dir = Path('/storage/plzen1/home/romanb/DreaMS_Atlas')
    name = f'DreaMS_Atlas_{k}NN_ngt_float16'
    out_pth = out_dir / f'{name}.npz'
    logger = setup_logger(out_pth.with_suffix('.log'))
    tqdm_logger = TqdmToLogger(logger)

    gems_pths = list(gems_dir.glob('msvn_C_H1000_KK1.*.hdf5'))
    gems_pths = sorted(gems_pths, key=lambda p: int(p.name.split('.')[-2]))  # Sort by chunk ids

    # Load spectral library embeddings
    logger.info(f'Loading embeddings from {lib_pth}.')
    f = h5py.File(lib_pth, 'r')
    embs = f['DreaMS_embedding'][:]
    embs = embs.astype(np.float16)
    f.close()

    # Create NGT index
    logger.info('Creating NGT index.')
    ngtpy.create(
        str(out_dir / f'{name}'),
        dimension=embs.shape[1],
        distance_type='Cosine',
        object_type='Float16',
        edge_size_for_creation=30,
        edge_size_for_search=60
    )
    ngt_index = ngtpy.Index(str(out_dir / f'{name}'))

    logger.info('Inserting spectral library embeddings into NGT index.')
    ngt_index.batch_insert(embs)
    ngt_index.save()

    # Load GeMS embeddings from all chunks
    for p in tqdm(gems_pths, desc='Adding GeMS chunks to NGT index', file=tqdm_logger):
        logger.info(f'Loading embeddings from {p}.')
        embs = load_gems_embs(p, logger)

        num_nans = np.count_nonzero(np.isnan(embs))
        logger.info(f'Num. of NaNs: {num_nans}.')
        if num_nans > 0:
            embs = np.nan_to_num(embs)
            logger.info(f'Num. of NaNs after replacing: {np.count_nonzero(np.isnan(embs))}.')

        embs = embs.astype(np.float16)

        logger.info('Inserting GeMS embeddings into NGT index.')
        ngt_index.batch_insert(embs)
        logger.info('Saving NGT index.')
        ngt_index.save()
        del embs

    logger.info('Constructing k-NN graph.')
    knn_i, knn_j, knn_w = [], [], []
    num_embs = ngt_index.get_num_of_objects()
    for i in tqdm(range(num_embs), desc='Constructing k-NN graph', file=tqdm_logger, total=num_embs):
        nns, sims = np.array(ngt_index.search(embs[i], k + 1))[1:].T
        sims = 1 - sims
        knn_i.extend([i] * k)
        knn_j.extend(nns)
        knn_w.extend(sims)
    knn_i, knn_j, knn_w = np.array(knn_i), np.array(knn_j), np.array(knn_w)

    logger.info('Saving k-NN graph.')
    np.savez(out_pth, i=knn_i, j=knn_j, w=knn_w)

    logger.info('Done.')


if __name__ == '__main__':
    main()
