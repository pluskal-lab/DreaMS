import sys
from tqdm import tqdm
tqdm.pandas()
from annoy import AnnoyIndex
# import msml.utils.io as io  # TODO
import logging
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time
import io as std_io


def load_gems_embs(pth, logger):
    with h5py.File(pth, 'r') as f:
        # Get rid of "blank" samples
        gemc_C_names = pd.Series(f['name'][:]).astype(str)
        logger.info(f'Num. of spectra: {gemc_C_names.shape}')
        # TODO: This is a temporary fix, we need to get rid of blank samples in a better way
        blank_substrs = ['blank', 'wash', 'no_inj', 'noinj', 'empty', 'solvent']
        gemc_C_names = gemc_C_names[gemc_C_names.apply(lambda n: all([s not in n.lower() for s in blank_substrs]))]
        idx_gems = gemc_C_names.index.tolist()
        logger.info(f'Num. of non-blank spectra: {gemc_C_names.shape}')
        # Get embeddings
        gemc_C_embs = f['DreaMS_Embedding'][:][idx_gems]
    return gemc_C_embs


def setup_logger(log_file_path=None, log_name='log'):

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TqdmToLogger(std_io.StringIO):
    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None, mininterval=5):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
        self.mininterval = mininterval
        self.last_time = 0

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        if len(self.buf) > 0 and time.time() - self.last_time > self.mininterval:
            self.logger.log(self.level, self.buf)
            self.last_time = time.time()


def main():
    out_dir = Path('/storage/plzen1/home/romanb/DreaMS_Atlas')
    lib_pth = Path('/auto/brno2/home/romanb/msml/msml/data/merged/datasets/nist20_mona_clean_A_merged_spectra_dreams.pkl')
    gems_dir = Path('/storage/plzen1/home/romanb/msvn_C')
    gems_name = 'msvn_C_H1000_KK1'
    gems_pths = list(gems_dir.glob(f'{gems_name}.*.hdf5'))
    gems_pths = sorted(gems_pths, key=lambda p: int(p.name.split('.')[-2]))  # Sort by chunk ids

    logger = setup_logger(out_dir / 'DreaMS_Atlas_annoy.log')
    tqdm_logger = TqdmToLogger(logger)

    df_lib = pd.read_pickle(lib_pth)
    lib_embs = np.stack(df_lib['DreaMS'].values) 

    annoy = AnnoyIndex(lib_embs.shape[1], metric='angular')
    for i, v in tqdm(enumerate(lib_embs), desc=f'Adding {lib_pth.name} DreaMS to annoy index', total=lib_embs.shape[0], file=tqdm_logger):
        annoy.add_item(i, v)

    i_offset = lib_embs.shape[0]  # Offset for the next chunk of GeMS
    for p in gems_pths:
        logger.info(f'Loading embeddings from {p}.')
        embs = load_gems_embs(p, logger)
        for i, v in tqdm(enumerate(embs), desc=f'Adding {p.name} DreaMS to annoy index', total=embs.shape[0]):
            annoy.add_item(i + i_offset, v)
        i_offset += embs.shape[0]

    logger.info('Building annoy index.')
    annoy.build(10, n_jobs=-1)

    logger.info('Saving annoy index.')
    annoy.save(str(out_dir / 'DreaMS_Atlas_annoy.ann'))


if __name__ == '__main__':
    main()
