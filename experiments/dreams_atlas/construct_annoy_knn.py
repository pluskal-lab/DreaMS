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
import h5py
import numpy as np
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity


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
    d = 1024  # Dimensionality of embeddings. TODO: Can it be inferred from the data?
    k = 5  # k in k-NN
    flush_each_n_samples = 1_000_000  # Define the number of iterations after which to update the .hdf5 file.
    out_dir = Path('/storage/plzen1/home/romanb/DreaMS_Atlas')
    annoy_pth = out_dir / 'DreaMS_Atlas_annoy.ann'
    out_pth = out_dir / f'DreaMS_Atlas_annoy_{k}NN.hdf5'

    logger = setup_logger(out_pth.with_suffix('.log'))
    tqdm_logger = TqdmToLogger(logger)

    # Initialize AnnoyIndex
    a = AnnoyIndex(d, 'angular')
    logger.info(f'Loading AnnoyIndex from {annoy_pth}.')
    a.load(str(annoy_pth), prefault=True)  # Load AnnoyIndex into memory
    logger.info(f'AnnoyIndex loaded. Number of items: {a.get_n_items()}.')

    # Create datasets in HDF5 file
    with h5py.File(out_pth, 'w') as hdf5_file:
        nns_dataset = hdf5_file.create_dataset('nns_idx', (a.get_n_items(), k), dtype='i8')
        sims_dataset = hdf5_file.create_dataset('nns_sims', (a.get_n_items(), k), dtype='f')

        # Iterate through all items
        for i in tqdm(list(range(a.get_n_items())), desc='Computing k-NN graph', file=tqdm_logger):
            nns = a.get_nns_by_item(i, k + 1)
            nns_vecs = np.stack([a.get_item_vector(j) for j in nns])
            sims = cosine_similarity(nns_vecs[0].reshape(1, -1), nns_vecs[1:])[0]

            # Update datasets
            nns_dataset[i] = nns[1:]
            sims_dataset[i] = sims

            # Update HDF5 file every `flush_each_n_samples` iterations
            if (i + 1) % flush_each_n_samples == 0:
                logger.info(f'Flushing after {i + 1} samples.')
                hdf5_file.flush()


if __name__ == '__main__':
    main()
