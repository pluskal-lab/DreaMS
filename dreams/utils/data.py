import os
import pathlib
import numpy as np
import h5py
import pandas as pd
import umap
import torch
from torch import nn
import random
import pickle
import igraph
import networkx as nx
import scipy
# import spectral_entropy
import pynndescent
import plotly.graph_objects as go
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from pandarallel import pandarallel
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from statistics import mean
from matchms import Spectrum
from matchms.exporting import save_as_mgf
from abc import ABC, abstractmethod
from collections.abc import Iterable
from rdkit import Chem
from rdkit.Chem import DataStructs
from tqdm import tqdm
from typing import Optional, List, Union, Tuple
from sklearn import metrics
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine as cos_dist
import dreams.utils.spectra as su
import dreams.utils.mols as mu
import dreams.utils.io as io
import dreams.utils.misc as utils
from dreams.utils.dformats import DataFormat, DataFormatA
from dreams.models.layers.feed_forward import FeedForward
from dreams.models.optimization.losses_metrics import FingerprintMetrics, CosSimLoss
from dreams.models.optimization.samplers import MaxVarBatchSampler
from dreams.definitions import *


class SpectrumPreprocessor:
    """
    A class for preprocessing mass spectrometry spectra.

    This class provides functionality to preprocess mass spectrometry spectra,
    including peak trimming, padding, intensity normalization, and various
    data augmentation techniques.

    Attributes:
        dformat (DataFormat): The data format specification.
        prec_intens (float): The intensity of the precursor peak.
        n_highest_peaks (int): The number of highest intensity peaks to keep.
        spec_entropy_cleaning (bool): Whether to apply spectral entropy cleaning.
        normalize_mzs (bool): Whether to normalize m/z values.
        to_relative_intensities (bool): Whether to convert intensities to relative values.
        precision (int): The precision of the output data (32 or 64 bit).
        mz_shift_aug_p (float): The probability of applying m/z shift augmentation.
        mz_shift_aug_max (float): The maximum m/z shift for augmentation.

    Methods:
        __call__(spec, prec_mz, high_form, augment): Preprocess a single spectrum.
    """

    def __init__(self, dformat: DataFormat, prec_intens=1.1, n_highest_peaks=None, spec_entropy_cleaning=False,
                 normalize_mzs=False, precision=32, mz_shift_aug_p=0, mz_shift_aug_max=0, to_relative_intensities=True):
        """
        Initialize the SpectrumPreprocessor.

        Args:
            dformat (DataFormat): The data format specification.
            prec_intens (float): The intensity of the precursor peak.
            n_highest_peaks (int): The number of highest intensity peaks to keep.
            spec_entropy_cleaning (bool): Whether to apply spectral entropy cleaning.
            normalize_mzs (bool): Whether to normalize m/z values.
            precision (int): The precision of the output data (32 or 64 bit).
            mz_shift_aug_p (float): The probability of applying m/z shift augmentation.
            mz_shift_aug_max (float): The maximum m/z shift for augmentation.
            to_relative_intensities (bool): Whether to convert intensities to relative values.
        """
        assert precision in {32, 64}

        self.dformat = dformat
        self.prec_intens = prec_intens
        self.n_highest_peaks = n_highest_peaks
        self.spec_entropy_cleaning = spec_entropy_cleaning
        self.normalize_mzs = normalize_mzs
        self.to_relative_intensities = to_relative_intensities
        self.precision = precision
        self.mz_shift_aug_p = mz_shift_aug_p
        self.mz_shift_aug_max = mz_shift_aug_max

        if self.n_highest_peaks is None:
            self.n_highest_peaks = self.dformat.max_peaks_n

    def __call__(self, spec: np.array, prec_mz=None, high_form='auto', augment=False):
        """
        Preprocess a single spectrum.

        Args:
            spec (np.array): The input spectrum.
            prec_mz (float): The precursor m/z value.
            high_form (bool): Whether the input is in high form (n_peaks, 2) or not.
            augment (bool): Whether to apply data augmentation.

        Returns:
            np.array: The preprocessed spectrum.
        """
        spec = spec.copy()

        if high_form == 'auto':
            if spec.shape[0] == 2 and spec.shape[1] == 2:
                raise ValueError(f'high_form = {"auto"} is ambiguous for a peak list with two peaks of shape {spec.shape}')
            else:
                high_form = spec.shape[1] == 2

        # (2, n_peaks) -> (n_peaks, 2)
        if not high_form:
            spec = spec.T

        if spec.shape[1] != 2:
            raise ValueError(f'Spectrum is not in the shape (n_peaks, 2) but has shape {spec.shape}')

        # Clean spectrum as in spectral entropy paper
        if self.spec_entropy_cleaning:
            raise NotImplementedError("Spectral entropy was removed as a dependency from the project.")
            spec = spectral_entropy.tools.clean_spectrum(spec)

        # Trim and pad peak list
        if self.n_highest_peaks:
            spec = su.trim_peak_list(spec.T, self.n_highest_peaks).T
            spec = su.pad_peak_list(spec.T, target_len=self.n_highest_peaks).T
        else:
            raise ValueError('It should never happen, since the class is designed for torch dataloaders requiring same'
                             'num. of peaks within batches.')

        # Normalize intensities to be relative to base peak
        if self.to_relative_intensities:
            spec = su.to_rel_intensity(spec.T).T

        # Prepend precursor peak
        if prec_mz is not None:
            spec = su.prepend_precursor_peak(spec, prec_mz, self.prec_intens, high=True)

        # Augment all m/z values by adding to them a constant random value
        if augment and self.mz_shift_aug_p > 0:
            if random.random() < self.mz_shift_aug_p:
                spec[:, 0] = spec[:, 0] + random.random() * self.mz_shift_aug_max

        # Normalize
        if self.normalize_mzs:
            spec = su.normalize_mzs(spec, self.dformat.max_prec_mz, high=True)

        # Adjust precision
        if self.precision == 32:
            spec = spec.astype(np.float32, copy=False)

        return spec


def load_hdf5_in_mem(dct):
    if isinstance(dct, h5py.Dataset):
        return dct[()]
    ret = {}
    for k, v in dct.items():
        ret[k] = load_hdf5_in_mem(v)
    return ret


class MSData:
    def __init__(
            self,
            hdf5_pth: Union[Path, str, List[Path]],
            in_mem: bool = False,
            mode: str = 'r',
            spec_col: str = SPECTRUM,
            prec_mz_col: str = PRECURSOR_MZ,
            index_col: Optional[str] = None
        ):
        if isinstance(hdf5_pth, list):
            self.f = io.ChunkedHDF5File(hdf5_pth)
            self.hdf5_pth = hdf5_pth[0]
        else:
            self.hdf5_pth = Path(hdf5_pth)
            self.f = h5py.File(hdf5_pth, mode)

        for k in [spec_col, prec_mz_col]:
            if k not in self.f.keys():
                raise ValueError(f'Column "{k}" is not present in the dataset {hdf5_pth}.')

        if self.f[spec_col].shape[1] != 2 or len(self.f[spec_col].shape) != 3:
            raise ValueError('Shape of spectra has to be (num_spectra, 2 (m/z, intensity), num_peaks).')

        num_spectra = set()
        for k in self.f.keys():
            num_spectra.add(self.f[k].shape[0])
        if len(num_spectra) != 1:
            raise ValueError(f'Columns in {hdf5_pth} have different number of entries.')

        self.in_mem = in_mem
        self.num_spectra = num_spectra.pop()
        self.data = self.f
        self.mode = mode

        if in_mem:
            print(f'Loading dataset {self.hdf5_pth.stem} into memory ({self.num_spectra} spectra)...')
            self.data = self.load_hdf5_in_mem(self.f)
        
        self.index_col = index_col
        self.index_to_i = None
        if index_col is not None:
            self.use_col_as_index(index_col)

    def __del__(self):
        if hasattr(self, 'f') and isinstance(self.f, h5py.File):
            self.f.close()

    def columns(self):
        return list(self.data.keys())

    def load_col_in_mem(self, col):
        if isinstance(col, h5py.Group):
            return self.load_hdf5_in_mem(col)
        else:
            col = col[:]
            # if col.dtype == object:
            #     col = np.char.decode(col.astype(bytes), 'utf-8', errors='ignore')
            return col

    def load_hdf5_in_mem(self, group):
        data = {}
        for key, item in group.items():
            data[key] = self.load_col_in_mem(item)
        return data

    @staticmethod
    def from_hdf5(pth: Path, **kwargs):
        return MSData(pth, **kwargs)

    @staticmethod
    def from_hdf5_chunks(pths: List[Path], **kwargs):
        return MSData(pths, **kwargs)

    @staticmethod
    def from_pandas(
        df: Union[Path, str, pd.DataFrame],
        n_highest_peaks=128,
        spec_col=SPECTRUM,
        prec_mz_col=PRECURSOR_MZ,
        adduct_col=ADDUCT,
        charge_col=CHARGE,
        mol_col=SMILES,
        ignore_cols=(),
        in_mem=True,
        hdf5_pth=None,
        compression_opts=0,
        mode='r'
    ):

        # Load dataframe
        if isinstance(df, str):
            df = Path(df)
        if isinstance(df, Path):
            hdf5_pth = df.with_suffix('.hdf5')
            df = pd.read_pickle(df)
        else:
            if hdf5_pth is None:
                raise ValueError('`hdf5_pth` has to be specified if `df` is not a Path.')

        # Check for empty dataset
        if not len(df):
            raise ValueError(f'Input dataset is empty (most likely doesn\'t contain MS2 spectra).')

        # Validate num. of peaks
        if n_highest_peaks is None:
            raise NotImplementedError('Not implemented yet. With this option, `n_highest_peaks` has to be set to max peaks in the dataset.')
        elif n_highest_peaks < 1:
            raise ValueError('`n_highest_peaks` has to be > 0.')

        for col in [spec_col, prec_mz_col]:#, adduct_col, charge_col, smiles_col]:
            if col not in df.columns:
                raise ValueError(f'Column "{col}" is not present in the dataframe. Available columns: {df.columns}.')

        # Convert dataframe columns to .hdf5 datasets
        with h5py.File(hdf5_pth, 'w') as f:
            for k, v in df.items():
                if k in ignore_cols:
                    continue

                if k == spec_col:
                    k = SPECTRUM
                    pls = []
                    for p in v:
                        p = su.trim_peak_list(p, n_highest_peaks)
                        p = su.pad_peak_list(p, n_highest_peaks)
                        pls.append(p)
                    v = np.stack(pls)
                else:
                    if k == prec_mz_col:
                        k = PRECURSOR_MZ
                    elif k == adduct_col:
                        k = ADDUCT
                    elif k == charge_col:
                        k = CHARGE
                    elif k == mol_col:
                        k = SMILES
                        if isinstance(v[0], str) and v[0].startswith('InChI='):
                            v = [Chem.MolFromInchi(i) for i in v]
                        if isinstance(v[0], Chem.rdchem.Mol):
                            v = pd.Series([Chem.MolToSmiles(m) for m in v])

                    if v.dtype == object:
                        v = v.astype(str)
                    v = v.values

                f.create_dataset(k, data=v, compression='gzip', compression_opts=compression_opts)
        return MSData(hdf5_pth, in_mem=in_mem, mode=mode)

    @staticmethod
    def from_mzml(
        pth: Union[Path, str],
        scan_range: Optional[Tuple[int, int]] = None,
        verbose_parser: bool = False,
        **kwargs
    ):
        # TODO: use mzml reader from process_ms_file.py, move it here
        # TODO: refactor trimming and padding, no hard-coded 128

        pth = Path(pth)
        df = io.read_mzml(pth, scan_range=scan_range, verbose=verbose_parser)
        hdf5_pth = pth.with_suffix('.hdf5')
        if scan_range:
            hdf5_pth = io.append_to_stem(pth, f'scans_{scan_range[0]}-{scan_range[1]}')
        return MSData.from_pandas(df, hdf5_pth=hdf5_pth, **kwargs)

    @staticmethod
    def from_msp(pth: Union[Path, str], in_mem=True, **kwargs):
        raise NotImplementedError('Not tested but should work.')
        pth = Path(pth)
        df = io.read_msp(pth)
        MSData.from_pandas(df, in_mem=in_mem, df_pth=pth)
        return MSData(pth.with_suffix('.hdf5'), in_mem=in_mem)

    @staticmethod
    def from_mgf(pth: Union[Path, str], in_mem=True, **kwargs):
        pth = Path(pth)
        df = io.read_mgf(pth)
        return MSData.from_pandas(df, hdf5_pth=pth.with_suffix('.hdf5'), in_mem=in_mem, **kwargs)

    @staticmethod
    def from_pickle(pth: Union[Path, str], in_mem=True, **kwargs):
        pth = Path(pth)
        df = pd.read_pickle(pth)
        return MSData.from_pandas(df, hdf5_pth=pth.with_suffix('.hdf5'), in_mem=in_mem, **kwargs)

    @staticmethod
    def load(pth: Union[Path, str], in_mem=False, **kwargs):
        pth = Path(pth)
        if pth.suffix.lower() == '.hdf5':
            return MSData.from_hdf5(pth, in_mem=in_mem, **kwargs)
        elif pth.suffix.lower() == '.mzml':
            return MSData.from_mzml(pth, in_mem=in_mem, **kwargs)
        elif pth.suffix.lower() == '.msp':
            return MSData.from_msp(pth, in_mem=in_mem, **kwargs)
        elif pth.suffix.lower() == '.mgf':
            return MSData.from_mgf(pth, in_mem=in_mem, **kwargs)
        elif pth.suffix.lower() == '.pkl':
            return MSData.from_pandas(pth, in_mem=in_mem, **kwargs)
        else:
            raise NotImplementedError(f'Loading from {pth.suffix} is not implemented.')

    def to_torch_dataset(self, spec_preproc: SpectrumPreprocessor, label=None, **kwargs):
        if label is not None:
            return LabeledSpectraDataset(msdata=self, label=label, spec_preproc=spec_preproc, **kwargs)
        else:
            return RawSpectraDataset(self.get_spectra(), self.get_prec_mzs(), spec_preproc, **kwargs)

    def to_pandas(self, unpad=True, ignore_cols=(DREAMS_EMBEDDING,)):
        df = {col: self.get_values(col) for col in self.columns() if col not in ignore_cols}

        if SPECTRUM not in ignore_cols:
            df[SPECTRUM] = list(self.get_spectra())
            if unpad:
                df[SPECTRUM] = [su.unpad_peak_list(s) for s in df[SPECTRUM]]
            
            df[SPECTRUM] = [s.tolist() for s in df[SPECTRUM]]

        return pd.DataFrame(df)
    
    def to_mgf(self, out_pth: Union[Path, str]):
        out_pth = Path(out_pth)
        if not out_pth.exists():
            out_pth.parent.mkdir(parents=True, exist_ok=True)

        spectra = []
        df = self.to_pandas()
        for _, row in tqdm(df.iterrows(), total=len(df), desc='Converting to matchms before saving to mgf'):
            spec = Spectrum(
                mz=row[SPECTRUM][0],
                intensities=row[SPECTRUM][1],
                metadata={k: v for k, v in row.items() if k != SPECTRUM and v is not None and not pd.isna(v)}
            )
            spectra.append(spec)
        save_as_mgf(spectra, str(out_pth))

    def get_values(self, col, idx=None, decode_strings=True):
        col = self.data[col]
        col = col[idx] if idx is not None else col[:]
        if decode_strings and isinstance(col, bytes):
            col = col.decode('utf-8')
        elif decode_strings and col.dtype == object:
            col = [s.decode('utf-8') if isinstance(s, bytes) else s for s in col]
        return col

    def __len__(self):
        return self.num_spectra

    def __getitem__(self, col):
        return self.get_values(col)

    def at(self, i, plot_mol=True, plot_spec=True, return_spec=False, vals=None, unpad_spec=True):

        if self.index_col is not None:
            if i not in self.index_to_i:
                raise ValueError(f'Index {i} is not present in the index.')
            i = self.index_to_i[i]
        else:
            if not isinstance(i, int):
                raise ValueError(f'Index {i} is not an integer.')
            if i >= self.num_spectra:
                raise ValueError(f'Index {i} is out of bounds for the dataset with {self.num_spectra} spectra.')

        if plot_spec:
            su.plot_spectrum(self.data[SPECTRUM][i], prec_mz=self.data[PRECURSOR_MZ][i])
        if plot_mol and SMILES in self.columns():
            display(Chem.MolFromSmiles(self.data[SMILES][i]))
        def process_val(v):
            if isinstance(v, bytes):
                return v.decode('utf-8')
            return v
        res = {k: process_val(self.data[k][i]) for k in self.columns() if (vals is None or k in vals)}

        if SPECTRUM in res:
            if not return_spec:
                del res[SPECTRUM]
            elif unpad_spec:
                res[SPECTRUM] = su.unpad_peak_list(res[SPECTRUM])

        return res

    def use_col_as_index(self, col):
        if col not in self.columns():
            raise ValueError(f'Column "{col}" is not present in the dataset.')
        self.index_to_i = {i: idx for idx, i in enumerate(self.get_values(col))}

    def get_spectra(self, idx=None):
        return self.get_values(SPECTRUM, idx)

    def get_prec_mzs(self, idx=None):
        return self.get_values(PRECURSOR_MZ, idx)

    def get_adducts(self, idx=None):
        return self.get_values(ADDUCT, idx)
    
    def get_charges(self, idx=None):
        return self.get_values(CHARGE, idx)
    
    def get_smiles(self, idx=None):
        return self.get_values(SMILES, idx)

    def remove_column(self, name):
        del self.f[name]

    def add_column(self, name, data, remove_old_if_exists=False):
        if self.mode != 'a':
            raise ValueError('Adding new columns is allowed only in append mode. Initialize msdata as '
                             '`MSData(..., mode="a")` to add new columns.')
        if remove_old_if_exists and name in self.columns():
            self.remove_column(name)
        self.f.create_dataset(name, data=data)

    def rename_column(self, old_name, new_name, remove_old_if_exists=False):
        if remove_old_if_exists and new_name in self.columns():
            print(f'Removing column "{new_name}"...')
            self.remove_column(new_name)
        self.f[new_name] = self.f[old_name]
        del self.f[old_name]
        if self.in_mem:
            self.data[new_name] = self.data.pop(old_name)

    def extend_column(self, name, data):
        if data.shape[1:] != self.f[name].shape[1:]:
            raise ValueError(f'Shape of the data ({data.shape}) in dimensions > 0 does not match the shape of the '
                             f'column "{name}" ({self.f[name].shape}).')
        self.f[name].resize((self.f[name].shape[0] + data.shape[0], *data.shape[1:]))
        self.f[name][-data.shape[0]:] = data

    def form_subset(self, idx, out_pth, verbose=False):
        with h5py.File(out_pth, 'w') as f:
            for k in self.columns():
                if verbose:
                    print(f'Creating dataset "{k}"...')
                f.create_dataset(k, data=self.get_values(k, decode_strings=False)[:][idx], dtype=self.f[k].dtype)
        return MSData(out_pth)

    def spec_to_matchms(self, i: int) -> Spectrum:
        spec = su.unpad_peak_list(self.get_spectra(i))
        metadata = self.at(i, plot_mol=False, plot_spec=False, return_spec=False)
        return Spectrum(
            mz=spec[0],
            intensities=spec[1],
            metadata=metadata
        )

    def to_matchms(self, progress_bar=True) -> List[Spectrum]:
        return [
            self.spec_to_matchms(i)
            for i in tqdm(range(len(self)), desc='Converting to matchms', disable=not progress_bar)
        ]

    def to_pynndescent(
            self,
            embedding_col=DREAMS_EMBEDDING,
            out_pth=None,
            store_index=False,
            n_neighbors=50,
            metric='cosine',
            verbose=True
        ):
        if embedding_col not in self.columns():
            raise ValueError(f'Column "{embedding_col}" is not present in the dataset.')

        index = pynndescent.NNDescent(
            self.get_values(embedding_col),
            n_neighbors=n_neighbors,
            metric=metric,
            verbose=verbose
        )

        if store_index:
            if out_pth is None:
                out_pth = self.hdf5_pth.with_suffix(f'.pynndescent{n_neighbors}.pkl')

            with open(out_pth, "wb") as f:
                pickle.dump(index, f)

        return index

    @staticmethod
    def merge(
        pths: List[Path],
        out_pth: Path,
        cols=MSDATA_COLUMNS.copy(),
        show_tqdm=True,
        logger=None,
        add_dataset_col=True,
        in_mem=False,
        spectra_already_trimmed=False,
        filter_idx=None
    ):
        # TODO: spectra_already_trimmed can be determined from the data

        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        logger.info(f'Using HDF5_USE_FILE_LOCKING={os.environ["HDF5_USE_FILE_LOCKING"]}.')

        for pth in pths:
            if not pth.exists():
                raise ValueError(f'File {pth} does not exist.')
            if pth.suffix.lower() != '.hdf5':
                raise ValueError(f'File {pth} is not a .hdf5 file.')

        if not cols:
            with h5py.File(pths[0], 'r') as f:
                cols = list(f.keys())
                logger.info(f'Columns not specified, using all columns from the first dataset: {cols}.')

        for col in cols:
            if not all(col in h5py.File(pth, 'r').keys() for pth in pths):
                raise ValueError(f'Column "{col}" is not present in all of the datasets.')
        
        if add_dataset_col and DATASET not in cols:
            cols.append(DATASET)

        if not spectra_already_trimmed:
            n_highest_peaks = max(h5py.File(pth, 'r')[SPECTRUM].shape[-1] for pth in pths)
            if logger:
                logger.info(f'Truncation spectra to {n_highest_peaks} highest peaks (max num. peaks among pths datasets).')

        with h5py.File(out_pth, 'w') as f_out:
            for i, pth in enumerate(tqdm(pths, desc='Merging hdfs')) if show_tqdm else enumerate(pths):
                with h5py.File(pth, 'r') as f:

                    f_len = f[list(f.keys())[0]].shape[0]
                    if logger:
                        logger.info(f'Appending {i+1}th / {len(pths)} hdf5 ({f_len} samples) file to {out_pth}.')

                    if filter_idx is not None:
                        idx = filter_idx(f)

                    for k in cols:

                        # New columns with constant file names
                        if k == DATASET:
                            data = np.array([pth.stem] * f_len, dtype='S')

                        # Mzs and intensities to spectra
                        elif k == SPECTRUM and not spectra_already_trimmed:
                            data = f[k][:]
                            spectra = su.trim_peak_list(data, n_highest=n_highest_peaks)
                            spectra = su.pad_peak_list(data, target_len=n_highest_peaks)
                            data = spectra

                        # Other metadata datasets
                        else:
                            data = f[k][:]

                        if filter_idx is not None:
                            data = data[idx]

                        if i == 0:
                            f_out.create_dataset(
                                k, data=data, shape=data.shape, maxshape=(None, *data.shape[1:]), dtype=data.dtype
                            )
                        else:
                            f_out[k].resize(f_out[k].shape[0] + data.shape[0], axis=0)
                            f_out[k][-data.shape[0]:] = data
        
        return MSData(out_pth, in_mem=in_mem)
    
    def __repr__(self) -> str:
        return f'MSData(pth={self.hdf5_pth}, in_mem={self.in_mem}) with {len(self):,} spectra.'


def subset_lsh(
        in_pth: Union[Path, str],
        out_pth: Optional[Path] = None,
        lsh_col: str = 'lsh',
        max_specs_per_lsh: int = 1,
        random_seed: int = 42
    ):
    if not isinstance(in_pth, Path):
        in_pth = Path(in_pth)
    if out_pth is None:
        out_pth = io.append_to_stem(in_pth, f'{max_specs_per_lsh}')
    print(f'Subsetting LSHs from {in_pth} to {out_pth}...')
    
    print(f'Loading dataset...')
    msdata = MSData(in_pth, in_mem=True)
    lshs = msdata.get_values(lsh_col)

    if max_specs_per_lsh == 1:
        # For max_specs_per_lsh=1, we can just take first occurrence of each unique LSH (fast)
        print(f'Subsetting LSHs...')
        _, idx = np.unique(lshs, return_index=True)
        filtered_idx = np.sort(idx)
    else:
        # For max_specs_per_lsh > 1, we need to subset each LSH to max_specs_per_lsh (slow)
        lshs_unique, lshs_counts = np.unique(lshs, return_counts=True)
        filtered_idx = []
        non_filtered_lshs = []
        np.random.seed(random_seed)
        for lsh, count in tqdm(zip(lshs_unique, lshs_counts), desc='Subsetting LSHs', total=len(lshs_unique)):
            if count > max_specs_per_lsh:
                idx = np.where(lshs == lsh)[0]
                filtered_idx.extend(np.random.choice(idx, max_specs_per_lsh, replace=False))
            else:
                non_filtered_lshs.append(lsh)
        filtered_idx.extend(np.where(np.isin(lshs, non_filtered_lshs))[0])
        filtered_idx = np.sort(np.array(filtered_idx))

    print(f'Saving LSH subset with {len(filtered_idx):,} spectra...')
    msdata.form_subset(filtered_idx, out_pth)
    return MSData(out_pth, in_mem=True)


class MaskedSpectraDataset(Dataset):
    """
    A dataset class for masked spectra used in self-supervised learning tasks.

    This class prepares spectra data for various self-supervised learning objectives
    such as peak masking, m/z masking, and intensity masking.

    Attributes:
        data (dict): The dataset containing spectra and related information.
        dformat (DataFormat): The data format specification.
        ssl_objective (str): The self-supervised learning objective.
        spec_preproc (SpectrumPreprocessor): The spectrum preprocessor.
        frac_masks (float): The fraction of peaks to mask.
        min_n_masks (int): The minimum number of peaks to mask.
        n_samples (int): The number of samples to use from the dataset.
        mask_val (float): The value to use for masked peaks.
        min_mask_intens (float): The minimum intensity for peaks to be considered for masking.
        mask_prec (bool): Whether to mask the precursor peak.
        deterministic_mask (bool): Whether to use deterministic masking.
        mask_peaks (bool): Whether to mask peaks.
        mask_intens_strategy (str): The strategy for masking intensities.
        ret_order_pairs (bool): Whether to return retention order pairs.
        return_charge (bool): Whether to return charge information.
        acc_est_weight (bool): Whether to use accuracy estimation weighting.
        lsh_weight (bool): Whether to use LSH weighting.
        bert801010_masking (bool): Whether to use BERT-style 80-10-10 masking.

    Methods:
        __len__(): Return the length of the dataset.
        __getitem__(i): Get a single item from the dataset.
        get_spec(i): Get a preprocessed spectrum.
    """

    def __init__(self, in_pth: Path, dformat: DataFormat, ssl_objective: str, spec_preproc: SpectrumPreprocessor,
                 mask_peaks=True, mask_intens_strategy='intens_p', frac_masks=0.2, min_n_masks=2, mask_val=-1.,
                 min_mask_intens=0.1, mask_prec=False, n_samples=None, logger=None, deterministic_mask=True,
                 ret_order_pairs=False, return_charge=False, acc_est_weight=False, lsh_weight=False,
                 bert801010_masking=False):
        """
        Initialize the MaskedSpectraDataset.

        Args:
            in_pth (Path): The input path for the dataset.
            dformat (DataFormat): The data format specification.
            ssl_objective (str): The self-supervised learning objective.
            spec_preproc (SpectrumPreprocessor): The spectrum preprocessor.
            mask_peaks (bool): Whether to mask peaks.
            mask_intens_strategy (str): The strategy for masking intensities.
            frac_masks (float): The fraction of peaks to mask.
            min_n_masks (int): The minimum number of peaks to mask.
            mask_val (float): The value to use for masked peaks.
            min_mask_intens (float): The minimum intensity for peaks to be considered for masking.
            mask_prec (bool): Whether to mask the precursor peak.
            n_samples (int): The number of samples to use from the dataset.
            logger (Logger): The logger object.
            deterministic_mask (bool): Whether to use deterministic masking.
            ret_order_pairs (bool): Whether to return retention order pairs.
            return_charge (bool): Whether to return charge information.
            acc_est_weight (bool): Whether to use accuracy estimation weighting.
            lsh_weight (bool): Whether to use LSH weighting.
            bert801010_masking (bool): Whether to use BERT-style 80-10-10 masking.
        """
        assert ssl_objective in {'mask_peak', 'mask_mz', 'mask_intensity', 'mask_mz_hot', 'mask_peak_hot', 'shuffling'}
        assert mask_peaks or mask_prec
        assert not mask_prec or spec_preproc.prec_intens
        assert not acc_est_weight or not lsh_weight, 'Weighintg by both instrument accuracies and LSHs is not implemented.'

        self.data = {}
        self.dformat = dformat
        self.ssl_objective = ssl_objective
        self.spec_preproc = spec_preproc
        self.frac_masks = frac_masks
        self.min_n_masks = min_n_masks
        self.n_samples = n_samples
        self.mask_val = mask_val
        self.min_mask_intens = min_mask_intens
        self.mask_prec = mask_prec
        self.deterministic_mask = deterministic_mask
        self.mask_peaks = mask_peaks
        self.mask_intens_strategy = mask_intens_strategy
        self.ret_order_pairs = ret_order_pairs
        self.return_charge = return_charge
        self.acc_est_weight = acc_est_weight
        self.lsh_weight = lsh_weight
        self.bert801010_masking = bert801010_masking

        # Load dataset features
        features = [SPECTRUM, PRECURSOR_MZ]
        if self.return_charge:
            features.append(CHARGE)
        if self.ret_order_pairs:
            features += [RT, NAME]
        if self.acc_est_weight:
            features.append('instrument accuracy est.')
        if self.lsh_weight:
            features.append('lsh')

        if in_pth.suffix == '.hdf5':
            with h5py.File(in_pth, 'r') as f:
                for k in features:
                    assert k in f.keys(), f'Features "{k}" are not present in dataset {in_pth}.'
                    if logger:
                        logger.info(f'Loading .hdf5 dataset "{k}" into memory...')
                    self.data[k] = f[k][:n_samples] if n_samples is not None else f[k][:]
        elif in_pth.suffix == '.pkl':
            logger.info('Loading .pkl dataset...')
            if len(features) > 2:
                raise ValueError(f'.pkl datasets currently do not support features '
                                 f'{[f for f in features if f not in ["spectra", "precursor mz"]]}.')
            df = pd.read_pickle(in_pth)  # TODO: use MSData
            self.data[SPECTRUM] = np.stack(df['PARSED PEAKS']).transpose((0, 2, 1))
            self.data[PRECURSOR_MZ] = np.stack(df['PRECURSOR M/Z'])
        else:
            raise ValueError(f'Not supported input format {in_pth.suffix} of the data file {in_pth}.')

        # Construct the mapping from file names to corresponding spectra to sample retention order pairs
        if self.ret_order_pairs:
            logger.info('Constructing file name to spectra indices mapping for sampling retention order pairs...')
            
            # Convert names to integers to speed up the process
            unique_names = np.unique(self.data[NAME])
            name_to_int = {name: i for i, name in enumerate(unique_names)}
            int_enc_names = np.array([name_to_int[name] for name in self.data[NAME]])

            # Construct the mapping from file names to corresponding spectrum indices
            self.name_idx = {}
            for name in tqdm(
                unique_names,
                desc='Constructing file name to spectra indices mapping for sampling retention order pairs...'
            ):
                self.name_idx[name] = np.where(int_enc_names == name_to_int[name])[0]

        if self.lsh_weight:
            logger.info('Constructing the Counter for LSHs...')
            self.lsh_weights = Counter(self.data['lsh'])

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        if self.n_samples:
            return self.n_samples
        return self.data[SPECTRUM].shape[0]

    def get_spec(self, i):
        """
        Get a preprocessed spectrum.

        Args:
            i (int): The index of the spectrum to retrieve.

        Returns:
            dict: A dictionary containing the preprocessed spectrum and related information.
        """
        # Get peak list
        spectrum = self.data[SPECTRUM][i]
        prec_mz = self.data[PRECURSOR_MZ][i]
        spectrum = self.spec_preproc(spectrum, prec_mz=prec_mz, augment=True, high_form=False)

        # Make masking deterministic within spectrum
        if self.deterministic_mask:
            np.random.seed(round(prec_mz))

        if self.mask_prec and not self.mask_peaks:
            raise NotImplementedError
        else:
            # Initialize mask for all but non-padding tokens
            mask = spectrum[:, 1] > 0

            # Do not mask precursor peak
            if not self.mask_prec:
                mask &= spectrum[:, 1] < 1  # < 1 instead of self.prec_intens since there are often "two" prec peaks

            # Choose only "high" peaks for masking
            if self.mask_intens_strategy == 'intens_cutoff':
                mask &= spectrum[:, 1] >= self.min_mask_intens

            n_peaks = mask.sum()
            n_masks = max(self.min_n_masks, round(n_peaks * self.frac_masks))
            if n_peaks > n_masks:
                idx = np.where(mask)[0]
                # Sample masking peaks proportionally to their intensities
                sampling_p = spectrum[idx, 1] / spectrum[idx, 1].sum() if self.mask_intens_strategy == 'intens_p' else None

                mask_i = np.random.choice(idx, size=n_masks, p=sampling_p, replace=False)

                mask[:] = False
                mask[mask_i] = True

        spectrum_mask = spectrum.copy()
        mask_dims = []
        if 'mask_mz' in self.ssl_objective or 'mask_peak' in self.ssl_objective:
            mask_dims.append(0)
        if 'mask_intensity' in self.ssl_objective or 'mask_peak' in self.ssl_objective:
            mask_dims.append(1)

        for d in mask_dims:
            spectrum_mask[mask, d] = self.mask_val

        if self.bert801010_masking:
            p801010 = np.random.random(mask.shape) * mask
            same10_mask = p801010 > 0.9
            rand10_mask = (p801010 > 0.8) & (p801010 <= 0.9)
            for d in mask_dims:

                # Keep same value with 10% probability
                spectrum_mask[same10_mask, d] = spectrum[same10_mask, d]

                # Random m/z, intensity with 10% probability
                spectrum_mask[rand10_mask, d] = np.random.rand()
                if d == 0:
                    spectrum_mask[rand10_mask, d] *= self.dformat.max_mz

        if 'shuffling' in self.ssl_objective:
            # TODO: modify to work with `mask` as return value instead of `mask_i`
            raise NotImplementedError('Not tested after model updates.')

            # Shuffle spectrum intensities with 50% probability
            # if np.random.random() > 0.5:
            #
            #     # Here mask_i "high" peaks are not being "masked" but ensure the shuffling of self.n_masks "high" peaks
            #     # Reserve indices of chosen "high" peaks and zero padding peaks
            #     idx_reserved = np.union1d(mask_i, np.nonzero(spectrum[:, 1] == 0))
            #
            #     # Select the complement to the reserved indices and to the base peak index
            #     idx_to_swap = np.arange(1, spectrum.shape[0])
            #     idx_to_swap = np.setdiff1d(idx_to_swap, np.intersect1d(idx_to_swap, idx_reserved))
            #
            #     # Sample at most n_masks complement indices peaks
            #     if idx_to_swap.size > self.n_masks:
            #         idx_to_swap = np.random.choice(idx_to_swap, size=self.n_masks, replace=False)
            #
            #     # Shuffle chosen "high" peaks and complement indices peaks all together
            #     idx_to_swap = np.union1d(idx_to_swap, mask_i)
            #     for i, e in enumerate(utils.complete_permutation(idx_to_swap)):
            #         spectrum_mask[idx_to_swap[i], 1] = spectrum[e, 1]
            #
            #     # Abuse the "spec_mask" and "spec_real" naming for spectrum and shuffled/non-shuffle label
            #     spectrum = 0.
            # else:
            #     spectrum = 1.

        item = {
            'spec_real': spectrum,
            'spec_mask': spectrum_mask,
            'mask': mask,
        }

        if self.return_charge:
            item[CHARGE] = self.data[CHARGE][i] / self.dformat.max_charge

        if self.acc_est_weight:
            acc_weight = self.data['instrument accuracy est.'][i]
            item['spec_weight'] = 1 / (acc_weight if acc_weight > 0 else 0.1)
        elif self.lsh_weight:
            item['spec_weight'] = 1 / self.lsh_weights[self.data['lsh'][i]]

        return item

    def __getitem__(self, i):

        # Return the pair of masked spectra and binary retention order label
        if self.ret_order_pairs:

            # Get 1st spectrum
            spec1 = self.get_spec(i)

            # Sample 2nd spectrum from the same file
            same_file_idx = self.name_idx[self.data[NAME][i]]
            if self.deterministic_mask:
                np.random.seed(round(spec1['spec_real'][0, 0]))
            i2 = np.random.choice(same_file_idx)
            spec2 = self.get_spec(i2)

            # Prepare data dictionaries along with the retention order label
            item = {f'{k}_1': v for k, v in spec1.items()} | {f'{k}_2': v for k, v in spec2.items()}
            item['ro_label'] = float(self.data[RT][i] < self.data[RT][i2])
            if self.spec_preproc.precision == 32:
                item['ro_label'] = np.float32(item['ro_label'])

        # Return a single masked spectrum
        else:
            item = self.get_spec(i)

        return item


class AnnotatedSpectraDataset(Dataset):
    """
    NOTE: This class is deprecated in favor of `LabeledSpectraDataset`.
    """
    def __init__(self, spectra: List[su.MSnSpectrum], label: str, spec_preproc: SpectrumPreprocessor,
                 dformat: DataFormat, return_smiles=False):
        self.spectra = spectra
        self.label = label
        self.spec_preproc = spec_preproc
        self.dformat = dformat
        self.return_smiles = return_smiles
        if self.label == 'mol_props':
            self.prop_calc = mu.MolPropertyCalculator()

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, i):
        spectrum = self.spectra[i].get_peak_list()
        spectrum = self.spec_preproc(spectrum, prec_mz=self.spectra[i].get_precursor_mz(), high_form=False)

        if self.label.startswith('num'):  # e.g. num_C
            label = float(self.spectra[i].get_precursor_formula(to_dict=True)[self.label.split('_')[1]])
        elif self.label.startswith('has'):  # e.g. has_C
            label = float(bool(self.spectra[i].get_precursor_formula(to_dict=True)[self.label.split('_')[1]]))
        elif self.label.startswith('fp'):  # e.g. fp_morgan_2048
            label = mu.fp_func_from_str(self.label)(self.spectra[i].get_precursor_mol())
        elif self.label == 'qed':
            label = float(Chem.QED.qed(self.spectra[i].get_precursor_mol()))
        elif self.label == 'mol_props':
            label = self.prop_calc.mol_to_props(self.spectra[i].get_precursor_mol(), min_max_norm=True)
        else:
            raise ValueError(f'Invalid label name "{self.label}".')

        item = {
            'spec': spectrum,
            'precursor mz': self.spectra[i].get_precursor_mz(),
            'charge': self.spectra[i].get_precursor_charge() / self.dformat.max_charge,
            'label': label,
        }

        if self.return_smiles:
            item['smiles'] = Chem.MolToSmiles(self.spectra[i].get_precursor_mol(), isomericSmiles=False, canonical=True)

        return item
    

class LabeledSpectraDataset(Dataset):
    def __init__(self, msdata: Union[Path, str, MSData], label: str, spec_preproc: SpectrumPreprocessor,
                 dformat: DataFormat, return_smiles=False):
        self.msdata = msdata if isinstance(msdata, MSData) else MSData(Path(msdata), in_mem=True)
        self.label = label
        self.spec_preproc = spec_preproc
        self.dformat = dformat
        self.return_smiles = return_smiles
        if self.label == 'mol_props':
            self.prop_calc = mu.MolPropertyCalculator()

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, i):
        spectrum = self.msdata.get_spectra(i)
        prec_mz = self.msdata.get_prec_mzs(i)
        spectrum = self.spec_preproc(spectrum, prec_mz=prec_mz, high_form=False)

        if self.label.startswith('num') or self.label.startswith('has'): # e.g. num_C or has_C
            elem = self.label.split('_')[1]
            formula = self.msdata.get_formulas(i, to_dict=True)
            if elem not in formula:
                label = 0.0
            else:
                label = float(formula[elem])
        else:
            mol = Chem.MolFromSmiles(self.msdata.get_smiles(i))
            if self.label.startswith('fp'):  # e.g. fp_morgan_2048
                label = mu.fp_func_from_str(self.label)(mol)
            elif self.label == 'qed':
                label = float(Chem.QED.qed(mol))
            elif self.label == 'mol_props':
                label = self.prop_calc.mol_to_props(mol, min_max_norm=True)
            else:
                raise ValueError(f'Invalid label name "{self.label}".')

        # TODO: spec to definitions.SPECTRUM, prec_mz to definitions.PRECURSOR_MZ, etc.
        item = {
            'spec': spectrum,
            'precursor mz': prec_mz,
            'label': label,
        }

        # TODO: remove, TMP fix
        item['charge'] = 1.0

        if self.return_smiles:
            item['smiles'] = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

        return item


class RawSpectraDataset(Dataset):
    def __init__(self, spectra, prec_mzs, spec_preproc: SpectrumPreprocessor):
        self.spectra = spectra
        self.prec_mzs = prec_mzs
        self.spec_preproc = spec_preproc

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, i):
        spectrum = self.spec_preproc(self.spectra[i], prec_mz=self.prec_mzs[i], high_form=False)
        return {SPECTRUM: spectrum, PRECURSOR_MZ: self.prec_mzs[i]}


class MatchmsSpectraDataset(Dataset):
    def __init__(self, spectra: List[Spectrum], spec_preproc: SpectrumPreprocessor):
        self.spectra = spectra
        self.spec_preproc = spec_preproc

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, i):
        spec = self.spectra[i]
        prec_mz = spec.metadata['precursor_mz']

        # Convert matchms spectrum to numpy array and preprocess
        spec = np.vstack([spec.peaks.mz, spec.peaks.intensities])       
        spec = self.spec_preproc(spec, prec_mz=prec_mz, high_form=False)

        return {SPECTRUM: spec, PRECURSOR_MZ: prec_mz}


# class RawPandasSpectraDataset(RawSpectraDataset):
#     def __init__(self, data: Union[Path, pd.DataFrame], spec_preproc: SpectrumPreprocessor, spec_col='PARSED PEAKS',
#                  prec_mz_col='PRECURSOR M/Z'):
#         if isinstance(data, Path):
#             if data.suffix == '.pkl':
#                 self.df = pd.read_pickle(data)
#             else:
#                 raise ValueError(f'Not supported input format "{data.suffix}" of the data file "{data}".')
#         elif isinstance(data, pd.DataFrame):
#             self.df = data
#         else:
#             raise ValueError(f'Not supported input format "{type(data)}" of the data.')
#         super().__init__(self.df[spec_col].tolist(), self.df[prec_mz_col].tolist(), spec_preproc)

    # def add_preds(self, labels: list, labels_name: str):
    #     if len(labels) != len(self):
    #         raise ValueError(f'Number of labels ({len(labels)}) does not match number of spectra ({len(self)}).')
    #     self.df[labels_name] = list(labels)
    #     if self.on_disk:
    #         self.df.to_pickle(self.data)
    #     return self.df


class ContrastiveSpectraDataset(Dataset):
    def __init__(self, df: pd.DataFrame, spec_preproc: SpectrumPreprocessor,
                 msn_spec_col='MSnSpectrum', pos_idx_col='pos_idx', neg_idx_col='neg_idx',
                 n_pos_samples=1, n_neg_samples=10, return_smiles=False, logger=None):
        self.df = df
        self.spec_preproc = spec_preproc
        self.msn_spec_col = msn_spec_col
        self.pos_idx_col = pos_idx_col
        self.neg_idx_col = neg_idx_col
        self.n_pos_samples = n_pos_samples
        self.n_neg_samples = n_neg_samples
        self.return_smiles = return_smiles
        self.logger = logger

    # def __len__(self):
    #     return len(self.df)

    def __getitem__(self, i):
        spec = self.df[self.msn_spec_col].loc[i]
        pos_idx = self.df[self.pos_idx_col].loc[i]
        neg_idx = self.df[self.neg_idx_col].loc[i]

        item = {'spec': self.spec_preproc(spec.get_peak_list(), prec_mz=spec.get_precursor_mz(), high_form=False)}
        if self.return_smiles:
            item['smiles'] = Chem.MolToSmiles(spec.get_precursor_mol())

        # Sample positive and negative spectra
        for k, idx, n_samples in [('pos', pos_idx, self.n_pos_samples), ('neg', neg_idx, self.n_neg_samples)]:
            idx = random.sample(idx, min(n_samples, len(idx)))
            specs = []
            for i in idx:
                spec = self.df[self.msn_spec_col].loc[i]
                specs.append(self.spec_preproc(spec.get_peak_list(), prec_mz=spec.get_precursor_mz(), high_form=False))
            item[f'{k}_specs'] = np.stack(specs)

            if self.return_smiles:
                item[f'{k}_smiles'] = [Chem.MolToSmiles(self.df['ROMol'].loc[i]) for i in idx]

        return item


class ImplExplValidation(ABC):

    def __init__(self, nist_like_pkl_pth, dformat: DataFormat, spec_preproc: SpectrumPreprocessor, df_idx=None,
                 n_samples=None, seed=1):

        # Load dataset with spectra
        if isinstance(nist_like_pkl_pth, pd.DataFrame):
            self.df = nist_like_pkl_pth
        else:
            self.df = io.cache_pkl(nist_like_pkl_pth).copy()
        if df_idx is not None:
            self.df = self.df.iloc[df_idx]
        if n_samples:
            self.df = self.df.sample(n=n_samples, random_state=seed)
        self.dformat = dformat
        self.spec_preproc = spec_preproc

        # Preprocess data
        self.spectra = []
        for i, row in self.df.iterrows():
            self.spectra.append(spec_preproc(row['PARSED PEAKS'], prec_mz=row['PRECURSOR M/Z'], high_form=False))
        self.spectra = np.stack(self.spectra)
        self.prec_mz = np.stack(self.df['PRECURSOR M/Z'].astype(float))
        self.charge = np.stack(self.df['CHARGE'].astype(int)) / dformat.max_charge
        self.df['i'] = range(len(self.df))  # For indexing torch tensors

        self.model_gains = None

    def get_data(self, device=None, torch_dtype=None):
        data = {
            'spec': torch.from_numpy(self.spectra),
            'prec_mz': torch.from_numpy(self.prec_mz),
            'charge': torch.from_numpy(self.charge)
        }
        if device:
            data = {k: v.to(device) for k, v in data.items()}
        if torch_dtype:
            data = {k: v.to(torch_dtype) for k, v in data.items()}
        return data

    def set_model_gains(self, model_gains):
        self.model_gains = model_gains

    @abstractmethod
    def get_res(self):
        return


class CSRKNN:
    def __init__(self, csr, one_minus_weights=True) -> None:
        self.csr = csr
        if one_minus_weights:
            edge_mask = csr.data > 0
            self.csr.data[edge_mask] = 1 - self.csr.data[edge_mask]

        self.k = len(self.neighbors(0)[0])
        self.n_nodes = self.csr.shape[0]
        self.n_edges = self.csr.nnz

    def neighbors(self, i, sort=True, exclude_self_loops=True) -> np.ndarray:
        """
        Get neighbors of the i-th node, i.e. all non-zero columns in i-th row of the CSR matrix.
        """
        nns = self.csr.indices[self.csr.indptr[i]:self.csr.indptr[i+1]]
        sims = self.csr.data[self.csr.indptr[i]:self.csr.indptr[i+1]]
        if exclude_self_loops:
            i_mask = nns != i
            nns, sims = nns[i_mask], sims[i_mask]
        if sort:
            sort_mask = np.argsort(sims)[::-1]
            nns, sims = nns[sort_mask], sims[sort_mask]
        return nns, sims

    def inv_neighbors(self, i, sort=True, exclude_self_loops=True) -> np.ndarray:
        """
        Get nodes that have i-th node as a neighbor.
        """
        nns = np.where(self.csr[:, i].toarray().flatten() > 0)[0]
        if exclude_self_loops:
            nns = nns[nns != i]
        sims = self.csr[nns, i].toarray().flatten()
        if sort:
            sort_mask = np.argsort(sims)[::-1]
            nns, sims = nns[sort_mask], sims[sort_mask]
        return nns, sims

    def __getitem__(self, i):
        return self.neighbors(i)

    @staticmethod
    def from_edge_list(s, t, w):
        """
        Construct CSR matrix from edge list.
        :param s: Source nodes.
        :param t: Target nodes.
        :param w: Edge weights.
        """
        n_nodes = max(s.max(), t.max()) + 1
        return CSRKNN(csr_matrix((w, (s, t)), shape=(n_nodes, n_nodes)))

    def to_edge_list(self, one_minus_weights=False):
        """
        Convert CSR matrix to edge list.
        """
        s, t, w = [], [], []
        for i in range(self.n_nodes):
            nns, sims = self.neighbors(i, sort=False)
            if one_minus_weights:
                sims = 1 - sims
            s.extend([i] * len(nns))
            t.extend(nns)
            w.extend(sims)
        return list(zip(s, t, w))

    @staticmethod
    def from_ngt_index(ngt_index, k, one_minus_for_weights=False):
        knn_i, knn_j, knn_w = [], [], []
        num_embs = ngt_index.get_num_of_objects()
        for i in tqdm(range(num_embs), desc='Constructing k-NN graph', total=num_embs):
            nns, sims = np.array(ngt_index.search(ngt_index.get_object(i), k + 1))[1:].T
            knn_i.extend([i] * k)
            knn_j.extend(nns)
            knn_w.extend(sims)
        knn_i, knn_j, knn_w = np.array(knn_i), np.array(knn_j), np.array(knn_w)
        if one_minus_for_weights:
            knn_w = 1 - knn_w
        return CSRKNN.from_edge_list(knn_i, knn_j, knn_w)

    def to_npz(self, pth: Path) -> None:
        """
        Save CSR matrix as COO matrix on disk. COO seems to work better and does not produce errors when saving large matrices.
        """
        coo_matrix = self.csr.tocoo()
        np.savez(pth, data=coo_matrix.data, row=coo_matrix.row, col=coo_matrix.col, shape=coo_matrix.shape)

    @staticmethod
    def from_npz(pth: Path, one_minus_weights=False):
        """
        Load CSR matrix that was stores as a COO matrix using `CSRKNN.save` method.
        """
        loaded_coo_matrix = np.load(pth)
        loaded_data = loaded_coo_matrix['data']
        loaded_row = loaded_coo_matrix['row']
        loaded_col = loaded_coo_matrix['col']
        loaded_shape = tuple(loaded_coo_matrix['shape'])
        return CSRKNN(
            csr=csr_matrix((loaded_data, (loaded_row, loaded_col)), shape=loaded_shape),
            one_minus_weights=one_minus_weights
        )

    def to_graph(self, directed=True, graph_class="igraph"):
        """
        Convert CSR matrix to a graph object using the specified library.
        
        Parameters:
        - directed (bool): If True, creates a directed graph. If False, creates an undirected graph.
        - graph_class (str): Specifies which graph library to use. 
                            "igraph" for igraph.Graph, "networkx" for networkx.Graph.
        
        Returns:
        - Graph object: igraph.Graph or networkx.Graph based on the specified `graph_class`.
        """

        s, t, w = [], [], []
        for i in tqdm(range(self.n_nodes), desc='Retrieving graph edges'):
            nns, sims = self.neighbors(i, sort=False)
            s.extend([i] * len(nns))
            t.extend(nns)
            w.extend(sims)

        if graph_class == "igraph":
            g = igraph.Graph(directed=directed)
            g.add_vertices(self.n_nodes)
            g.add_edges(list(zip(s, t)))
            g.es['weight'] = w
            if not directed:
                g.simplify(combine_edges='first')
            return g

        elif graph_class == "networkx":
            if directed:
                g = nx.DiGraph()
            else:
                g = nx.Graph()
            g.add_weighted_edges_from(zip(s, t, w))

            return g

        else:
            raise ValueError(f"Unknown graph_class: {graph_class}. Use 'igraph' or 'networkx'.")


def condense_dreams_knn(graph, thld, embs, logger):
    tqdm_logger = io.TqdmToLogger(logger)

    visited = set()
    clusters = []

    # Sort nodes by degree
    degrees = graph.degree()
    vertices = sorted(list(range(graph.vcount())), key=lambda i: degrees[i], reverse=True)
    
    for node in tqdm(vertices, desc='Forming clusters', file=tqdm_logger):
        if node not in visited:
            current_cluster = [node]
            visited.add(node)
            queue = [node]
            
            # Perform BFS from `node`
            while queue:
                current_node = queue.pop(0)

                in_one_hop = current_node == node

                for neighbor in graph.neighbors(current_node):

                    # Do not revisit nodes
                    if neighbor in visited:
                        continue
                    
                    # Go over each neighbour with similairty >= thld and add it to a cluster only if it guaranteed to
                    # transitively have similairty >= thld to cluster representative `node`
                    if graph.es[graph.get_eid(current_node, neighbor)]['weight'] >= thld:
                        if in_one_hop or 1 - cos_dist(embs[node], embs[neighbor]) >= thld:
                            visited.add(neighbor)
                            queue.append(neighbor)
                            current_cluster.append(neighbor)
            
            clusters.append(current_cluster)
    return clusters


class ManualValidation(ImplExplValidation):

    def __init__(self, nist_like_pkl_pth, dformat: DataFormat, spec_preproc: SpectrumPreprocessor, n_samples=None, seed=1, df_idx=None):
        super().__init__(nist_like_pkl_pth, dformat, spec_preproc, n_samples=n_samples, seed=seed, df_idx=df_idx)

    def get_res(self):
        assert self.model_gains is not None, 'First set embeddings with set_model_gains.'
        self.df['embedding'] = list(self.model_gains)
        return self.df


class AttentionEntropyValidation(ImplExplValidation):

    def __init__(self, nist_like_pkl_pth, dformat: DataFormat, spec_preproc: SpectrumPreprocessor, n_samples=None,
                 as_plot=False, save_out_basename=None):
        super().__init__(nist_like_pkl_pth, dformat, spec_preproc, n_samples=n_samples)
        self.as_plot = as_plot
        self.save_out_basename = save_out_basename

    def get_res(self):
        assert self.model_gains is not None, 'First set attention scores with set_model_gains.'
        return utils.calc_attention_entropy(self.model_gains, as_plot=self.as_plot,
                                            save_out_basename=self.save_out_basename)


class CorrelationValidation(ImplExplValidation):

    def __init__(self, nist_like_pkl_pth, corr_pkl_pth, dformat: DataFormat, spec_preproc: SpectrumPreprocessor, n_samples=None):

        self.df_corr = pd.read_pickle(corr_pkl_pth)

        if n_samples:
            self.df_corr = self.df_corr[:n_samples]

        super().__init__(nist_like_pkl_pth, dformat, spec_preproc, df_idx=np.unique(np.vstack([self.df_corr['i'], self.df_corr['j']])))

    def get_res(self):
        # TODO: refactor distances
        assert self.model_gains is not None, 'First set embeddings with set_model_gains.'

        self.df_corr['emb_cos'] = self.df_corr.apply(lambda row:
             1 - F.cosine_similarity(
                 self.model_gains[self.df['i'][int(row['i'])]],
                 self.model_gains[self.df['i'][int(row['j'])]],
                 dim=0
             ).detach().item()
         , axis=1)

        self.df_corr['emb_eucl'] = self.df_corr.apply(lambda row:
            (self.model_gains[self.df['i'][int(row['i'])]] - self.model_gains[self.df['i'][int(row['j'])]])
            .pow(2).sum().sqrt().detach().item()
        , axis=1)

        # Compute correlation for 1000 bootstrapped samples
        corrs_cos, corrs_eucl = [], []
        for i in range(1000):
            df_bootstrap = self.df_corr.sample(frac=1, replace=True)
            corrs_cos.append(df_bootstrap['emb_cos'].corr(df_bootstrap['metric']))
            corrs_eucl.append(df_bootstrap['emb_eucl'].corr(df_bootstrap['metric']))
        corrs_cos, corrs_eucl = np.array(corrs_cos), np.array(corrs_eucl)

        return {
            'Cos corr mean': corrs_cos.mean(),
            'Eucl corr mean': corrs_eucl.mean(),
            'Cos corr std': corrs_cos.std(),
            'Eucl corr std': corrs_eucl.std()
        }


class SpecRetrievalValidation(ImplExplValidation):
    """
    Cosine similarity on embeddings <-> equality of InChI keys validation.
    """

    def __init__(self, nist_like_pkl_pth, pairs_pkl_pth, dformat: DataFormat, spec_preproc: SpectrumPreprocessor):
        self.df_pairs = pd.read_pickle(pairs_pkl_pth)
        super().__init__(nist_like_pkl_pth, dformat, spec_preproc, df_idx=np.unique(self.df_pairs[['i', 'j']]))

    def get_res(self):

        # Compute cosine similarities of embeddings on the given pairs of spectra
        cos_sims = self.df_pairs.apply(
            lambda row: F.cosine_similarity(
                self.model_gains[self.df['i'][int(row['i'])]],
                self.model_gains[self.df['i'][int(row['j'])]]
            , dim=0).item()
        , axis=1)

        # Compute AUC against InChI keys equality binary labels
        fpr, tpr, thresholds = metrics.roc_curve(self.df_pairs['label'], cos_sims)
        auc = metrics.auc(fpr, tpr)

        return {'Spectrum retrieval AUC': auc}


class ContrastiveValidation(ImplExplValidation):

    def __init__(self, nist_like_pkl_pth, pairs_pkl_pth, dformat: DataFormat, spec_preproc: SpectrumPreprocessor,
                 n_instances=None, n_samples=None, seed=3, save_out_basename: pathlib.Path = None, euclidean=False):
        random.seed(seed)
        self.save_out_basename = save_out_basename
        self.euclidean = euclidean

        # Load dataset with defined spectra groups
        self.df_groups = pd.read_pickle(pairs_pkl_pth)

        # Select subsets if required
        if n_instances:
            self.df_groups = self.df_groups.iloc[:n_instances]
        if n_samples:
            self.df_groups['index'] = self.df_groups['index'].apply(lambda p: p[:n_samples])

        # Initialize main dataframe
        super().__init__(nist_like_pkl_pth, dformat, spec_preproc, df_idx=np.unique(np.concatenate(self.df_groups['index'].values)))

        self.labels = self.df[self.df_groups.index.name]

    def get_name(self):
        return self.df_groups.attrs['name']

    def get_labels(self):
        return self.labels

    def get_res(self):
        assert self.model_gains is not None, 'First set embeddings with set_model_gains.'

        res = {'Cos': []}
        if self.euclidean:
            res['Eucl'] = []

        for i in range(len(self.df_groups)):
            idx_i = self.df_groups['index'].iloc[i]
            embs_i = self.model_gains[self.df['i'][idx_i].tolist()]

            for metric_name, metric in [('Cos', pairwise_cosine_similarity), ('Eucl', pairwise_euclidean_distance)]:

                if metric_name == 'Eucl' and not self.euclidean:
                    break

                pos_dists_i = metric(embs_i, embs_i)
                pos_dists_i = torch.masked_select(pos_dists_i, torch.triu(pos_dists_i, diagonal=1).to(bool)).tolist()

                n = (len(self.df_groups) - 1) * (len(idx_i) ** 2 - len(idx_i)) // 2
                if len(pos_dists_i) > n:
                    pos_dists_i = random.sample(pos_dists_i, k=n)
                k = round(len(pos_dists_i) / (len(self.df_groups) - 1))

                neg_dists_i = []
                for j in range(len(self.df_groups)):
                    if j == i:
                        continue
                    idx_j = self.df_groups['index'].iloc[j]
                    embs_j = self.model_gains[self.df['i'][idx_j].tolist()]
                    dists = metric(embs_i, embs_j) if metric_name == 'Eucl' else 1 - metric(embs_i, embs_j)
                    dists = torch.masked_select(dists, torch.triu(dists, diagonal=1).to(bool)).tolist()
                    neg_dists_i.extend(random.sample(dists, k=k))

                res[metric_name].append(mean(pos_dists_i) - mean(neg_dists_i))
        return {f'{self.df_groups.index.name} {k} intra-inter diff': mean(v) for k, v in res.items()}

    def get_umap_plot(self):
        assert self.model_gains is not None, 'First set_embeddings with set_model_gains.'

        reducer = umap.UMAP(metric='cosine')
        umap_embs = reducer.fit_transform(self.model_gains.cpu())

        if self.save_out_basename:
            pd.DataFrame({'x': umap_embs[:, 0], 'y': umap_embs[:, 1], 'label': self.labels}).to_pickle(
                self.save_out_basename.with_suffix('.umap.pkl')
            )

        if self.labels.dtype == 'O':
            labels_ = self.labels.astype('category').cat.rename_categories(range(self.labels.nunique()))
        else:
            labels_ = self.labels
        fig = go.Figure(data=go.Scatter(x=umap_embs[:, 0], y=umap_embs[:, 1], mode='markers', marker_color=labels_,
                                        text=self.labels, marker=dict(size=10, colorscale='rainbow')))
        fig.update_layout(autosize=False, width=400, height=400, margin=dict(l=10, r=10, b=10, t=10, pad=4),
                          template='plotly_white')
        return fig


class KNNValidation(ContrastiveValidation):
    def __init__(self, nist_like_pkl_pth, pairs_pkl_pth, dformat: DataFormat, spec_preproc: SpectrumPreprocessor, k=3,
                 n_instances=None, n_samples=None, seed=3, save_out_basename: pathlib.Path = None):
        super().__init__(nist_like_pkl_pth, pairs_pkl_pth, dformat, spec_preproc, n_instances, n_samples, seed,
                         save_out_basename)
        self.k = k

    def get_res(self):
        assert self.model_gains is not None, 'First set embeddings with set_model_gains.'

        diffs = pairwise_cosine_similarity(self.model_gains)

        if not isinstance(self.k, Iterable):
            self.k = [self.k]

        res = {}
        for k in self.k:
            knn_precisions = []
            for i in range(diffs.shape[0]):
                topk = torch.topk(diffs[i], k).indices
                labels_knn = self.labels.iloc[topk.cpu().tolist()]
                knn_precisions.append((labels_knn == self.labels.iloc[i]).sum() / len(labels_knn))
            res[f'{self.get_name()} {k}-NN precision'] = mean(knn_precisions)
        return res


class CVDataModule(pl.LightningDataModule):

    def __init__(self, dataset: Dataset, fold_idx: pd.Series, batch_size: int, num_workers=0):
        super().__init__()
        self.dataset = dataset
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fold = None
        self.val_fold = None

    def setup_fold_index(self, fold_i: int) -> None:
        self.train_fold = Subset(self.dataset, self.fold_idx[self.fold_idx != fold_i].index.values)
        self.val_fold = Subset(self.dataset, self.fold_idx[self.fold_idx == fold_i].index.values)

    def get_num_folds(self) -> int:
        return self.fold_idx.nunique()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)


class RandomSplitDataModule(pl.LightningDataModule):

    def __init__(self, dataset, batch_size: int, max_var_features=None, val_frac=0.1, num_workers=0):
        super().__init__()
        self.dataset = dataset
        self.max_var_features = max_var_features
        if self.max_var_features is not None:
            assert len(self.dataset) == len(self.max_var_features)
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.num_workers = num_workers

        val_size = round(val_frac * len(dataset))
        train_size = len(dataset) - val_size
        self.train_subset, self.val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        if self.max_var_features is not None:
            batch_sampler = MaxVarBatchSampler(
                self.train_subset,
                self.max_var_features[self.train_subset.indices],
                batch_size=self.batch_size
            )
            return DataLoader(self.train_subset, batch_sampler=batch_sampler, num_workers=self.num_workers, shuffle=True)
        else:
            return DataLoader(self.train_subset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True,
                              drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_subset, drop_last=True, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return


class SplittedDataModule(pl.LightningDataModule):

    def __init__(self, dataset, split_mask: Union[pd.Series, np.ndarray, list], batch_size: Optional[int], num_workers=0,
                 n_train_samples=None, seed=None, include_val_in_train=False):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.include_val_in_train = include_val_in_train
        self.n_train_samples = n_train_samples
        self.seed = seed

        if not isinstance(split_mask, pd.Series) and not isinstance(split_mask, np.ndarray):
            split_mask = pd.Series(split_mask)

        if isinstance(split_mask, pd.Series):
            if split_mask.dtype == bool:
                # True for validation fold, False for train fold
                self.train_subset = Subset(self.dataset, split_mask[~split_mask].index.values)
                self.val_subset = Subset(self.dataset, split_mask[split_mask].index.values)
                self.test_subset = None
            elif set(split_mask) <= {'train', 'val', 'test', 'none'}:
                if self.include_val_in_train:
                    self.train_subset = Subset(self.dataset, split_mask[split_mask != 'test'].index.values)
                    self.val_subset = None
                    self.test_subset = Subset(self.dataset, split_mask[split_mask == 'test'].index.values)
                else:
                    self.train_subset = Subset(self.dataset, split_mask[split_mask == 'train'].index.values)
                    self.val_subset = Subset(self.dataset, split_mask[split_mask == 'val'].index.values)
                    self.test_subset = Subset(self.dataset, split_mask[split_mask == 'test'].index.values)
            else:
                raise ValueError(f'Invalid split mask with unique values: {set(split_mask)}.')
        elif isinstance(split_mask, np.ndarray):
            if split_mask.dtype == bool:
                # True for validation fold, False for train fold
                self.train_subset = Subset(self.dataset, np.where(~split_mask)[0])
                self.val_subset = Subset(self.dataset, np.where(split_mask)[0])
                self.test_subset = None
            else:
                raise ValueError('Invalid split mask.')
        else:
            raise ValueError('Invalid split mask.')

        # Sample random n_train_samples
        if self.n_train_samples:
            assert self.n_train_samples <= len(self.train_subset), 'n_train_samples is larger than the dataset size.'
            random.seed(seed.seed)
            rand_idx = random.sample(list(range(len(self.train_subset))), self.n_train_samples)
            self.train_subset = Subset(self.train_subset, rand_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_subset, shuffle=True, drop_last=True, num_workers=self.num_workers,
                          batch_size=self.batch_size if self.batch_size else len(self.train_subset))

    def val_dataloader(self) -> DataLoader:
        if self.val_subset:
            return DataLoader(self.val_subset, shuffle=False, drop_last=False, num_workers=self.num_workers,
                              batch_size=self.batch_size if self.batch_size else len(self.val_subset))

    def test_dataloader(self) -> DataLoader:
        if self.test_subset:
            return DataLoader(self.test_subset, shuffle=False, drop_last=False, num_workers=self.num_workers,
                              batch_size=self.batch_size if self.batch_size else len(self.test_subset))


class SSLProbingValidation(pl.Callback):

    def __init__(self, labeled_data_module: SplittedDataModule, evaluator_impl='torch', n_hidden_layers=[0], n_epochs=100,
                 probing_batch_freq=2500, prefix=None, save_fps_dir=Optional[Path]):
        # TODO: generalize beyond just fingerprint probing?
        super().__init__()
        assert not evaluator_impl == 'sklearn' or 0 not in n_hidden_layers, 'sklearn probing supports only a single linear layer.'
        self.labeled_data_module = labeled_data_module
        self.evaluator_impl = evaluator_impl
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.probing_batch_freq = probing_batch_freq
        self.prefix = prefix
        self.save_fps_dir = save_fps_dir
        if self.save_fps_dir:
            self.save_fps_dir.mkdir(parents=True, exist_ok=True)

    def _validate_sklearn(self, pl_module):
        raise NotImplementedError('Not tested after implementing fingerprint TorchMetrics.')
        # # Extract model embeddings for the labeled data
        # embs, fps = self._get_labeled_embs(pl_module, val=False, to_numpy=True)
        # embs_val, fps_val = self._get_labeled_embs(pl_module, val=True, to_numpy=True)
        # n_bits = fps.shape[1]
        #
        # # Train separate Logistic Regression model for each target label
        # def train_probe(i):
        #     probe = LogisticRegression(fit_intercept=False, max_iter=self.n_epochs)
        #     probe.fit(embs, fps[:, i])
        #     return probe
        # probes = Parallel(n_jobs=32, verbose=10)(delayed(train_probe)(i) for i in range(n_bits))
        #
        # # Predict target labels for each validation sample
        # fps_pred = np.stack([probes[i].predict_proba(embs_val)[:, 1] for i in range(n_bits)]).T
        #
        # # Compute metrics
        # return {f'[Linear probing] {k}': v for k, v in self.fp_metrics(fps_pred, fps).items()}

    # def _get_labeled_embs(self, pl_module, val: bool, to_numpy: bool):
    #     embs, labels = [], []
    #     data_loader = self.labeled_data_module.val_dataloader() if val else self.labeled_data_module.train_dataloader()
    #     for batch in data_loader:
    #         embs.append(dreams.get_embeddings(pl_module, batch, batch_size=32))
    #         labels.append(batch['label'])
    #     embs, labels = torch.cat(embs), torch.cat(labels)
    #
    #     if to_numpy:
    #         embs = embs.cpu().numpy()
    #         labels = labels.cpu().numpy()
    #
    #     return embs, labels

    def _validate_torch(self, pl_module):

        # Define linear probe and optimization
        n_bits = self.labeled_data_module.train_subset[0]['label'].shape[0]
        for n_hidden_layers in self.n_hidden_layers:
            if n_hidden_layers == 0:
                probe = nn.Linear(pl_module.d_model, n_bits, bias=True)
                lr = 0.1
            else:
                probe = FeedForward(in_dim=pl_module.d_model, out_dim=n_bits, depth=n_hidden_layers,
                                    hidden_dim=pl_module.d_model, bias=True, act_last=False)
                lr = 0.05  # TODO: find best lr.
            probe = nn.Sequential(probe, nn.Sigmoid()).to(device=pl_module.device, dtype=pl_module.dtype)
            optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
            criterion = nn.BCELoss() #CosSimLoss()

            # Forward pass for the probe
            def step(batch):
                spec = batch['spec'].to(device=pl_module.device, dtype=pl_module.dtype)
                labels = batch['label'].to(device=pl_module.device, dtype=pl_module.dtype)
                with torch.no_grad(), self._switch_ssl_training(pl_module, False):
                    embs = pl_module(spec)[:, 0, :]
                labels_pred = probe(embs)
                return labels_pred, labels

            fp_metrics = FingerprintMetrics(device=pl_module.device)
            best_metrics = {}
            for epoch in tqdm(range(self.n_epochs), desc='Probing train epoch', disable=True):

                # Train
                probe.train()
                for batch in tqdm(self.labeled_data_module.train_dataloader(), desc='Probing train batch', disable=True):

                    fps_pred, fps = step(batch)
                    loss = criterion(fps_pred, fps)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Validate
                probe.eval()
                with torch.no_grad():
                    for batch_i, batch in tqdm(enumerate(self.labeled_data_module.val_dataloader()), desc='Probing val batch', disable=True):
                        fps_pred, fps = step(batch)
                        if self.save_fps_dir:
                            torch.save({
                                'fps_pred': fps_pred.detach(),
                                'smiles': batch['smiles']
                            }, self.save_fps_dir / f'probing_pred_ssl_step={pl_module.trainer.global_step}_probe_epoch={epoch}_batch={batch_i}.pt')
                        fp_metrics.update(fps_pred, fps)

                # Compute epoch metrics
                metrics = fp_metrics.compute()
                if not best_metrics or metrics['CosineSimilarity'] > best_metrics['CosineSimilarity']:
                    best_metrics = metrics

                fp_metrics.reset()

            pl_module.log_dict({
                f'Probing {k} ({self.prefix + ", " if self.prefix else ""}depth={n_hidden_layers})': v
                for k, v in best_metrics.items()
            }, rank_zero_only=True)

    @contextmanager
    def _switch_ssl_training(self, module: nn.Module, mode: bool):
        """
        Context manager to set training mode for main SSL model. When exit, recover the original training mode.
        :arg module: module to set training mode
        :arg mode: whether to set training mode (True) or evaluation mode (False).
        """
        original_mode = module.training
        try:
            module.train(mode)
            yield module
        finally:
            module.train(original_mode)

    # from pytorch_lightning.utilities import rank_zero_only
    # @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):

        if batch_idx % self.probing_batch_freq != 0 and batch_idx != 0:
            return

        if self.evaluator_impl == 'sklearn':
            self._validate_sklearn(pl_module)
        elif self.evaluator_impl == 'torch':
            self._validate_torch(pl_module)
        else:
            raise ValueError(f'Invalid "evaluator_impl": {self.evaluator_impl}.')


def evaluate_split(df_split, n_workers=5, smiles_col=SMILES, fold_col='fold'):
    """
    Evaluates data split based on the Morgan Tanimoto similarity between validation and train folds.

    Args:
        df_split (pd.DataFrame): DataFrame containing the data split.
        n_workers (int): Number of workers for parallel processing. Default is 5.
        smiles_col (str): Column name for SMILES strings. Default is the SMILES constant.
        fold_col (str): Column name for fold information. Default is 'fold'.

    Returns:
        dict: A dictionary containing maximum Tanimoto similarities for each fold.

    Raises:
        ValueError: If the fold column is not found or contains invalid values.
    """

    # Validate input data
    if fold_col not in df_split.columns:
        raise ValueError(f'Fold column "{fold_col}" not found in DataFrame columns: {df_split.columns}.')
    if not set(df_split[fold_col]) <= set(['train', 'val', 'test']):
        raise ValueError(f'Invalid fold values: {set(df_split[fold_col])}.')

    # Generate Morgan fingerprints for each fold
    fps = {}
    for fold in df_split[fold_col].unique():
        df_fold = df_split[df_split[fold_col] == fold]
        df_fold = df_fold.drop_duplicates(subset=smiles_col)
        fps[fold] = df_fold[smiles_col].progress_apply(lambda s: mu.morgan_fp(Chem.MolFromSmiles(s), as_numpy=False))

    # Function to compute maximum Tanimoto similarity
    def max_train_tanimoto(test_fp, train_fps=fps['train']):
        return max([DataStructs.FingerprintSimilarity(test_fp, train_fp) for train_fp in train_fps])

    # Initialize parallel processing
    pandarallel.initialize(nb_workers=n_workers, progress_bar=True, use_memory_fs=False)

    # Compute maximum Tanimoto similarities
    max_train_tanimotos = {}
    if len(df_split[fold_col].unique()) == 2:
        # For binary split (e.g., train/val)
        max_train_tanimotos[fold] = fps[fold].parallel_apply(lambda f: max_train_tanimoto(f, fps['train']))
    else:
        # For ternary split (e.g., train/val/test)
        max_train_tanimotos[fold] = fps[fold].parallel_apply(lambda f: max_train_tanimoto(f, pd.concat([fps['train'], fps['val']])))

    return max_train_tanimotos


# NOTE: deprecated because not suited for (wandb) logging
# class CVLoop(Loop):
#
#     def __init__(self, num_folds: int, export_path: str) -> None:
#         super().__init__()
#         self.num_folds = num_folds
#         self.current_fold: int = 0
#         self.export_path = export_path
#
#     @property
#     def done(self) -> bool:
#         return self.current_fold >= self.num_folds
#
#     def connect(self, fit_loop: FitLoop) -> None:
#         self.fit_loop = fit_loop
#
#     def reset(self) -> None:
#         """Nothing to reset in this loop."""
#
#     def on_run_start(self, *args: Any, **kwargs: Any) -> None:
#         """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
#         model."""
#         assert isinstance(self.trainer.datamodule, CVDataModule)
#         self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())
#
#     def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
#         """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
#         print(f'STARTING FOLD {self.current_fold}')
#         assert isinstance(self.trainer.datamodule, CVDataModule)
#         self.trainer.datamodule.setup_fold_index(self.current_fold)
#         # self.trainer.lightning_module.__setattr__('fold_i', self.current_fold)
#         setattr(self.trainer.lightning_module, 'fold_i', self.current_fold)
#         print('getattr(self.trainer.lightning_module, "fold_i"):', getattr(self.trainer.lightning_module, 'fold_i'))
#
#     def advance(self, *args: Any, **kwargs: Any) -> None:
#         """Used to the run a fitting and testing on the current hold."""
#         self._reset_fitting()  # requires to reset the tracking stage.
#         self.fit_loop.run()
#
#         self._reset_testing()  # requires to reset the tracking stage.
#         self.trainer.test_loop.run()
#         self.current_fold += 1  # increment fold tracking number.
#
#     def on_advance_end(self) -> None:
#         """Used to save the weights of the current fold and reset the LightningModule and its optimization."""
#         self.trainer.save_checkpoint(osp.join(self.export_path, f'cv_model.{self.current_fold}.pt'))
#         # restore the original weights + optimization and schedulers.
#         self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
#         self.trainer.strategy.setup_optimizers(self.trainer)
#         self.replace(fit_loop=FitLoop)
#
#     # def on_run_end(self) -> None:
#     #     """Used to compute the performance of the ensemble model on the test set."""
#     #     checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
#     #     voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
#     #     voting_model.trainer = self.trainer
#     #     # This requires to connect the new model and move it the right device.
#     #     self.trainer.strategy.connect(voting_model)
#     #     self.trainer.strategy.model_to_device()
#     #     self.trainer.test_loop.run()
#
#     def on_save_checkpoint(self) -> Dict[str, int]:
#         return {'current_fold': self.current_fold}
#
#     def on_load_checkpoint(self, state_dict: Dict) -> None:
#         self.current_fold = state_dict['current_fold']
#
#     def _reset_fitting(self) -> None:
#         self.trainer.reset_train_dataloader()
#         self.trainer.reset_val_dataloader()
#         self.trainer.state.fn = TrainerFn.FITTING
#         self.trainer.training = True
#
#     def _reset_testing(self) -> None:
#         self.trainer.reset_test_dataloader()
#         self.trainer.state.fn = TrainerFn.TESTING
#         self.trainer.testing = True
#
#     def __getattr__(self, key) -> Any:
#         # requires to be overridden as attributes of the wrapped loop are being accessed.
#         if key not in self.__dict__:
#             return getattr(self.fit_loop, key)
#         return self.__dict__[key]
#
#     def __setstate__(self, state: Dict[str, Any]) -> None:
#         self.__dict__.update(state)
