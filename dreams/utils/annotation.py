import h5py
import atexit
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
# from FPSim2 import FPSim2Engine  # NOTE: installed from local fork
from abc import abstractmethod
from pathlib import Path
from typing import List, Union, Iterable
from collections import Counter

import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
tqdm.pandas()
from rdkit.DataStructs import ExplicitBitVect
import dreams.utils.mols as mu
from dreams.utils.io import setup_logger
from dreams.definitions import *


class SpectralLibraryRetrieval:
    def __init__(
            self,
            df_lib: Union[Path, pd.DataFrame],
            # df_query: Union[Path, pd.DataFrame]
    ):
        self.df_lib = df_lib.reset_index()
        # self.df_query = df_query
        if isinstance(self.df_lib, Path):
            self.df_lib = pd.read_pickle(self.df_lib)
        # if isinstance(self.df_query, Path):
        #     self.df_query = pd.read_pickle(self.df_query)

        assert 'ROMol' in self.df_lib.columns, 'ROMol column is not present in spectral library.'
        if 'DreaMS' not in self.df_lib.columns:
            # TODO: compute DreaMS embeddings
            raise NotImplementedError()
        self.dreams_lib = np.stack(self.df_lib['DreaMS'].values)

    def retrieve(self, query_dreams_emb: np.ndarray, top_k=np.inf, precursor_mz=None, prec_mz_tolerance=None):
        assert (precursor_mz is None) + (prec_mz_tolerance is None) in {0, 2}

        if prec_mz_tolerance:
            df_lib = self.df_lib[(self.df_lib['PRECURSOR M/Z'] - precursor_mz).abs() < prec_mz_tolerance]
            if len(df_lib) == 0:
                return None
            dreams_lib = self.dreams_lib[df_lib.index]
        else:
            df_lib = self.df_lib
            dreams_lib = self.dreams_lib

        # Compute cosine similarities between query fingerprint and candidate fingerprints
        sims = cosine_similarity(dreams_lib, query_dreams_emb[np.newaxis, :]).flatten()

        # Sort similarities from highest to lowest
        sorted_idx = np.argsort(sims)[::-1]

        top_k_idx = sorted_idx[:min(top_k, len(sorted_idx))]

        df_res = df_lib.iloc[top_k_idx].copy()
        df_res['Library DreaMS score'] = sims[top_k_idx]

        return df_res

    def retrieve_df(self, df, prec_mz_tolerance=0.01, top_k=10):
        # TODO: ID is hardcoded now

        if 'DreaMS' not in df.columns:
            # TODO: compute DreaMS embeddings
            raise NotImplementedError()
        if prec_mz_tolerance and 'PRECURSOR M/Z' not in df.columns:
            raise ValueError('PRECURSOR M/Z column is not present in query DataFrame.')

        res_cols = {
            'Library SMILES': [],
            'Library DreaMS score': [],
            'Library ID': []
        }
        for i, row in tqdm(df.iterrows(), total=len(df)):
            df_res = self.retrieve(np.array(row['DreaMS']), precursor_mz=row['PRECURSOR M/Z'], top_k=top_k, prec_mz_tolerance=prec_mz_tolerance)
            if df_res is None:
                for k in res_cols.keys():
                    res_cols[k].append(None)
            else:
                res_cols['Library SMILES'].append(df_res['SMILES'].values)
                res_cols['Library DreaMS score'].append(df_res['Library DreaMS score'].values)
                res_cols['Library ID'].append(df_res['ID'].values)
        for k, v in res_cols.items():
            df[k] = v

        # def retrieve_row(row):
        #     df_res = self.retrieve(np.array(row['DreaMS']), precursor_mz=row['PRECURSOR M/Z'], top_k=top_k, prec_mz_tolerance=prec_mz_tolerance)
        #     # Return tuple of column lists from the retrieval result DataFrame
        #     if df_res is None:
        #         print('++++++++++')
        #         return [None] * 3
        #     print(len(df_res))
        #     return df_res[['SMILES', 'Library DreaMS score', 'ID']].T.itertuples(index=False, name=None)
        # df[['Library SMILES', 'Library DreaMS score', 'Library ID']] = df.progress_apply(retrieve_row, axis=1)

        return df


class FingerprintInChIRetrieval:
    def __init__(
            self,
            df_pkl_pth: Path,
            candidate_smiles_col: str,
            fp_name: str,
            top_k: Union[int, List[int]],
            index_smiles_col: str = 'SMILES',
            candidate_inchi14_col: str = None
    ):
        self.df = pd.read_pickle(df_pkl_pth)
        for c in [index_smiles_col, candidate_smiles_col]:
            assert not c or c in self.df.columns, f'"{c}" is not in DataFrame columns.'

        # Set SMILES columns as index
        self.df = self.df.drop_duplicates(subset=[index_smiles_col])
        self.df = self.df.set_index(index_smiles_col, drop=True)

        self.candidate_smiles_col = candidate_smiles_col
        self.candidate_inchi14_col = candidate_inchi14_col
        self.fp_func = mu.fp_func_from_str(fp_name)
        self.top_k = [top_k] if isinstance(top_k, int) else top_k
        self.__reset_metrics()
        mu.disable_rdkit_log()

    def __reset_metrics(self):
        self.metrics = Counter({f'Accuracy @ {k}': 0 for k in self.top_k})
        self.n_retrievals = 0

    def __update_metrics(self, retrieved_inchi14s, label_inchi14):
        for k in self.top_k:
            self.metrics[f'Accuracy @ {k}'] += label_inchi14 in retrieved_inchi14s[:k]
        self.n_retrievals += 1

    def retrieve_inchi14s(self, query_fp, label_smiles):
        # Init metrics
        if not self.metrics:
            self.__reset_metrics()

        # Get candidate SMILES
        candidate_smiles = self.df[self.candidate_smiles_col][label_smiles]

        # Compute candidate fingerprints
        candidate_fps = np.stack([self.fp_func(Chem.MolFromSmiles(s)) for s in candidate_smiles])

        # Compute cosine similarities between query fingerprint and candidate fingerprints
        sims = cosine_similarity(candidate_fps, query_fp[np.newaxis, :]).flatten()

        # Sort similarities from highest to lowest
        sorted_idx = np.argsort(sims)[::-1]

        # Retrieve InChI key first blocks
        retrieved_smiles, retrieved_inchi14s = [], []
        for i in sorted_idx:
            smiles = self.df[self.candidate_smiles_col][label_smiles][i]
            retrieved_smiles.append(smiles)
            if self.candidate_inchi14_col:
                inchi14 = self.df[self.candidate_inchi14_col][label_smiles][i]
            else:
                inchi14 = mu.smiles_to_inchi14(smiles)
            retrieved_inchi14s.append(inchi14)

        # Update metrics and return retrievals
        self.__update_metrics(retrieved_inchi14s, mu.smiles_to_inchi14(label_smiles))
        return retrieved_inchi14s[:max(self.top_k)], retrieved_smiles[:max(self.top_k)], sims[sorted_idx][:max(self.top_k)]

    def compute_reset_metrics(self, metrics_prefix='', return_unaveraged=False):
        # Average and reset metrics
        metrics_avg = {f'{metrics_prefix} {k}'.strip(): v / self.n_retrievals for k, v in self.metrics.items()}
        metrics = self.metrics.copy()
        metrics['n_retrievals'] = self.n_retrievals
        self.__reset_metrics()

        if return_unaveraged:
            return metrics_avg, metrics
        return metrics_avg


# class MolFingerprintRetrieval:
#
#     @abstractmethod
#     def similarity_search(self, query_fp, **sim_kwargs):
#         pass
#
#     def retrieve_fp(self, fp: np.ndarray, topk, return_k_idx=False, **sim_kwargs):
#         """
#         :param return_k_idx: If True, returns tuple (retrieved molecules, k_idx), where k_idx is 1D array of length
#         min(topk, unique retrieval scores) with i-th value representing the last index of top i retrieved molecules.
#         """
#
#         res = self.similarity_search(fp, **sim_kwargs)
#         if len(res) == 0:
#             return []
#
#         # Update k considering possible identical top similarities
#         # TODO: other options
#         scores = np.array([r[1] for r in res])
#         _, unique_indices = np.unique(scores[::-1], return_index=True)
#         top_idx = scores.size - unique_indices[::-1]
#         k = top_idx[min(topk - 1, len(top_idx) - 1)]
#
#         # k = topk
#         # top_idx = np.arange(topk) + 1
#
#         res = res[:k]
#
#         if return_k_idx:
#             return res, top_idx[:k]
#
#         return res
#
#
# class FPSim2Retrieval(MolFingerprintRetrieval):
#     def __init__(self, fpsim2_index: Path, in_mem=True):
#
#         logger = setup_logger()
#         if in_mem:
#             logger.info(f'Loading FPSim2 index {fpsim2_index}...')
#
#         self.index = FPSim2Engine(fpsim2_index, in_memory_fps=in_mem)
#         logger.info(f'Retrieval index was constructed with {self.index.fp_type} {self.index.fp_params} fingerprints'
#                     f'(RDKit version {self.index.rdkit_ver}).')
#
#         f = h5py.File(fpsim2_index, 'r')
#         self.smiles = f['smiles']
#         atexit.register(f.close)
#
#         self.in_mem = in_mem
#
#     def similarity_search(self, query_fp, thld, n_workers):
#         """ Tanimoto similarity search """
#         query_fp = mu.np_to_rdkit_fp(query_fp)
#         if self.in_mem:
#             res = self.index.similarity(query_fp, thld, n_workers)
#         else:
#             res = self.index.on_disk_similarity(query_fp, thld, n_workers)
#         return [(self._i_to_smiles(r[0]), r[1]) for r in res]
#
#     def _i_to_smiles(self, i):
#         return self.smiles[i - 1].decode()  # FPSim2 indices start from 1 and stored as binary strings
#
#
# class PreAnnotatedRetrieval(MolFingerprintRetrieval):
#     def __init__(self, pkl_pth: Path, candidates_col='retrieval_candidates', id_col='SMILES', metric='cos',
#                  fp_str='fp_morgan_4096'):
#         """
#         :param pkl_pth: Path to pickled Pandas dataframe.
#         :param candidates_col: Name of the column containing candidate SMILES.
#         :param id_col: Name of the column serving as id to recover candidates for each query.
#         """
#         self.candidates_col = candidates_col
#         self.df = pd.read_pickle(pkl_pth)
#         assert self.candidates_col in self.df, f'No {self.candidates_col} in retrieval index columns.'
#         self.df = self.df.drop_duplicates(subset=id_col)
#         self.df = self.df.set_index(id_col)
#         mu.disable_rdkit_log()
#         self.metric = metric
#         self.fp_func = mu.fp_func_from_str(fp_str)
#
#     def _smooth_iou(self, query, fps, smooth=1):
#         intersection = (fps * query).sum(axis=-1)
#         total = (fps + query).sum(axis=-1)
#         union = total - intersection
#         return (intersection + smooth) / (union + smooth)
#
#     def _cos(self, query, fps):
#         return cosine_similarity(fps, query[np.newaxis, :]).flatten()
#
#     def similarity_search(self, query_fp, candidates_id):
#
#         # Select candidate SMILES
#         cands = self.df.loc[candidates_id][self.candidates_col]
#
#         # Compute fingerprint for each candidate
#         fps = np.stack([self.fp_func(Chem.MolFromSmiles(c)) for c in cands])
#
#         # Compute similarities of fps to query fp
#         if self.metric == 'cos':
#             sims = self._cos(query_fp, fps)
#         elif self.metric == 'smooth_iou':
#             sims = self._smooth_iou(query_fp, fps)
#         else:
#             ValueError('Invalid metric name:', self.metric)
#
#         # Rank candidates
#         sorted_idx = np.argsort(sims)[::-1]
#         res = np.vstack([sorted_idx, sims[sorted_idx]]).T
#
#         # Get SMILES
#         res = [(cands[int(r[0])], r[1]) for r in res]
#
#         return res
#
#
# def val_retrieval_at_k(preds: List, real: str, ks, topk_idx):
#     # TODO: more metrics
#
#     if not isinstance(ks, Iterable):
#         ks = [ks]
#
#     # Compute accuracy based on InChI key connectivity blocks
#     acc = {}
#     preds = [mu.smiles_to_inchi14(s[0]) for s in preds]
#     real = mu.smiles_to_inchi14(real)
#     for k in ks:
#         k_i = topk_idx[min(k - 1, len(topk_idx) - 1)]
#         acc[f'Accuracy @ {k}'] = int(real in preds[:k_i])
#     return acc
#
#
# def val_retrievals_at_k(preds: List[List], real: List, ks, topk_idx):
#
#     # Compute metrics for each fp
#     acc = {}  # collections.Counter removes zero-valued keys...
#     for i in range(len(preds)):
#         for k, v in val_retrieval_at_k(preds[i], real[i], ks, topk_idx[i]).items():
#             if k not in acc.keys():
#                 acc[k] = v
#             else:
#                 acc[k] += v
#
#     # Mean over fps
#     acc = {k: v / len(preds) for k, v in acc.items()}
#     return acc