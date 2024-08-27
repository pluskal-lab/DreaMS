import h5py
import torch
import pandas as pd
import typing as T
import networkx as nx
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import deque
from torch.utils.data.dataloader import DataLoader
import dreams.utils.data as du
import dreams.utils.io as io
import dreams.utils.dformats as dformats
import dreams.utils.misc as utils
from dreams.models.dreams.dreams import DreaMS as DreaMSModel
from dreams.models.heads.heads import *
from dreams.definitions import *


class PreTrainedModel:
    def __init__(self, model: T.Union[DreaMSModel, FineTuningHead], n_highest_peaks: int = 100):
        self.model = model.eval()
        self.n_highest_peaks = n_highest_peaks

    @classmethod
    def from_ckpt(cls, ckpt_path: Path, ckpt_cls: T.Union[T.Type[DreaMSModel], T.Type[FineTuningHead]], n_highest_peaks: int):
        return cls(
            ckpt_cls.load_from_checkpoint(
                ckpt_path,
                backbone_pth=PRETRAINED / 'ssl_model.ckpt',
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ),
            n_highest_peaks=n_highest_peaks
        )

    @classmethod
    def from_name(cls, name: str):
        if name == DREAMS_EMBEDDING:
            ckpt_path = PRETRAINED / 'embedding_model.ckpt'
            ckpt_cls = ContrastiveHead
            n_highest_peaks = 100
        # elif name == 'Fluorine probability':
        #     ckpt_path = EXPERIMENTS_DIR / 'pre_training/HAS_F_1.0/CtDh6OHlhA/epoch=6-step=71500_v2_16bs_5e-5lr_gamma0.5_alpha0.8/epoch=30-step=111000.ckpt'
        #     ckpt_cls = BinClassificationHead
        #     n_highest_peaks = 100
        # elif name == 'Molecular properties':
        #     ckpt_path = EXPERIMENTS_DIR / f'pre_training/MS2PROP_1.0/lr3e-5_bs64/epoch=4-step=4000.ckpt'
        #     ckpt_cls = RegressionHead
        #     n_highest_peaks = 100
        else:
            # TODO: Include all pre-trained models
            raise ValueError(f'{name} is not a valid pre-trained model name. Choose from: {cls.available_models()}')

        return cls.from_ckpt(ckpt_path, ckpt_cls, n_highest_peaks)

    # def __init_model(self, model: T.Union[DreaMSModel, FineTuningHead], n_highest_peaks: int = 100):
    #     self.model = model.eval()
    #     self.n_highest_peaks = n_highest_peaks

    @staticmethod
    def available_models():
        return ['Fluorine probability', 'Molecular properties', DREAMS_EMBEDDING]


def compute_dreams_predictions(
        model_ckpt: T.Union[PreTrainedModel, FineTuningHead, DreaMSModel, Path, str], spectra: T.Union[Path, str],
        model_cls=None, batch_size=32, tqdm_batches=True, write_log=False, n_highest_peaks=None, title='',
        **msdata_kwargs
    ):

    # Load pre-trained model
    if not isinstance(model_ckpt, PreTrainedModel):
        if isinstance(model_ckpt, str):
            if '/' in model_ckpt:
                model_ckpt = Path(model_ckpt)
            else:
                title = model_ckpt
        if isinstance(model_ckpt, str):
            model_ckpt = PreTrainedModel.from_name(model_ckpt)
        elif isinstance(model_ckpt, Path):
            model_ckpt = PreTrainedModel.from_ckpt(model_ckpt, model_cls, n_highest_peaks)
        else:
            model_ckpt = PreTrainedModel(model_ckpt, n_highest_peaks)

    # Initialize spectrum preprocessing
    spec_preproc = du.SpectrumPreprocessor(dformat=dformats.DataFormatA(), n_highest_peaks=model_ckpt.n_highest_peaks)

    # Load a dataset of spectra
    if isinstance(spectra, str):
        spectra = Path(spectra)
    spectra_pth = spectra

    msdata = du.MSData.load(spectra, **msdata_kwargs)
    spectra = msdata.to_torch_dataset(spec_preproc)
    dataloader = DataLoader(spectra, batch_size=batch_size, shuffle=False, drop_last=False)

    logger = io.setup_logger(spectra_pth.with_suffix('.log'))
    tqdm_logger = io.TqdmToLogger(logger)

    # Compute predictions
    model = model_ckpt.model
    # TODO: consider model name
    if not title:
        title = 'DreaMS_prediction'
    preds = None
    progress_bar = tqdm(
        total=len(spectra),
        desc='Computing ' + title.replace('_', ' '),
        disable=not tqdm_batches,
        file=tqdm_logger if write_log else None
    )
    for i, batch in enumerate(dataloader):
        with torch.inference_mode():
            pred = model(batch['spec'].to(device=model.device, dtype=model.dtype))

            # Store predictions to cpu to avoid high memory allocation issues
            pred = pred.cpu()
            if preds is None:
                preds = pred
            else:
                preds = torch.cat([preds, pred])

            # Update the progress bar by the number of samples in the current batch
            progress_bar.update(len(batch['spec']))
    progress_bar.close()

    preds = preds.squeeze().cpu().numpy()

    # TODO: move to outer scope
    # msdata.add_column(title, preds)
    return preds


def compute_dreams_embeddings(pth, batch_size=32, tqdm_batches=True, write_log=False, **msdata_kwargs):
    return compute_dreams_predictions(
        DREAMS_EMBEDDING, pth, batch_size=batch_size, tqdm_batches=tqdm_batches, write_log=write_log, **msdata_kwargs
    )


def generate_all_dreams_predictions(pth: T.Union[Path, str], batch_size=32, tqdm_batches=True,
                                 spec_col='PARSED PEAKS', prec_mz_col='PRECURSOR M/Z'):

    if isinstance(pth, str):
        pth = Path(pth)

    out_pth = io.append_to_stem(pth, 'DreaMS').with_suffix('.hdf5')

    with h5py.File(out_pth, 'w') as f:
        for m in PreTrainedModel.available_models():
            preds = compute_dreams_predictions(m, pth, batch_size=batch_size, tqdm_batches=tqdm_batches,
                                           spec_col=spec_col, prec_mz_col=prec_mz_col)
            preds = preds.squeeze().cpu().numpy()

            if m == 'Molecular properties':
                for i, p in enumerate(mu.MolPropertyCalculator().prop_names):
                    f.create_dataset(p, data=preds[:, i])
            else:
                f.create_dataset(m, data=preds)


# TODO: refactor after `get_dreams_predictions` is refactored
# def get_dreams_embeddings(model: T.Union[Path, str, DreaMS], df_spectra: T.Union[Path, str, pd.DataFrame], layers_idx=None,
#                           precursor_only=True, batch_size=32, tqdm_batches=True, spec_col='PARSED PEAKS',
#                           prec_mz_col='PRECURSOR M/Z', n_highest_peaks=128, return_attention_matrices=False,
#                           spec_preproc: du.SpectrumPreprocessor = None):
#
#     # Load model and spectra
#     model = load_model(model, model_cls=DreaMS)
#     dataloader = load_spectra(df_spectra, batch_size, spec_col=spec_col, prec_mz_col=prec_mz_col,
#                               n_highest_peaks=n_highest_peaks, spec_preproc=spec_preproc)
#
#     # Determine layers to extract embeddings from
#     if not layers_idx:
#         layers_idx = [model.n_layers - 1]
#
#     # Register hooks extracting embeddings
#     hook_handles = []
#     embeddings = {}
#     def get_embeddings_hook(name):
#         def hook(model, input, output):
#             embs = output.detach()
#             if precursor_only:
#                 embs = embs[:, 0, :]
#             if name not in embeddings.keys():
#                 embeddings[name] = embs
#             else:
#                 embeddings[name] = torch.cat([embeddings[name], embs])
#         return hook
#     for i in layers_idx:
#         hook_handles.append(model.transformer_encoder.ffs[i].register_forward_hook(get_embeddings_hook(i)))
#
#     # Register hooks extracting attention matrices
#     attn_matrices = {}
#     if return_attention_matrices:
#         def get_attn_scores_hook(name):
#             def hook(model, input, output):
#                 output = output[1].detach()
#                 if name not in attn_matrices.keys():
#                     attn_matrices[name] = output
#                 else:
#                     attn_matrices[name] = torch.cat([attn_matrices[name], output])
#             return hook
#         for i in layers_idx:
#             hook_handles.append(model.transformer_encoder.atts[i].register_forward_hook(get_attn_scores_hook(i)))
#
#     # Perform forward passes
#     for batch in tqdm(dataloader, desc='Computing DreaMS', disable=not tqdm_batches):
#         with torch.inference_mode():
#             _ = model(batch['spec'].to(device=model.device, dtype=model.dtype))
#
#     # Remove hooks
#     for h in hook_handles:
#         h.remove()
#
#     # Simplify output if only one layer was requested (no 1-element list)
#     if len(layers_idx) == 1:
#         embeddings = embeddings[layers_idx[0]]
#
#     if return_attention_matrices:
#         return embeddings, attn_matrices
#     return embeddings


# TODO: get_dreams_embeddings as a wrapper over pre-defined get_dreams_predictions
# TODO: cache loaded models in class


class DreaMSAtlas:
    def __init__(self):

        print('Initializing DreaMS Atlas data structures...')
        self.lib = du.MSData(
            utils.gems_hf_download('DreaMS_Atlas/nist20_mona_clean_merged_spectra_dreams.hdf5'),
            in_mem=False
        )
        print(f'Loaded spectral library ({len(self.lib):,} spectra).')

        self.gems = du.MSData.from_hdf5_chunks(
            [utils.gems_hf_download(f'GeMS_C/GeMS_C1_DreaMS.{i}.hdf5') for i in range(10)],
            in_mem=False
        )
        print(f'Loaded GeMS-C1 dataset ({len(self.gems):,} spectra).')

        self.csrknn = du.CSRKNN.from_npz(
            utils.gems_hf_download(f'DreaMS_Atlas/DreaMS_Atlas_3NN.npz'),
        )
        print(f'Loaded DreaMS Atlas edges ({self.csrknn.n_edges:,} edges).')

        self.dreams_clusters = pd.read_csv(
            utils.gems_hf_download(f'DreaMS_Atlas/DreaMS_Atlas_3NN_clusters.csv'),
            in_mem=False
        )['clusters']
        print(f'Loaded DreaMS Atlas nodes representing DreaMS k-NN clusters of GeMS-C1 ({self.dreams_clusters.nunique():,} nodes).')

        self.gems_lsh = du.MSData.from_hdf5_chunks(
            [utils.gems_hf_download(f'GeMS_C/GeMS_C.{i}.hdf5') for i in range(10)],
            in_mem=False
        )
        self.lsh_clusters = self.gems_lsh['lsh'][:]
        print(f'Loaded LSH clusters of DreaMS Atlas nodes representing GeMS-C ({len(self.lsh_clusters):,} spectra).')

        self.msv_metadata = pd.read_csv(
            utils.gems_hf_download('DreaMS_Atlas/massive_metadata.tsv'), sep='\t', comment='#', low_memory=False
        )
        self.msv_metadata = self.msv_metadata.set_index('dataset')
        self.msv_metadata = self.msv_metadata[[
            'species', 'species_resolved', 'instrument', 'instrument_resolved', 'title', 'description', 'create_time',
            'user', 'keywords'
        ]]
        self.msv_metadata = self.msv_metadata.rename(columns={c: 'msv_' + c for c in self.msv_metadata.columns})

        self.knn_i_to_repr = np.unique(self.dreams_clusters)
        self.repr_to_knn_i = dict(zip(
            self.knn_i_to_repr,
            list(range(len(self.dreams_clusters)))
        ))

    def get_node_repr(self, i):
        return self.dreams_clusters.iloc[i]

    def get_node_cluster(self, i, data=True, lsh=False, msv_metadata=False):
        node_repr_i = self.get_node_repr(i)
        idx = np.where(self.dreams_clusters == node_repr_i)[0]
        if lsh:
            return {i: self.get_lsh_cluster(i) for i in idx}
        elif data:
            return self.get_data(idx, msv_metadata=msv_metadata)
        
        return idx

    def get_lsh_cluster(
            self,
            i,
            as_dataframe=False,
            vals=None,
            msv_metadata=False
        ):
        if self.is_library_i(i):
            return np.array([i])
        lsh = self.get_data(i)[i]['lsh']
        idx = np.where(self.lsh_clusters == lsh)[0]

        cluster = []
        for j in idx:
            data = self.gems_lsh.at(
                j, plot_mol=False, plot_spec=False, return_spec=True, vals=vals
            )
            if msv_metadata:
                data = self._add_msv_metadata(data)
            if vals:
                data = self._subset_data(data, vals)
            data['spec_id'] = j
            data['node_id'] = i
            cluster.append(data)
        if as_dataframe:
            return pd.DataFrame(cluster)
        return cluster

    def get_neighbors(
            self,
            i,
            n_hops=1,
            inv_neighbors=False,
            sim_thld=-np.inf,
            as_dataframe=False,
            data_vals=None,
            msv_metadata=False,
            return_spec=True
        ):
        bfs_graph = self._bfs(node=self.get_node_repr(i), n_hops=n_hops, inv_neighbors=inv_neighbors, sim_thld=sim_thld)
        nodes_data = self.get_data(bfs_graph.nodes(), vals=data_vals, msv_metadata=msv_metadata, return_spec=return_spec)
        nx.set_node_attributes(bfs_graph, nodes_data)
        if as_dataframe:
            return utils.networkx_to_dataframe(bfs_graph)
        return bfs_graph

    def get_data(
            self,
            idx: T.Union[int, T.Iterable[int]],
            vals=None,
            plot=False,
            return_spec=True,
            msv_metadata=False
        ):
        if not isinstance(idx, T.Iterable):
            idx = [idx]

        data = {}
        for i in idx:
            if i < self.lib.num_spectra:
                data[i] = self.lib.at(
                    i, plot_mol=plot, plot_spec=plot, return_spec=return_spec
                )
            else:
                data[i] = self.gems.at(
                    i - self.lib.num_spectra, plot_mol=plot, plot_spec=plot, return_spec=return_spec
                )
    
            # NOTE: tmp fix for newly renamed datasets
            # TODO: assign columns with proper names and reuploaded the data to HF
            if 'dataset' in data[i].keys():
                del data[i]['dataset']

            if msv_metadata:
                data[i] = self._add_msv_metadata(data[i])

            if vals:
                data[i] = self._subset_data(data[i], vals)

        return data

    def is_library_i(self, i):
        return i < self.lib.num_spectra

    def _add_msv_metadata(self, data):
        if NAME in data.keys():
            # Split MassIVE ID and file name
            data['msv_id'] = data[NAME].split('_')[0]
            data[NAME] = '_'.join(data[NAME].split('_')[1:])

            # Get MassIVE metadata
            if data['msv_id'] not in self.msv_metadata.index:
                print(f'MassIVE ID {data["msv_id"]} not found in metadata. Most likely the dataset was made private.')
            else:
                metadata = self.msv_metadata.loc[data['msv_id']]
                data.update(metadata.to_dict())
        return data
    
    def _subset_data(self, data, vals):
        return {k: v for k, v in data.items() if k in vals}

    def _bfs(self, node, n_hops, inv_neighbors=False, sim_thld=-np.inf):

        # Encode full data index to knn index
        node = self.encode_knn_i(node)

        visited = set()
        queue = deque([(node, 0)])  # Queue stores node index and its depth

        bfs_graph = nx.DiGraph()
        bfs_graph.add_node(node)

        while queue:
            current_index, depth = queue.popleft()

            if depth >= n_hops:
                break

            visited.add(current_index)
            nns, sims = self.csrknn.neighbors(current_index)
            nns, sims = nns.tolist(), sims.tolist()

            if inv_neighbors:
                nns_inv, sims_inv = self.csrknn.inv_neighbors(current_index)
                nns_inv, sims_inv = nns_inv.tolist(), sims_inv.tolist()
                for i in range(len(nns_inv)):
                    if nns_inv[i] not in nns:
                        nns.append(nns_inv[i])
                        sims.append(sims_inv[i])

            for neighbor, similarity in zip(nns, sims):
                if similarity > sim_thld:
                    bfs_graph.add_edge(current_index, neighbor, weight=similarity)
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        
        # Decode knn index to full data index
        bfs_graph = nx.relabel_nodes(bfs_graph, {n: self.decode_knn_i(n) for n in bfs_graph.nodes()})

        return bfs_graph

    def encode_knn_i(self, node_repr_i):
        return self.repr_to_knn_i[node_repr_i]

    def decode_knn_i(self, knn_i):
        return self.knn_i_to_repr[knn_i]
    
    def get_lib_idx(self):
        # return np.array(range(self.lib.num_spectra))
        return np.where(self.lib[PRECURSOR_MZ] != -1)[0]  # -1 values are hidden NIST20 spectra
    
    def __len__(self):
        return self.lib.num_spectra + self.gems.num_spectra
