import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API"
)
import sys
import platform
import torch
import pandas as pd
import typing as T
import networkx as nx
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import deque
from torch.utils.data.dataloader import DataLoader
from argparse import Namespace
import dreams.utils.data as du
import dreams.utils.io as io
import dreams.utils.spectra as su
import dreams.utils.dformats as dformats
import dreams.utils.misc as utils
from dreams.models.dreams.dreams import DreaMS as DreaMSModel
from dreams.models.heads.heads import *
from dreams.definitions import *

try:
    import faiss
except ImportError:
    faiss = None


if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath


class PreTrainedModel:
    def __init__(self, model: T.Union[DreaMSModel, FineTuningHead], n_highest_peaks: int = 100):
        self.model = model.eval()
        self.n_highest_peaks = n_highest_peaks

    def remove_unused_backbone_parameters(model):
        """Helper function to remove unused heads from the pre-trained DreaMS backbone model."""
        if hasattr(model, 'ff_out'):
            delattr(model, 'ff_out')
        if hasattr(model, 'mz_masking_loss'):
            delattr(model, 'mz_masking_loss')
        if hasattr(model, 'ro_out'):
            delattr(model, 'ro_out')
        return model

    @classmethod
    def from_ckpt(
        cls,
        ckpt_path: Path,
        ckpt_cls: T.Union[T.Type[DreaMSModel], T.Type[FineTuningHead]],
        n_highest_peaks: int,
        remove_unused_backbone_parameters: bool = True,
        dreams_args: T.Optional[dict] = None
    ):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if ckpt_cls == DreaMSModel:

            ckpt = ckpt_cls.load_from_checkpoint(ckpt_path, map_location=device)

            # If DreaMS arguments are provided, reload the model with the updated arguments
            # (first load is needed to get the original arguments)
            if dreams_args is not None:
                args_dict = vars(ckpt.hparams["args"])
                args_dict.update(dreams_args)
                ckpt = ckpt_cls.load_from_checkpoint(ckpt_path, map_location=device, args=Namespace(**args_dict))

            model = cls(
                ckpt,
                n_highest_peaks=n_highest_peaks
            )

            if remove_unused_backbone_parameters:
                model.model = cls.remove_unused_backbone_parameters(model.model)
            
            return model
        else:

            if dreams_args is not None:
                raise NotImplementedError('Custom DreaMS arguments are currently not supported for fine-tuning heads')

            # Download backbone model if it doesn't exist
            backbone_pth = PRETRAINED / 'ssl_model.ckpt'
            if not backbone_pth.exists():
                utils.download_pretrained_model('ssl_model.ckpt')

            model = cls(
                ckpt_cls.load_from_checkpoint(
                    ckpt_path,
                    backbone_pth=backbone_pth,
                map_location=device
                ),
                n_highest_peaks=n_highest_peaks
            )
            if remove_unused_backbone_parameters:
                model.model.backbone = cls.remove_unused_backbone_parameters(model.model.backbone)
            return model

    @classmethod
    def from_name(cls, name: str):
        if name == DREAMS_EMBEDDING:
            ckpt_path = PRETRAINED / 'embedding_model.ckpt'
            ckpt_cls = ContrastiveHead
            n_highest_peaks = 100

            # Download model if it doesn't exist
            if not ckpt_path.exists():
                ckpt_path = utils.download_pretrained_model('embedding_model.ckpt')

        # elif name == 'Fluorine probability':
        #     ckpt_path = EXPERIMENTS_DIR / 'pre_training/HAS_F_1.0/CtDh6OHlhA/epoch=6-step=71500_v2_16bs_5e-5lr_gamma0.5_alpha0.8/epoch=30-step=111000.ckpt'
        #     ckpt_cls = BinClassificationHead
        #     n_highest_peaks = 100
        # elif name == 'Molecular properties':
        #     ckpt_path = EXPERIMENTS_DIR / f'pre_training/MS2PROP_1.0/lr3e-5_bs64/epoch=4-step=4000.ckpt'
        #     ckpt_cls = RegressionHead
        #     n_highest_peaks = 100
        else:
            raise ValueError(f'{name} is not a valid pre-trained model name. Choose from: {cls.available_models()}')

        return cls.from_ckpt(ckpt_path, ckpt_cls, n_highest_peaks)

    @staticmethod
    def available_models():
        return ['Fluorine probability', 'Molecular properties', DREAMS_EMBEDDING]


def dreams_predictions(
        model_ckpt: T.Union[PreTrainedModel, FineTuningHead, DreaMSModel, Path, str],
        spectra: T.Union[Path, str, du.MSData],
        model_cls=None,
        batch_size=32,
        progress_bar=True,
        n_highest_peaks=None,
        title='',
        logger_pth=None,
        store_preds=False,
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
    if not isinstance(spectra, du.MSData):
        if isinstance(spectra, str):
            spectra = Path(spectra)
        msdata = du.MSData.load(spectra, mode='a' if store_preds else 'r', **msdata_kwargs)
    else:
        msdata = spectra
        if msdata.mode != 'a' and store_preds:
            raise ValueError('Adding new columns is allowed only in append mode. Initialize msdata as '
                             '`MSData(..., mode="a")` to add new columns.')
    spectra = msdata.to_torch_dataset(spec_preproc)
    dataloader = DataLoader(spectra, batch_size=batch_size, shuffle=False, drop_last=False)

    # Setup logger writing progress to a file
    if logger_pth:
        logger = io.setup_logger(logger_pth)
        tqdm_logger = io.TqdmToLogger(logger)
    else:
        write_log = False

    # Compute predictions
    model = model_ckpt.model
    # TODO: consider model name
    if not title:
        title = 'DreaMS_prediction'

    # Preallocate memory for predictions
    num_samples = len(spectra)
    output_shape = model(next(iter(dataloader))[SPECTRUM].to(device=model.device, dtype=model.dtype)).shape[1:]
    preds = torch.zeros((num_samples, *output_shape), dtype=model.dtype)

    progress_bar = tqdm(
        total=num_samples,
        desc='Computing ' + title.replace('_', ' ') + 's',  # 's' for plural (NOTE: will not work with e.g. "probability")
        disable=not progress_bar,
        file=tqdm_logger if logger_pth else None
    )

    start_idx = 0
    for batch in dataloader:
        with torch.inference_mode():
            pred = model(batch[SPECTRUM].to(device=model.device, dtype=model.dtype))

            # Store predictions to cpu to avoid high memory allocation issues
            batch_size = pred.shape[0]
            preds[start_idx:start_idx + batch_size] = pred.cpu()
            start_idx += batch_size

            # Update the progress bar by the number of samples in the current batch
            progress_bar.update(batch_size)
    progress_bar.close()

    preds = preds.numpy()

    if store_preds:
        msdata.add_column(title, preds)

    return preds


def dreams_embeddings(pth, batch_size=32, progress_bar=True, logger_pth=None, store_embs=False, **msdata_kwargs):
    return dreams_predictions(
        DREAMS_EMBEDDING, pth, batch_size=batch_size, progress_bar=progress_bar, logger_pth=logger_pth,
        store_preds=store_embs, **msdata_kwargs
    )


def dreams_intermediates(
        model: T.Union[Path, str, PreTrainedModel],
        msdata: T.Union[Path, str],
        layers_idx=None,
        precursor_only=True,
        batch_size=32,
        progress_bar=True,
        spec_col=SPECTRUM,
        prec_mz_col=PRECURSOR_MZ,
        n_highest_peaks=60,
        compute_attn_matrices=True,
        compute_embeddings=False,
        spec_preproc: du.SpectrumPreprocessor=None
    ):
    """
    Extracts intermediate representations (embeddings and attention matrices) from individual layers of a DreaMS model.

    This function allows for the extraction of both embeddings and attention matrices from specified layers of a DreaMS model.
    It supports loading the model from a checkpoint and processing mass spectrometry data to obtain the desired intermediate
    representations. The function is flexible, allowing for customization of various parameters such as batch size, the number
    of highest peaks to consider, and whether to compute embeddings or attention matrices.

    Args:
        model (Union[Path, str, PreTrainedModel]): The model instance or the path to the model checkpoint file.
        msdata (Union[Path, str]): The mass spectrometry data or the path to the data file.
        layers_idx (list, optional): A list of layer indices from which to extract embeddings. If not provided, defaults to the last layer.
        precursor_only (bool, optional): If True, only extract embeddings for the precursor ion. Defaults to True.
        batch_size (int, optional): The number of samples to process in each batch. Defaults to 32.
        progress_bar (bool, optional): If True, display a progress bar during processing and print a log. Defaults to True.
        spec_col (str, optional): The column name in the data that contains the spectra. Defaults to SPECTRUM.
        prec_mz_col (str, optional): The column name in the data that contains the precursor m/z values. Defaults to PRECURSOR_MZ.
        n_highest_peaks (int, optional): The number of highest intensity peaks to consider in each spectrum. Defaults to 60.
        compute_attn_matrices (bool, optional): If True, compute and return attention matrices from the model. Defaults to True.
        compute_embeddings (bool, optional): If True, compute and return embeddings from the model. Defaults to False.
        spec_preproc (du.SpectrumPreprocessor, optional): An instance of SpectrumPreprocessor for preprocessing the spectra. Defaults to None.

    Returns:
        dict: A dictionary containing the extracted embeddings and/or attention matrices. The keys are the layer indices, and the values
              are the corresponding embeddings or attention matrices.
    """

    if not compute_attn_matrices and not compute_embeddings:
        raise ValueError('Either attention matrices or embeddings must be set to True.')

    # Load model if not already a PreTrainedModel instance
    if not isinstance(model, PreTrainedModel):
        model = PreTrainedModel.from_ckpt(model, DreaMSModel, n_highest_peaks)

    # Prepare data
    if not isinstance(msdata, du.MSData):
        msdata = du.MSData.load(msdata, spec_col=spec_col, prec_mz_col=prec_mz_col)

    # Initialize spectrum preprocessing
    spec_preproc = spec_preproc or du.SpectrumPreprocessor(
        dformat=dformats.DataFormatA(),
        n_highest_peaks=n_highest_peaks
    )

    # Prepare torch data loader
    msdata = msdata.to_torch_dataset(spec_preproc)
    dataloader = DataLoader(msdata, batch_size=batch_size, shuffle=False, drop_last=False)

    # Determine layers to extract embeddings from
    if not layers_idx:
        layers_idx = [model.model.n_layers - 1]

    # Preallocate memory for embeddings
    if compute_embeddings:
        embeddings = {
            i: torch.zeros((
                len(msdata),
                model.model.d_model
            ), device='cpu', dtype=model.model.dtype)
            for i in layers_idx
        }
    else:
        embeddings = None
    
    # Preallocate memory for attention matrices
    if compute_attn_matrices:
        attn_matrices = {
            i: torch.zeros((
                len(msdata),
                model.model.n_heads,
                msdata[0][SPECTRUM].shape[0],
                msdata[0][SPECTRUM].shape[0]
            ), device='cpu', dtype=model.model.dtype)
            for i in layers_idx
        }
    else:
        attn_matrices = None

    def get_embeddings_hook(layer_idx):
        def hook(module, input, output):
            if precursor_only:
                embs = output[:, 0, :]
            else:
                embs = output
            embs = embs.detach().cpu()
            start_idx = batch_start_idx
            end_idx = start_idx + embs.size(0)
            embeddings[layer_idx][start_idx:end_idx] = embs
        return hook

    def get_attn_scores_hook(layer_idx):
        def hook(module, input, output):
            attn = output[1]
            attn = attn.detach().cpu()
            start_idx = batch_start_idx
            end_idx = start_idx + attn.size(0)
            attn_matrices[layer_idx][start_idx:end_idx] = attn
        return hook

    # Register hooks
    hooks = []
    for i in layers_idx:
        if compute_embeddings:
            hooks.append(model.model.transformer_encoder.ffs[i].register_forward_hook(get_embeddings_hook(i)))
        if compute_attn_matrices:
            hooks.append(model.model.transformer_encoder.atts[i].register_forward_hook(get_attn_scores_hook(i)))

    # Perform forward passes
    progress_bar = tqdm(
        total=len(msdata),
        desc='Computing DreaMS intermediate representations',
        disable=not progress_bar
    )
    batch_start_idx = 0
    for batch in dataloader:
        with torch.inference_mode():
            model.model(batch[SPECTRUM].to(device=model.model.device, dtype=model.model.dtype))
        batch_start_idx += len(batch[SPECTRUM])
        progress_bar.update(len(batch[SPECTRUM]))
    progress_bar.close()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Convert to numpy
    if compute_embeddings:
        embeddings = {i: v.numpy() for i, v in embeddings.items()}
    if compute_attn_matrices:
        attn_matrices = {i: v.numpy() for i, v in attn_matrices.items()}

    # Simplify output if only one layer was requested
    if len(layers_idx) == 1:
        if compute_embeddings:
            embeddings = embeddings[layers_idx[0]]
        if compute_attn_matrices:
            attn_matrices = attn_matrices[layers_idx[0]]

    # Return embeddings and/or attention matrices
    if compute_embeddings and compute_attn_matrices:
        return embeddings, attn_matrices
    elif compute_embeddings:
        return embeddings
    return attn_matrices


def dreams_attn_scores(
        model: T.Union[Path, str, DreaMSModel],
        msdata: T.Union[Path, str],
        layers_idx=None,
        precursor_only=True,
        batch_size=32,
        progress_bar=True,
        spec_col=SPECTRUM,
        prec_mz_col=PRECURSOR_MZ,
        n_highest_peaks=None,
        spec_preproc: du.SpectrumPreprocessor = None
    ):
    return dreams_intermediates(
        model=model,
        msdata=msdata,
        layers_idx=layers_idx,
        precursor_only=precursor_only,
        batch_size=batch_size,
        progress_bar=progress_bar,
        spec_col=spec_col,
        prec_mz_col=prec_mz_col,
        n_highest_peaks=n_highest_peaks,
        attention_matrices=True,
        spec_preproc=spec_preproc
    )[1]


class DreaMSAtlas:
    def __init__(self, local_dir: T.Optional[T.Union[str, Path]] = None):
        """
        Initialize a DreaMSAtlas object enabling access to the DreaMS Atlas k-NN graph and associated data for
        individual nodes in the graph.

        Args:
            local_dir (Union[str, Path], optional): Local directory to download and cache data. Defaults to
            ~/.cache/huggingface/hub.
        """

        print('Initializing DreaMS Atlas data structures...')
        self.lib = du.MSData(
            local_dir / 'nist20_mona_clean_merged_spectra_dreams.hdf5',
            # utils.gems_hf_download(
            #     'DreaMS_Atlas/nist20_mona_clean_merged_spectra_dreams_hidden_nist20.hdf5',
            #     local_dir=local_dir
            # ),
            # in_mem=False
        )
        print(f'Loaded spectral library ({len(self.lib):,} spectra).')

        self.gems = du.MSData.from_hdf5_chunks(
            [utils.gems_hf_download(f'GeMS_C/GeMS_C1_DreaMS.{i}.hdf5', local_dir=local_dir) for i in range(10)],
            in_mem=False
        )
        print(f'Loaded GeMS-C1 dataset ({len(self.gems):,} spectra).')

        self.csrknn = du.CSRKNN.from_npz(
            # utils.gems_hf_download(f'DreaMS_Atlas/DreaMS_Atlas_3NN.npz', local_dir=local_dir),
            local_dir / 'DreaMS_Atlas_3NN_with_nist.npz'
        )
        print(f'Loaded DreaMS Atlas edges ({self.csrknn.n_nodes:,} nodes and {self.csrknn.n_edges:,} edges).')

        self.dreams_clusters = pd.read_csv(
            # utils.gems_hf_download(f'DreaMS_Atlas/DreaMS_Atlas_3NN_clusters.csv', local_dir=local_dir)
            local_dir / 'DreaMS_Atlas_3NN_clusters_with_nist.csv'
        )['clusters']
        print(f'Loaded DreaMS Atlas k-NN cluster representatives from GeMS-C1 ({self.dreams_clusters.nunique():,} representatives).')

        self.gems_lsh = du.MSData.from_hdf5_chunks(
            [utils.gems_hf_download(f'GeMS_C/GeMS_C.{i}.hdf5', local_dir=local_dir) for i in range(10)],
            in_mem=False
        )
        self.lsh_clusters = self.gems_lsh['lsh'][:]
        print(f'Loaded LSH clusters of DreaMS Atlas nodes representing GeMS-C ({len(self.lsh_clusters):,} spectra).')

        self.msv_metadata = pd.read_csv(
            utils.gems_hf_download('DreaMS_Atlas/massive_metadata.tsv', local_dir=local_dir), sep='\t', comment='#', low_memory=False
        )
        self.msv_metadata = self.msv_metadata.set_index('dataset')
        self.msv_metadata = self.msv_metadata[[
            'species', 'species_resolved', 'instrument', 'instrument_resolved', 'title', 'description', 'create_time',
            'user', 'keywords'
        ]]
        self.msv_metadata = self.msv_metadata.rename(columns={c: 'msv_' + c for c in self.msv_metadata.columns})

        # self.knn_i_to_repr = np.unique(self.dreams_clusters)  # When NIST20 is not hidden
        self.knn_i_to_repr = np.load(
            utils.gems_hf_download(f'DreaMS_Atlas/DreaMS_Atlas_3NN_knn_i_to_repr.npz', local_dir=local_dir)
        )['knn_i_to_repr']
        print(f'Loaded mapping from k-NN indices to node representatives (corresponding to {len(self.knn_i_to_repr):,} nodes).')
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
        if node_repr_i not in self.repr_to_knn_i.keys():
            raise ValueError(f'Node {node_repr_i} not found in the DreaMS Atlas k-NN graph. If you are using a standard '
                             'public version of the DreaMS Atlas, most likely the queried entry is associated with a '
                             'NIST20 spectrum which was masked in the public version of the Atlas due to the NIST20 '
                             'licensing restrictions.')
        return self.repr_to_knn_i[node_repr_i]

    def decode_knn_i(self, knn_i):
        return self.knn_i_to_repr[knn_i]
    
    def get_lib_idx(self):
        # return np.array(range(self.lib.num_spectra))  # When NIST20 is not hidden
        return np.where(self.lib[PRECURSOR_MZ] != -1)[0]  # -1 values are hidden NIST20 spectra
    
    def __len__(self):
        return self.lib.num_spectra + self.gems.num_spectra


class DreaMSSearch:
    def __init__(
        self,
        ref_spectra: T.Union[Path, str, du.MSData],
        verbose: bool = True,
        store_embs: bool = True
    ):
        if faiss is None:
            raise ImportError('Faiss is not installed. Please install it using `pip install faiss==1.9.0`.')

        self.verbose = verbose
        self.store_embs = store_embs

        # Fixes weird script hanging on macOS caused by the index.search method
        # if sys.platform.lower().startswith('darwin'):
        faiss.omp_set_num_threads(1)

        # Compute embeddings for reference spectra
        if not isinstance(ref_spectra, du.MSData):
            ref_spectra = du.MSData.load(ref_spectra, in_mem=True, mode='a' if self.store_embs else 'r')
        self.ref_spectra = ref_spectra
        if DREAMS_EMBEDDING in self.ref_spectra.columns():
            self.embs_ref = self.ref_spectra[DREAMS_EMBEDDING]
        else:
            self.embs_ref = dreams_embeddings(self.ref_spectra, store_embs=self.store_embs)
        self.embs_ref = self.embs_ref.astype('float32', copy=False)
        if self.embs_ref.ndim == 1:
            self.embs_ref = self.embs_ref[np.newaxis, :]

        # Build search index
        if self.verbose:
            print(f'Building search index (num spectra: {len(self.ref_spectra):,})...')
        faiss.normalize_L2(self.embs_ref)
        self.index = faiss.IndexFlatIP(self.embs_ref.shape[1])
        self.index.add(self.embs_ref)

    def query(
        self,
        query_spectra: T.Union[Path, str, du.MSData],
        out_path: T.Optional[Path] = None,
        k: int = 10,
        dreams_sim_thld: float = -np.inf,
        out_all_metadata: bool = True,
        out_spectra: bool = True,
        out_embs: bool = False
    ):
        if k > len(self.ref_spectra):
            raise ValueError(f'Requested more neighbors ({k})) than available in the reference spectral library '
                             f'(num spectra: {len(self.ref_spectra):,}).')

        if out_path is not None:
            if not isinstance(out_path, Path):
                out_path = Path(out_path)
            if out_path.suffix != '.tsv':
                raise ValueError(f'Output file {out_path} must have a .tsv extension.')

        # Compute embeddings for query spectra
        if not isinstance(query_spectra, du.MSData):
            query_spectra = du.MSData.load(query_spectra, in_mem=True, mode='a' if self.store_embs else 'r')
        if DREAMS_EMBEDDING in query_spectra.columns():
            embs = query_spectra[DREAMS_EMBEDDING]
        else:
            embs = dreams_embeddings(query_spectra, store_embs=self.store_embs)
        embs = embs.astype('float32', copy=False)
        if embs.ndim == 1:
            embs = embs[np.newaxis, :]
        faiss.normalize_L2(embs)

        # Search for top-k neighbors
        if self.verbose:
            print(f'Searching for top-{k} neighbors...')
        similarities, idx = self.index.search(embs, k=k)

        # Build DataFrame with results
        df = []
        for i in range(len(embs)):
            for k, j in enumerate(idx[i]):
                if similarities[i][k] > dreams_sim_thld:
                    row = {}

                    # Add main metadata columns for query and reference spectra
                    main_cols = [SCAN_NUMBER, RT, PRECURSOR_MZ]
                    for col in main_cols:
                        if col in query_spectra.columns():
                            row[f'{col}'] = query_spectra.get_values(col, i)
                        if col in self.ref_spectra.columns():
                            row[f'ref_{col}'] = self.ref_spectra.get_values(col, j)
                    
                    # Add all metadata columns for query and reference spectra
                    if out_all_metadata:
                        for col in query_spectra.columns():
                            if col not in main_cols + [SPECTRUM, DREAMS_EMBEDDING]:
                                row[f'{col}'] = query_spectra.get_values(col, i)
                        for col in self.ref_spectra.columns():
                            if col not in main_cols + [SPECTRUM, DREAMS_EMBEDDING]:
                                row[f'ref_{col}'] = self.ref_spectra.get_values(col, j)

                    # Add spectra columns for query and reference spectra
                    if out_spectra:
                        row[SPECTRUM] = query_spectra.get_spectra(i)
                        row[f'ref_{SPECTRUM}'] = self.ref_spectra.get_spectra(j)
                    
                    # Add DreaMS embeddings for query and reference spectra
                    if out_embs:
                        row[DREAMS_EMBEDDING] = embs[i].tolist()
                        row[f'ref_{DREAMS_EMBEDDING}'] = self.embs_ref[j].tolist()

                    # Add DreaMS similarity, top-k index, and query/reference index
                    row.update({
                        'index' : i,
                        'ref_index' : j,
                        'topk' : k + 1,
                        'DreaMS_similarity' : similarities[i][k],
                    })
                    df.append(row)
        
        # Return None if no neighbors found
        if len(df) == 0:
            if self.verbose:
                print('No neighbors found for the query spectra.')
            return None

        # Create DataFrame with results
        df = pd.DataFrame(df)
        df = df.sort_values('DreaMS_similarity', ascending=False)

        # Save results to file
        if out_path is not None:
            if out_spectra:
                df[SPECTRUM] = df[SPECTRUM].apply(lambda x: su.unpad_peak_list(x).tolist())
                df[f'ref_{SPECTRUM}'] = df[f'ref_{SPECTRUM}'].apply(lambda x: su.unpad_peak_list(x).tolist())
            df.to_csv(out_path, index=False, sep='\t')
            if self.verbose:
                print(f'Saved results to {out_path}')
        return df
