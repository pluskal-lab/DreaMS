import pathlib
import numpy as np
import heapq
import torch
import h5py
import typing as T
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import Counter
from typing import Sequence, Union
from huggingface_hub import hf_hub_download
from pathlib import Path
from dreams.definitions import PRETRAINED
from tqdm import tqdm


def hf_download(repo_id: str, file_pth: str, local_dir: T.Optional[T.Union[str, Path]] = None, repo_type: str = "dataset") -> str:
    """
    Download a file from the Hugging Face Hub and return its location on disk.
    
    Args:
        repo_id (str): Hugging Face repository ID.
        file_pth (str): Name of the file to download.
        local_dir (Optional[Union[str, Path]]): Local directory to download the file to.
        repo_type (str): Type of the repository.
    """
    return hf_hub_download(
        repo_id=repo_id,
        filename=file_pth,
        repo_type=repo_type,
        local_dir=local_dir,
        cache_dir=local_dir
    )


def download_pretrained_model(model_name: str = 'embedding_model.ckpt', download_dir: Path = PRETRAINED, verbose: bool = True):
    """
    Download a pre-trained model from the Hugging Face Hub and return its location on disk.
    
    Args:
        model_name (str): Name of the model to download.
        download_dir (Path): Local directory to download the model to.
        verbose (bool): Whether to print verbose output.
    """
    # Old method of downloading from Zenodo
    # target_path = download_dir / model_name
    # url = 'https://zenodo.org/records/10997887/files/' + model_name
    
    # # Create the download directory if it doesn't exist
    # target_path.parent.mkdir(parents=True, exist_ok=True)

    # def download_with_progress(url, target_path):
    #     with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading {url.split('/')[-1]}") as pbar:
    #         def report_hook(count, block_size, total_size):
    #             if total_size != -1:
    #                 pbar.total = total_size
    #             pbar.update(block_size)
    #         urllib.request.urlretrieve(url, target_path, reporthook=report_hook)
            
    # download_with_progress(url, target_path)
    # return target_path

    # New method of downloading from Hugging Face
    if verbose:
        print(f"Downloading {model_name} from Hugging Face to {download_dir}/{model_name}")
    return hf_download(
        repo_id="roman-bushuiev/DreaMS",
        file_pth=model_name,
        local_dir=download_dir,
        repo_type="model"
    )


def gems_hf_download(file_pth: str, local_dir: T.Optional[T.Union[str, Path]] = None) -> str:
    """
    Download a GeMS file from the Hugging Face Hub and return its location on disk.
    
    Args:
        file_pth (str): Name of the file to download.
        local_dir (Optional[Union[str, Path]]): Local directory to download the file to.
    """
    return hf_download(
        repo_id="roman-bushuiev/GeMS",
        file_pth="data/" + file_pth,
        local_dir=local_dir,
        repo_type="dataset"
    )


def networkx_to_dataframe(G: nx.Graph) -> pd.DataFrame:
    # Initialize a list to store the data for each node
    data = []
    
    # Gather all node and edge attribute keys
    node_attr_keys = set()
    edge_attr_keys = set()
    
    for node, attrs in G.nodes(data=True):
        node_attr_keys.update(attrs.keys())
        
    for u, v, attrs in G.edges(data=True):
        edge_attr_keys.update(attrs.keys())

    # Iterate over all nodes in the graph
    for node in G.nodes(data=True):
        node_id = node[0]
        node_attrs = node[1]  # Node attributes
        
        # Initialize node data with default values (None for missing attributes)
        node_data = {'node_id': node_id}
        for key in node_attr_keys:
            node_data[key] = node_attrs.get(key, None)
        
        # Get neighbors
        neighbors = list(G.neighbors(node_id))
        node_data['neighbors'] = neighbors
        
        # Get edge attributes for each neighbor
        for key in edge_attr_keys:
            node_data[f'edge_{key}'] = [
                G.get_edge_data(node_id, neighbor).get(key, None)
                for neighbor in neighbors
            ]
        
        # Append the dictionary to the data list
        data.append(node_data)
    
    # Create a DataFrame from the data list
    df = pd.DataFrame(data)
    
    return df


def _ensure_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert numpy array to torch tensor; pass through tensors unchanged."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    if isinstance(x, torch.Tensor):
        return x
    raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")


def knn_search(
    query: Union[torch.Tensor, np.ndarray, "h5py.Dataset", "dreams.utils.io.ChunkedDatasetAccessor"],
    ref: Union[torch.Tensor, np.ndarray, "h5py.Dataset", "dreams.utils.io.ChunkedDatasetAccessor"],
    topk: int = 1,
    query_batch_size: int = 1024,
    ref_batch_size: int = 10000,
    verbose: bool = False,
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform exact cosine similarity search on CPU or GPU using batched matrix multiplication.
    Handles large reference databases that may not fit in memory by processing
    reference in chunks. Supports in-memory arrays and on-disk HDF5 datasets for
    both query and reference. Caller is responsible for opening HDF5 files and
    passing the dataset, e.g. h5py.File(path, 'r')['DreaMS_embedding'].

    Args:
        query: Query embeddings of shape (n_query, d). torch.Tensor, np.ndarray,
            or h5py.Dataset or ChunkedDatasetAccessor (on-disk, loaded in batches).
        ref: Reference embeddings of shape (n_ref, d). torch.Tensor, np.ndarray,
            or h5py.Dataset or ChunkedDatasetAccessor (on-disk, only necessary chunks are loaded).
        topk: Number of top results to return for each query.
        query_batch_size: Batch size for processing queries.
        ref_batch_size: Batch size for processing reference chunks (to avoid memory OOM).
        verbose: If True, show a tqdm progress bar over query batches.

    Returns:
        Tuple of (indices, similarities):
        - indices: Tensor of shape (n_query, topk) with topk most similar reference indices.
        - similarities: Tensor of shape (n_query, topk) with cosine similarity scores.
    """
    from dreams.utils.io import ChunkedDatasetAccessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve query: in-memory array vs on-disk h5py.Dataset
    query_tensor: T.Optional[torch.Tensor] = None
    query_ds = None
    if h5py is not None and (isinstance(query, h5py.Dataset) or isinstance(query, ChunkedDatasetAccessor)):
        query_ds = query
        n_query = query_ds.shape[0]
    else:
        query_tensor = _ensure_tensor(query).to(device)
        query_tensor = torch.nn.functional.normalize(query_tensor, dim=1)
        n_query = query_tensor.shape[0]

    # Resolve reference: in-memory array vs on-disk h5py.Dataset
    ref_cpu: T.Optional[torch.Tensor] = None
    ref_ds = None
    if h5py is not None and (isinstance(ref, h5py.Dataset) or isinstance(ref, ChunkedDatasetAccessor)):
        ref_ds = ref
        n_ref = ref_ds.shape[0]
    else:
        ref_cpu = _ensure_tensor(ref).cpu()
        ref_cpu = torch.nn.functional.normalize(ref_cpu, dim=1)
        n_ref = ref_cpu.shape[0]

    # Process queries in batches
    query_batch_size = min(query_batch_size, n_query)
    all_topk_idx = []
    all_topk_scores = []
    query_batch_starts = range(0, n_query, query_batch_size)
    if verbose:
        query_batch_starts = tqdm(query_batch_starts, desc="k-NN query", unit="batch")
    for i in query_batch_starts:
        # Load query batch (from tensor or HDF5 slice)
        end = min(i + query_batch_size, n_query)
        if query_tensor is not None:
            qbatch = query_tensor[i:end]
        else:
            chunk_np = np.asarray(query_ds[i:end])
            qbatch = torch.from_numpy(chunk_np).float().to(device)
            qbatch = torch.nn.functional.normalize(qbatch, dim=1)

        # Initialize per-query top-k accumulators for this batch
        topk_scores = torch.full(
            (qbatch.shape[0], topk),
            float("-inf"),
            device=device,
            dtype=qbatch.dtype,
        )
        topk_idx = torch.zeros((qbatch.shape[0], topk), device=device, dtype=torch.long)

        # Scan reference in chunks and merge top-k
        ref_chunk_starts = range(0, n_ref, ref_batch_size)
        if verbose:
            ref_chunk_starts = tqdm(ref_chunk_starts, desc="Searching reference chunk", unit="chunk", leave=False)
        for ref_start in ref_chunk_starts:
            ref_end = min(ref_start + ref_batch_size, n_ref)
            # Load reference chunk (from tensor or HDF5 slice)
            if ref_cpu is not None:
                ref_chunk = ref_cpu[ref_start:ref_end].to(device)
            else:
                chunk_np = np.asarray(ref_ds[ref_start:ref_end])
                ref_chunk = torch.from_numpy(chunk_np).float().to(device)
                ref_chunk = torch.nn.functional.normalize(ref_chunk, dim=1)

            # Cosine similarity (vectors already normalized) and merge with running top-k
            sim = torch.mm(qbatch, ref_chunk.t())
            chunk_topk = min(topk, ref_chunk.shape[0])
            chunk_scores, chunk_idx = torch.topk(sim, k=chunk_topk, dim=1)
            chunk_idx = chunk_idx + ref_start
            combined_scores = torch.cat([topk_scores, chunk_scores], dim=1)
            combined_idx = torch.cat([topk_idx, chunk_idx], dim=1)
            topk_scores, topk_indices_in_combined = torch.topk(combined_scores, k=topk, dim=1)
            topk_idx = torch.gather(combined_idx, 1, topk_indices_in_combined)

        all_topk_idx.append(topk_idx)
        all_topk_scores.append(topk_scores)

    indices = torch.cat(all_topk_idx, dim=0)
    similarities = torch.cat(all_topk_scores, dim=0)
    return indices, similarities


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def lists_to_legends(lists):
    """
    E.g. [['a', 'b', 'c'], [1, 2, 3]] -> ['a | 1', 'b | 2', 'c | 3'].
    """
    def process_element(element):
        if isinstance(element, float):
            return f'{element:.2f}'
        else:
            return str(element)
    return [' | '.join([process_element(l) for l in legend]) for legend in list(zip(*lists))]


def get_closest_values(lst, query_val, n=1, return_idx=False):
    lst = np.array(lst)
    closest_vals_idx = np.abs(lst - query_val).argsort()[:n]
    if return_idx:
        return closest_vals_idx
    else:
        return lst[closest_vals_idx]


def contains_similar(lst, query_val, epsilon, return_idx=False):
    lst = np.array(lst)
    similar_idx = np.argwhere(np.abs(lst - query_val) < epsilon)
    res = False if similar_idx.size == 0 else True
    return (res, similar_idx) if return_idx else res


def all_close_pairwise(numbers, eps=1e-2):
    # https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
    # For finite values, isclose uses the following equation to test whether two floating point values are equivalent.
    #   absolute(a - b) <= (atol + rtol * absolute(b))
    return np.all(np.isclose(numbers, numbers[0], rtol=0, atol=eps))


def is_sorted(lst):
    return all(lst[i] <= lst[i+1] for i in range(len(lst) - 1))


def chunk_list(lst, chunks_n):
    return (lst[i::chunks_n] for i in range(chunks_n))


def chunk_list_eq_sum(lst, chunks_n, val=lambda e: e):
    """
    Partitions list lst to n bins with approximately equal sums. If elements of lst are not
    numbers, val function can be specified in order to extract desired numbers from each
    element.
    """

    # Initialize heap of bins. Bin - [sum of bin, elements].
    bins = [[0, []] for _ in range(chunks_n)]
    heapq.heapify(bins)  # NOTE: heapify builds heap wrt to first elements of bins

    # Traverse over sorted (desc) input list.
    lst = sorted(lst, reverse=True, key=val)
    for e in lst:

        # Append each element to a bin with the minimal sum.
        min_bin = bins[0]
        min_bin[0] += val(e)
        min_bin[1].append(e)
        heapq.heapreplace(bins, min_bin)

    # Return elements of bins.
    return [b[1] for b in bins]


def interpolate_interval(a, b, n, only_inter=False, rounded=False):
    """
    :param a: Start point.
    :param b: End point.
    :param n: Num. of steps.
    :param only_inter: Does not return interval ends a and b.
    :param rounded:
    """
    x_min, x_max = min(a, b), max(a, b)
    res = [x_min + i * (x_max - x_min) / (n + 1) for i in range(1 if only_inter else 0, n + 1 if only_inter else n + 2)]
    if x_max == a:
        res.reverse()
    if rounded:
        res = [round(x) for x in res]
    return res


def complete_permutation(arr: np.array):
    """
    Returns a copy of arr with shuffled elements such that each element has a different position
    :param arr: 1D NumPy array
    """
    a = arr.copy()
    for i in range(0, len(a) - 1):
        i_new = np.random.randint(i + 1, len(a))
        a[i], a[i_new] = a[i_new], a[i]
    return a


def calc_attention_entropy(attention_scores, as_plot=True, save_out_basename: pathlib.Path = None):
    """
    :param attention_scores: dict with [0, num_layers] keys and tensor (batch_size, num_heads, seq_len, seq_len) values.
    """

    df = []
    for layer_i in range(len(attention_scores)):
        for head_i in range(attention_scores[layer_i].shape[1]):
            attention = attention_scores[layer_i][:, head_i, :, :]
            df.append({
                'Layer': layer_i,
                'Head': head_i,
                'H': torch.distributions.Categorical(attention, validate_args=False).entropy().mean().item()
            })
    df = pd.DataFrame(df)

    if save_out_basename:
        df.to_csv(save_out_basename.with_suffix('.csv'))

    if as_plot:
        fig = go.Figure(data=go.Scatter(x=df['Layer'], y=df['H'], mode='markers', marker_color=df['Head'],
                                        text=df['Head'], marker=dict(size=7)))
        fig.update_layout(autosize=False, width=400, height=200, margin=dict(l=10, r=10, b=10, t=10, pad=4),
                          template='plotly_white', xaxis_title='Layer', yaxis_title='H')
        return fig
    return df


def merge_stats(stats: Sequence[dict], sets_len=False):

    # List of dicts to dict of lists
    stats = {k: [d[k] for d in stats] for k in stats[0].keys()}

    # Merge each statistic withing list
    for k in stats.keys():
        if isinstance(stats[k][0], int):
            stats[k] = sum(stats[k])
        elif isinstance(stats[k][0], Counter):
            stats[k] = dict(sum(stats[k], Counter()))
        elif isinstance(stats[k][0], set):
            stats[k] = set().union(*stats[k])
            if sets_len:
                stats[k] = len(stats[k])
        else:
            raise NotImplementedError

    return stats
