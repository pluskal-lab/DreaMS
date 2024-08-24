import pathlib
import numpy as np
import heapq
import torch
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
from typing import Sequence
from huggingface_hub import hf_hub_download


def gems_hf_download(file_pth: str) -> str:
    """
    Download a file from the Hugging Face Hub and return its location on disk.
    
    Args:
        file_pth (str): Name of the file to download.
    """
    return hf_hub_download(
        repo_id="roman-bushuiev/GeMS",
        filename="data/" + file_pth,
        repo_type="dataset",
    )


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
