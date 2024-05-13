import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import igraph
import h5py
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()
from rdkit import Chem
from pandarallel import pandarallel
import msml.utils.data as du
import msml.utils.spectra as su
import msml.utils.mols as mu
import msml.utils.io as io
import msml.utils.plots as plots
from msml.definitions import *


def main():

    k = 100_000
    seed = 1

    atlas_dir = Path('/storage/plzen1/home/romanb/DreaMS_Atlas')
    gems_dir = Path('/storage/plzen1/home/romanb/msvn_C/')
    out_pth = atlas_dir / f'knn_lib_dists_k={k}_seed={seed}.npz'

    logger = io.setup_logger(out_pth.with_suffix('.log'))
    tqdm_logger = io.TqdmToLogger(logger)

    logger.info('Loading data.')
    lib = du.MSData('/auto/brno2/home/romanb/msml/msml/data/merged/datasets/nist20_mona_clean_merged_spectra_dreams.hdf5')
    gems = du.MSData('/storage/plzen1/home/romanb/msvn_C/msvn_C_H1000_KK1.merged.hdf5')
    def get_item(i, val=PRECURSOR_MZ):
        if i <= lib.num_spectra:
            return lib.get_values(val, i)
        else:
            res = gems.get_values(val, i - lib.num_spectra)
            if val == SPECTRUM:
                res = res.T
            return res

    knn = du.CSRKNN.from_npz(atlas_dir / 'DreaMS_Atlas_3NN_merged40079300_09_pruned.npz')
    df_clusters = pd.read_csv(atlas_dir / 'DreaMS_Atlas_3NN_merged40079300_09_pruned_clusters.csv')
    cluster_idx_1 = np.array(io.read_pickle(atlas_dir / 'DreaMS_Atlas_3NN_merged40079300_09_cluster_idx.pkl'))
    cluster_idx_2 = np.array(io.read_pickle(atlas_dir / 'DreaMS_Atlas_3NN_merged40079300_09_pruned_cluster_idx.pkl'))
    def encode_knn_i(i):
        e1 = np.where(np.array(cluster_idx_1) == i)[0].item()
        e2 = np.where(np.array(cluster_idx_2) == e1)[0].item()
        return e2
    def decode_knn_i(i):
        return cluster_idx_1[cluster_idx_2[i]]
    
    logger.info('Constructing igraph graph.')
    g = knn.to_igraph(directed=False)
    components = g.components()
    component1 = components[0]

    logger.info('Computing lib and random nodes.')
    pandarallel.initialize(nb_workers=16, progress_bar=True, use_memory_fs=False)
    lib_nodes = set(df_clusters['clusters1_and_clusters2'].iloc[:lib.num_spectra].parallel_apply(encode_knn_i).unique())
    lib_nodes = lib_nodes.intersection(set(component1))
    random.seed(seed)
    rand_nodes = set(random.sample(range(g.vcount()), k=len(lib_nodes)))

    def bfs_to_nodes_subset(graph, start_vertex, subset):
        visited = set()
        queue = [(start_vertex, 0)]  # (vertex, distance)
        
        while queue:
            vertex, distance = queue.pop(0)
            visited.add(vertex.index)
            
            # if vertex.index < stop_index:
            #     return distance
            
            if decode_knn_i(vertex.index) in subset:
                return distance, vertex.index

            for neighbor in vertex.neighbors():
                if neighbor.index not in visited:
                    queue.append((neighbor, distance + 1))
        
        return -1, -1  # If the stop index is not found within the BFS traversal

    lib_dists, rand_dists = [], []
    for i in tqdm(random.sample(component1, k=k), desc='Computing distances', file=tqdm_logger):
        lib_dists.append(bfs_to_nodes_subset(g, g.vs[i], lib_nodes)[0])
        rand_dists.append(bfs_to_nodes_subset(g, g.vs[i], rand_nodes)[0])
    lib_dists = np.array(lib_dists)
    rand_dists = np.array(rand_dists)
    
    logger.info('Saving results.')
    np.savez(out_pth, lib_dists=lib_dists, rand_dists=rand_dists)

    logger.info('Plotting.')
    plots.init_plotting(figsize=(2, 2))
    palette = [plots.get_nature_hex_colors()[i] for i in [1, 2]]
    df_res = pd.DataFrame({
        'Distance (num. edges)': lib_dists.tolist() + rand_dists.tolist(),
        'Type': ['To MoNA or NIST20'] * len(lib_dists) + ['To random set of nodes'] * len(rand_dists)
    })
    sns.boxplot(
        data=df_res, y='Distance (num. edges)', hue='Type', x='Type', palette=palette,
        flierprops={'markersize': 3, 'markeredgewidth': 0.2}
    )
    plt.yticks([0, 5, 6, 10, 15])
    plots.save_fig('atlas_paths_boxplots.svg', dir=atlas_dir)

    logger.info('Done.')


if __name__ == '__main__':
    main()
