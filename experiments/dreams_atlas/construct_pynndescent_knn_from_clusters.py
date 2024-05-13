import h5py
import numpy as np
import pynndescent
from pathlib import Path
from scipy.spatial.distance import cosine as cos_dist
from tqdm import tqdm
import msml.utils.io as io
from msml.utils.data import CSRKNN
from msml.definitions import *


def form_clusters(graph, thld, embs, logger):
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


def main():

    k = 3
    out_dir = Path('/storage/plzen1/home/romanb/DreaMS_Atlas')
    lib_pth = Path('/auto/brno2/home/romanb/msml/msml/data/merged/datasets/nist20_mona_clean_merged_spectra_dreams.hdf5')
    gems_pth = Path('/storage/plzen1/home/romanb/msvn_C/msvn_C_H1000_KK1.merged.hdf5')

    clusters1 = io.read_pickle(out_dir / f'DreaMS_Atlas_3NN_before40000000_clusters_09.pkl')
    clusters2 = io.read_pickle(out_dir / f'DreaMS_Atlas_3NN_after40000000_clusters_09.pkl')
    d = sum(len(c) for c in clusters1)  # Shift for clusters2: 40_000_000 from GeMS and the rest from spectral library
    idx = [c[0] for c in clusters1] + [c[0] + d for c in clusters2]
    idx = sorted(idx)

    out_pth = out_dir / f'DreaMS_Atlas_{k}NN_merged{d}_09.npz'
    logger = io.setup_logger(out_pth.with_suffix('.log'))

    # Load spectral library embeddings
    logger.info(f'Loading embeddings from {lib_pth}.')
    f = h5py.File(lib_pth, 'r')
    embs_lib = f[DREAMS_EMBEDDING][:]
    embs_lib = embs_lib.astype(np.float32)
    f.close()
    logger.info(f'Loaded {embs_lib.shape[0]} embeddings.')

    # Load GeMS embeddings
    logger.info(f'Loading embeddings from {gems_pth}.')
    f = h5py.File(gems_pth, 'r')
    embs_gems = f[DREAMS_EMBEDDING][:]
    embs_gems = embs_gems.astype(np.float32)
    f.close()
    logger.info(f'Loaded {embs_gems.shape[0]} embeddings.')

    # Concatenate embeddings
    logger.info('Concatenating embeddings.')
    all_embs = np.vstack([embs_lib, embs_gems])
    del embs_lib, embs_gems

    # Select embeddings of cluster representatives
    logger.info('Selecting embeddings of cluster representatives.')
    all_embs = all_embs[idx]

    # Create PyNNDescent index
    logger.info('Creating PyNNDescent index.')
    pynn_knn = pynndescent.PyNNDescentTransformer(
        metric='cosine', n_neighbors=k, search_epsilon=0.25, n_jobs=1, low_memory=True, verbose=True
    ).fit_transform(all_embs)

    logger.info('Initializing CSRKNN object.')
    knn = CSRKNN(pynn_knn)
    logger.info(f'Num. k-NN nodes: {knn.n_nodes}, num. edges: {knn.n_edges}.')

    logger.info(f'Saving k-NN to {out_pth}.')
    knn.to_npz(out_pth)
    io.write_pickle(idx, io.append_to_stem(out_pth, 'cluster_idx').with_suffix('.pkl'))

    logger.info('Pruning k-NN graph.')
    g = knn.to_igraph(directed=False)
    clusters = form_clusters(g, 0.90, all_embs, logger)
    io.write_pickle(clusters, io.append_to_stem(out_pth, 'clusters').with_suffix('.pkl'))
    # logger.info('Loading clusters.')
    # clusters = io.read_pickle(io.append_to_stem(out_pth, 'clusters').with_suffix('.pkl'))

    # Create final PyNNDescent index for merged and pruned graph
    logger.info('Selecting embeddings of cluster representatives.')
    idx = [c[0] for c in clusters]
    idx = sorted(idx)
    all_embs = all_embs[idx]
    logger.info('Creating final PyNNDescent index for merged and pruned graph.')
    pynn_knn = pynndescent.PyNNDescentTransformer(
        metric='cosine', n_neighbors=k, search_epsilon=0.25, n_jobs=1, low_memory=True, verbose=True
    ).fit_transform(all_embs) 

    logger.info('Initializing CSRKNN object.')
    knn = CSRKNN(pynn_knn)
    logger.info(f'Num. k-NN nodes: {knn.n_nodes}, num. edges: {knn.n_edges}.')

    out_pth = io.append_to_stem(out_pth, 'pruned')
    logger.info(f'Saving k-NN to {out_pth}.')
    knn.to_npz(out_pth)
    io.write_pickle(idx, io.append_to_stem(out_pth, 'cluster_idx').with_suffix('.pkl'))

    logger.info('Done.')


if __name__ == '__main__':
    main()
