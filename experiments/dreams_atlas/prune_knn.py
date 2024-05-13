import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine as cos_dist
from pathlib import Path
import msml.utils.data as du
import msml.utils.io as io
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

    thld = 0.95

    knn_pth = Path('/storage/plzen1/home/romanb/DreaMS_Atlas/DreaMS_Atlas_3NN_before40000000.npz')
    # knn_pth = Path('/storage/plzen1/home/romanb/DreaMS_Atlas/DreaMS_Atlas_3NN_after40000000.npz')
    lib_pth = Path(MERGED_DATASETS / 'nist20_mona_clean_merged_spectra_dreams.hdf5')
    gems_pth = Path('/storage/plzen1/home/romanb/msvn_C/msvn_C_H1000_KK1.merged.hdf5')
    out_pth = io.append_to_stem(knn_pth.with_suffix('.pkl'), f'clusters_{str(thld).replace(".", "")}')
    logger = io.setup_logger(out_pth.with_suffix('.log'))

    logger.info(f'Loading k-NN from {knn_pth}.')
    knn = du.CSRKNN.from_npz(knn_pth)
    logger.info(f'Loaded {knn.n_nodes} nodes and {knn.n_edges} edges.')
    
    msd_lib = du.MSData(lib_pth)
    msd_gems = du.MSData(gems_pth)
    logger.info(f'Loaded {msd_lib.num_spectra} spectra from the library and {msd_gems.num_spectra} spectra from GeMS.')
    
    logger.info('Loading and concatenating embeddings.')
    embs = np.vstack([
        msd_lib.get_values(DREAMS_EMBEDDING),
        msd_gems.get_values(DREAMS_EMBEDDING, np.arange(knn.n_nodes - msd_lib.num_spectra))
        # msd_gems.get_values(DREAMS_EMBEDDING, np.arange(msd_gems.num_spectra - knn.n_nodes, msd_gems.num_spectra))
    ])
    logger.info(f'Concatenated embeddings have shape {embs.shape}.')
    
    logger.info('Creating igraph.')
    g = knn.to_igraph(directed=False)

    logger.info('Forming clusters.')
    clusters = form_clusters(g, thld, embs, logger)
    logger.info(f'Formed {len(clusters)} clusters from {g.vcount()} nodes.')

    clusters = list(reversed(sorted(clusters, key=len)))
    logger.info(f'Top 10 cluster sizes: {[len(c) for c in clusters][:10]}')

    logger.info(f'Saving clusters to {out_pth}.')
    io.write_pickle(clusters, out_pth)

    logger.info('Done.')

if __name__ == '__main__':
    main()
