from pathlib import Path
import msml.utils.data as du
import msml.utils.io as io


def main():
    atlas_dir = Path('/storage/plzen1/home/romanb/DreaMS_Atlas')
    logger = io.setup_logger(atlas_dir / 'diameter.log')
    knn = du.CSRKNN.from_npz(atlas_dir / 'DreaMS_Atlas_3NN_merged40079300_09_pruned.npz')
    logger.info(f'Loaded k-NN with {knn.n_nodes} nodes and {knn.n_edges} edges.')
    g = knn.to_igraph(directed=False)
    logger.info('Constructed igraph graph.')
    diameter = g.diameter()
    logger.info(f'Graph diameter: {diameter}.')
    logger.info('Done.')


if __name__ == '__main__':
    main()
