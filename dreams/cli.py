import fire
import pandas as pd
from dreams.api import DreaMSSearch, predict_fluorine
from dreams.utils.io import (
    lcmsms_to_hdf5, downloadpublicdata_to_hdf5s, merge_lcmsms_hdf5s, compress_hdf,
)
from dreams.utils.data import subset_lsh


def serialize(x):
    if isinstance(x, pd.DataFrame):
        # Show a nicely-truncated DataFrame preview in the terminal
        with pd.option_context(
            'display.max_rows', 5,
            'display.max_columns', 10,
            'display.width', 120,
            'display.max_colwidth', 30,
            'display.expand_frame_repr', False
        ):
            return x.__repr__()
    return x


if __name__ == '__main__':
    fire.Fire({
        'lcmsms_to_hdf5': lcmsms_to_hdf5,
        'downloadpublicdata_to_hdf5s': downloadpublicdata_to_hdf5s,
        'merge_lcmsms_hdf5s': merge_lcmsms_hdf5s,
        'compress_hdf5': compress_hdf,
        'subset_lsh': subset_lsh,
        'dreams_search': DreaMSSearch,
        'dreams_fluorine': predict_fluorine,
    }, serialize=serialize)
