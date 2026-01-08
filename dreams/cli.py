import fire
import pandas as pd
from dreams.api import DreaMSSearch
from dreams.utils.io import (
    lcmsms_to_hdf5, downloadpublicdata_to_hdf5s, merge_lcmsms_hdf5s, compress_hdf,
)
from dreams.utils.data import subset_lsh


def serialize(x):
    if isinstance(x, pd.DataFrame):
        return x.head().to_string(index=False)
    return x


if __name__ == '__main__':
    fire.Fire({
        'lcmsms_to_hdf5': lcmsms_to_hdf5,
        'downloadpublicdata_to_hdf5s': downloadpublicdata_to_hdf5s,
        'merge_lcmsms_hdf5s': merge_lcmsms_hdf5s,
        'compress_hdf5': compress_hdf,
        'subset_lsh': subset_lsh,
        'dreams_search': DreaMSSearch,
    }, serialize=serialize)
