import fire
import dreams.utils.io as io


if __name__ == '__main__':
    fire.Fire({
        'lcmsms_to_hdf5': io.lcmsms_to_hdf5,
        'downloadpublicdata_to_hdf5s': io.downloadpublicdata_to_hdf5s,
        'merge_lcmsms_hdf5s': io.merge_lcmsms_hdf5s
    })
