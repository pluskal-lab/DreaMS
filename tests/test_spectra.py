import numpy as np
import dreams.utils.spectra as su


def test_get_peak_intens_nbhd():
    pl = np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [300, 500, 200, 400, 800, 600, 700]
    ])
    assert su.get_peak_intens_nbhd(pl, 4, 100, intens_thld_below=False) == (0, 6)
    assert su.get_peak_intens_nbhd(pl, 0, 100, intens_thld_below=False) == (0, 6)
    assert su.get_peak_intens_nbhd(pl, 6, 100, intens_thld_below=False) == (0, 6)
    assert su.get_peak_intens_nbhd(pl, 4, 750, intens_thld_below=False) == (4, 4)
    assert su.get_peak_intens_nbhd(pl, 0, 250, intens_thld_below=False) == (0, 1)
    assert su.get_peak_intens_nbhd(pl, 0, 350) == (0, 0)
    assert su.get_peak_intens_nbhd(pl, 6, 750) == (5, 6)
    pl = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 10, 1, 10, 10, 10, 1, 10]
    ])
    assert su.get_peak_intens_nbhd(pl, 1, 5, intens_thld_below=False) == (1, 1)
    assert su.get_peak_intens_nbhd(pl, 3, 5, intens_thld_below=False) == (3, 5)
    assert su.get_peak_intens_nbhd(pl, 4, 5, intens_thld_below=False) == (3, 5)
    assert su.get_peak_intens_nbhd(pl, 5, 5, intens_thld_below=False) == (3, 5)
    assert su.get_peak_intens_nbhd(pl, 7, 5, intens_thld_below=False) == (7, 7)


def test_trimming():

    # Single spectrum case
    pl = np.array([
        [1, 10, 20, 30, 40, 50, 60],
        [100, 200, 300, 700, 500, 600, 400]
    ])
    res = su.trim_peak_list(pl, n_highest=2)
    res_ref = np.array([
        [30, 50],
        [700, 600]
    ])
    assert np.array_equal(res, res_ref)

    # Batched case
    pl = np.array([
        [[1, 10, 20], [100, 200, 300]],
        [[1, 10, 20], [300, 100, 200]],
        [[1, 10, 20], [50, 400, 150]]
    ])
    res = su.trim_peak_list(pl, n_highest=2)
    res_ref = np.array([
        [[10, 20], [200, 300]],
        [[1, 20], [300, 200]],
        [[10, 20], [400, 150]]
    ])
    assert np.array_equal(res, res_ref)


def test_padding():

    # Single spectrum case
    pl = np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [100, 200, 300, 400, 500, 600, 700]
    ])
    res = su.pad_peak_list(pl, 10)
    res_ref = np.array([
        [1, 2, 3, 4, 5, 6, 7, 0, 0, 0],
        [100, 200, 300, 400, 500, 600, 700, 0, 0, 0]
    ])
    assert np.array_equal(res, res_ref)

    # Batched case
    pl = np.array([
        [[1, 10, 20], [100, 200, 300]],
        [[1, 10, 20], [300, 100, 200]],
        [[1, 10, 20], [50, 400, 150]]
    ])
    res = su.pad_peak_list(pl, 5)
    res_ref = np.array([
        [[1, 10, 20, 0, 0], [100, 200, 300, 0, 0]],
        [[1, 10, 20, 0, 0], [300, 100, 200, 0, 0]],
        [[1, 10, 20, 0, 0], [50, 400, 150, 0, 0]]
    ])
    assert np.array_equal(res, res_ref)
