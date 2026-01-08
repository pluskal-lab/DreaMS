import pandas as pd
from enum import Enum, auto, unique
import statistics as stats
import numpy as np
import contextlib
import io as std_io
with contextlib.redirect_stderr(std_io.StringIO()):
    import pyopenms as pyms
from collections import Counter
import dreams.utils.spectra as su
import dreams.utils.misc as utils


@unique
class SpecType(Enum):
    """
    Enum representing the type of spectrum (centroid, profile and other corner cases).
    """

    CENTROID = auto()
    PROFILE = auto()
    THRESHOLDED = auto()
    UNKNOWN = auto()
    SIZE_OF_SPECTRUMTYPE = auto()  # from pyopenms (https://github.com/OpenMS/OpenMS/blob/develop/src/pyOpenMS/pxds/SpectrumSettings.pxd#L56)
    INVALID = auto()


@unique
class MSLevelsOrder(Enum):
    """
    Enum representing order of spectra in MS file.

    Further l_i denotes MS level of i-th spectrum.
    """

    # No spectra
    EMPTY = auto()

    # File contains only one spectrum
    # e.g. [1]
    SINGLE_MS1 = auto()
    # e.g. [2]
    SINGLE_MSN = auto()

    # For all i: l_i = l_(i-1)
    # e.g. [1, 1, 1, 1]
    UNIFORM_MS1 = auto()
    # e.g. [2, 2, 2]
    UNIFORM_MSN = auto()

    # For all i: l_i = l_(i-1) or
    #            l_i > l_(i-1) and l_i - l_(i-1) = 1 or
    #            l_i < l_(i-1) and l_i = 1
    # and not UNIFORM_MS1 and not UNIFORM_MSN
    # e.g. [1, 2, 2, 3, 3, 1, 2]
    CONSEQUENT_MSN = auto()

    # For all i: l_i = l_(i-1) or
    #            l_i > l_(i-1) and l_i - l_(i-1) = 1 or
    #            l_i < l_(i-1) and l_(i-1) - l_1 = 1
    # and not UNIFORM_MS1 and not UNIFORM_MSN
    # e.g. [1, 2, 3, 2, 3, 1, 2]
    MIXED_MSN = auto()

    # Invalid
    # e.g. [-1, 2, 3]
    # e.g. [1, 3]
    INVALID = auto()

    # Other
    #OTHER = auto()


def get_order_of_spectra(msdata) -> MSLevelsOrder:

    # No spectra
    if not msdata.getSpectra():
        return MSLevelsOrder.EMPTY

    ms_levels = [spectrum.getMSLevel() for spectrum in msdata]

    # Check that all MS levels are positive integers
    for ms_level in ms_levels:
        if not isinstance(ms_level, int) or ms_level < 1:
            return MSLevelsOrder.INVALID

    if len(ms_levels) == 1:
        if ms_levels[0] == 1:
            return MSLevelsOrder.SINGLE_MS1
        else:
            return MSLevelsOrder.SINGLE_MSN

    # Go over all pairs of subsequent MS levels and classify
    # their difference to MSLevelOrder's
    pairwise_orders = set()
    for level1, level2 in zip(ms_levels[:-1], ms_levels[1:]):

        if level1 == level2:
            if level1 == 1:
                pairwise_orders.add(MSLevelsOrder.UNIFORM_MS1)
            else:
                pairwise_orders.add(MSLevelsOrder.UNIFORM_MSN)
        elif level2 < level1:
            if level2 == 1:
                pairwise_orders.add(MSLevelsOrder.CONSEQUENT_MSN)
            else:
                pairwise_orders.add(MSLevelsOrder.MIXED_MSN)
        else:  # level2 > level1:
            if level2 - level1 == 1:
                pairwise_orders.add(MSLevelsOrder.CONSEQUENT_MSN)
            else:
                return MSLevelsOrder.INVALID

    if MSLevelsOrder.UNIFORM_MS1 in pairwise_orders and len(pairwise_orders) == 1:
        return MSLevelsOrder.UNIFORM_MS1
    elif MSLevelsOrder.UNIFORM_MSN in pairwise_orders and len(pairwise_orders) == 1:
        return MSLevelsOrder.UNIFORM_MSN
    #elif ms_levels[0] != 1:  # NOTE: possible [2, 2, 3, 3] is ok
    #    return MSLevelsOrder.INVALID
    elif MSLevelsOrder.MIXED_MSN in pairwise_orders:
        return MSLevelsOrder.MIXED_MSN
    else:
        return MSLevelsOrder.CONSEQUENT_MSN


def get_tight_xics(msdata, mz_tol_1=0.5, mz_tol_2=0.01, intensity_rel_tol=0.1, xic_len_thld=5, n_highest_peaks=3):
    """
    Tight XIC at given m/z is a cut of ms data accross rt dimension, containing highest peak and all peaks in its
    neighbourhood wrt rt. Length of the neighbourhood is defined independently in each direction by m/z and intensity
    tolerance parameters. When algorigtmm builds XIC it starts from some particular peak (xic_mz, xic_in) and consequently
    examines peaks in its neighbourhood peak by peak. Suppose (prev_mz, prev_in) and (next_mz, next_in) are two peaks compared
    during the run, where (prev_mz, prev_in) is a current border of the neighbourhood, then the neighbourhood will be extended on
    (next_mz, next_in) only if it satisfies two conditions:
        1) abs(next_mz - xic_mz) <= m/z tolerance
        2) next_in <= prev_in * intensity tolerance

    Algorithm performs 2 traversals accross all MS1 spectra:
        I. Builds tight XICs for m/z's of n_highest_peaks highest peaks of each spectrum, where m/z
            tolerance windown is "wide" (mz_tol_1).
        II. Computes medians of m/z's accross XICs obtained in step I., which are used to build new tight
            XICs with "smaller" m/z tolerance window (mz_tol_2).

    :param msdata: ms data to boild XICs from
    :param mz_tol_1: absolute width of m/z tolerance windown for I. traversal
    :param mz_tol_2: absolute width of m/z tolerance windown for II. traversal
    :param intensity_rel_tol: peaks 
    :param xic_len_thld: threshold for the number of peaks in XICs (XICs are filtered both after I. and II.)
    :param n_highest_peaks: number of highest peaks to choose in I.

    NOTE: Since such XICs contain all peaks in the neighbourhood, they are refered to as tight XICs.

    TODO: improve speed, very slow.
    """

    ms1_spectra = [spectrum for spectrum in msdata if spectrum.getMSLevel() == 1]

    # I. First traversal
    # Build XIC for m/z of each base peak

    xics = []
    xics_mzs = []

    for i in range(len(ms1_spectra)):
        spectrum = ms1_spectra[i]

        highest_peaks = su.get_highest_peaks(spectrum.get_peaks(), n_highest_peaks)

        for xic_mz, xic_in in highest_peaks:

            # NOTE: 1.5 m/z: small enough to capture many distinct m/z's
            # and high enough not to capture distinct m/z's together
            if utils.contains_similar(xics_mzs, xic_mz, 1.5):
                continue

            # Add the peak to final XIC
            xic = [(xic_mz, xic_in, spectrum.getRT())]

            last_intensity = xic_in
            # Search "same" (up to mz_tol) m/z values in previous spectra
            for j in reversed(range(i)):

                prev_spectrum = ms1_spectra[j]
                if len(prev_spectrum.get_peaks()[0]) == 0:
                    break

                mz, intensity = su.get_closest_mz_peak(prev_spectrum.get_peaks(), xic_mz)

                if abs(xic_mz - mz) > mz_tol_1 or intensity < last_intensity * intensity_rel_tol:
                    break
                else:
                    xic.insert(0, (mz, intensity, prev_spectrum.getRT()))

            last_intensity = xic_in
            # Search "same" (up to mz_tol) m/z values in next spectra
            for j in range(i + 1, len(ms1_spectra)):

                next_spectrum = ms1_spectra[j]
                if len(next_spectrum.get_peaks()[0]) == 0:
                    break

                mz, intensity = su.get_closest_mz_peak(next_spectrum.get_peaks(), xic_mz)

                if abs(xic_mz - mz) > mz_tol_1 or intensity < last_intensity * intensity_rel_tol:
                    break
                else:
                    xic.append((mz, intensity, next_spectrum.getRT()))

            xics.append(np.array(xic).T)
            xics_mzs.append(xic_mz)

    # Filter out xics having less than xic_len_thld peaks
    xics = [xic for xic in xics if len(xic[0]) >= xic_len_thld]
    xics1 = xics

    # II. Second traversal
    # Build new XICs based on median values of previous XICs

    median_mzs = [stats.median(xic[0]) for xic in xics]

    # 1. Find highest peaks of the new XICs
    # Mzs close to median_mzs but with highest intensities across the whole msdata
    highest_peaks = []
    for median_mz in median_mzs:

        # m/z, intensity, i
        highest_peak = -1, -1, -1
        for i, spectrum in enumerate(ms1_spectra):

            if len(spectrum.get_peaks()[0]) == 0:
                continue

            mz, intensity = su.get_closest_mz_peak(spectrum.get_peaks(), median_mz)
            if intensity > highest_peak[1] and abs(median_mz - mz) < mz_tol_1:
                highest_peak = mz, intensity, i

        if highest_peak[2] != -1:
            highest_peaks.append(highest_peak)

    # 2. Build new (tight) XICs: go left and right from highest peaks
    xics = []
    for highest_peak in highest_peaks:
        xic_mz = highest_peak[0]
        xic_in = highest_peak[1]
        i = highest_peak[2]

        xic = [(highest_peak[0], highest_peak[1], ms1_spectra[i].getRT())]

        last_intensity = xic_in
        # Search "same" (up to mz_tol) m/z values in previous spectra
        for j in reversed(range(i)):

            prev_spectrum = ms1_spectra[j]
            if len(prev_spectrum.get_peaks()[0]) == 0:
                break

            mz, intensity = su.get_closest_mz_peak(prev_spectrum.get_peaks(), xic_mz)

            if abs(xic_mz - mz) > mz_tol_2 or intensity < last_intensity * intensity_rel_tol:
                break
            else:
                xic.insert(0, (mz, intensity, prev_spectrum.getRT()))

        last_intensity = xic_in
        # Search "same" (up to mz_tol) m/z values in next spectra
        for j in range(i + 1, len(ms1_spectra)):

            next_spectrum = ms1_spectra[j]
            if len(next_spectrum.get_peaks()[0]) == 0:
                break

            mz, intensity = su.get_closest_mz_peak(next_spectrum.get_peaks(), xic_mz)

            if abs(xic_mz - mz) > mz_tol_2 or intensity < last_intensity * intensity_rel_tol:
                break
            else:
                xic.append((mz, intensity, next_spectrum.getRT()))

        xics.append(np.array(xic).T)

    # Filter out xics having less than xic_len_thld peaks
    xics = [xic for xic in xics if len(xic[0]) >= xic_len_thld]

    return xics1, xics


def sorted_by_rt(msdata):
    return utils.is_sorted([s.getRT() for s in msdata])


def sort_by_rt(msdata):
    return msdata.setSpectra(sorted(msdata.getSpectra(), key=lambda s: s.getRT(), reverse=True))

def remove_electromagnetic_spectra(msdata):
    filtered_spectra = [spectrum for spectrum in msdata if not spectrum.getMetaValue('lowest observed wavelength')]
    msdata.setSpectra(filtered_spectra)
    return msdata

def get_instrument_props(msdata):

    xics1, xics = get_tight_xics(msdata)
    xics_stdev = [stats.stdev(xic[0]) for xic in xics]

    quality_props = {
        'instrument name': msdata.getInstrument().getName(),
        '#TBXICs(1)': len(xics1),
        '#TBXICs': len(xics),
        'TBXICs mean stdev': stats.mean(xics_stdev) if xics_stdev else None,
        'TBXICs median stdev': stats.median(xics_stdev) if xics_stdev else None
    }

    return quality_props


def get_pwiz_stats(msdata):
    """
    Checks the presence of spectra centroided by ProteoWizard msconvert yet having zero intensities. Outputs the number
    of such spectra and the histogram of types of spectra converted by msconvert.
    """

    pwiz_stats = Counter()
    for i, spectrum in enumerate(msdata):
        spec_type = spectrum.getType()
        for dp in spectrum.getDataProcessing():
            pwiz = 'proteowizard' in dp.getSoftware().getName().lower()
            conversion_mzml = pyms.ProcessingAction.CONVERSION_MZML in dp.getProcessingActions()
            if pwiz and conversion_mzml:
                pwiz_stats['pwiz_to_mzml_type={}'.format(pyopenms_type_to_spectype(spec_type))] += 1
                peaks = spectrum.get_peaks()
                if spec_type == pyms.SpectrumSettings.SpectrumType.PEAKS and peaks and np.count_nonzero(peaks[1] == 0):
                    pwiz_stats[f'pwiz_zero_mz_centroid'] += 1
    return pwiz_stats


def get_spectrum_type(spec: pyms.MSSpectrum, to_int=False) -> SpecType:

    if spec is None:
        return None
    pyopenms_type = spec.getType()
    if pyopenms_type is None:
        return None

    if pyopenms_type == pyms.SpectrumSettings.SpectrumType.UNKNOWN:  # 0 enum int
        spec_type = SpecType.UNKNOWN
    elif pyopenms_type == pyms.SpectrumSettings.SpectrumType.CENTROID:  # 1 enum int
        spec_type = SpecType.CENTROID
    elif pyopenms_type == pyms.SpectrumSettings.SpectrumType.PROFILE:  # 2 enum int
        spec_type = SpecType.PROFILE
    elif pyopenms_type == pyms.SpectrumSettings.SpectrumType.SIZE_OF_SPECTRUMTYPE:  # 3 enum int
        spec_type = SpecType.SIZE_OF_SPECTRUMTYPE
    else:
        spec_type = SpecType.INVALID

    return spec_type.value if to_int else spec_type


def estimate_peak_list_type(pl: np.array, to_int=True, verbose=False):
    """
    Reproduced from MZmine.
    https://github.com/mzmine/mzmine3/blob/master/src/main/java/io/github/mzmine/util/scans/ScanUtils.java#L609

    ASSUMES PEAK LIST TO BE SORTED BY M/Z (no check in favor of performance).
    """

    peaks_n = su.get_num_peaks(pl)
    if verbose:
        print('Num. peaks:', peaks_n)
    if peaks_n < 5:
        return SpecType.CENTROID.value if to_int else SpecType.CENTROID

    mzs = pl[0]
    intensities = pl[1]

    bp_mz, bp_in, bp_i = su.get_base_peak(pl, return_i=True)
    bp_min_i, bp_max_i = su.get_peak_intens_nbhd(pl, bp_i, bp_in / 2, intens_thld_below=False)

    bp_span = bp_max_i - bp_min_i + 1
    bp_mz_span = mzs[bp_max_i] - mzs[bp_min_i]
    mz_span = mzs[-1] - mzs[0]
    if verbose:
        print('Size of base peak span:', bp_span)
        print(f'Base peak m/z span: {bp_mz_span:.2f}')
        print(f'M/z span: {mz_span:.2f} (0.1% of m/z span: {mz_span / 1000:.2f})')
    if bp_span < 3 or bp_mz_span > mz_span / 1000:
        spec_type = SpecType.CENTROID
    else:
        if (intensities == 0).any():
            spec_type = SpecType.PROFILE
        else:
            spec_type = SpecType.THRESHOLDED

    return spec_type.value if to_int else spec_type


def standartize_gnps_species(species: pd.Series):

    # Lowercase
    species = species.str.lower()

    # Add NCBITaxon suffix if known from other entries
    ncbi_suffix = ' (NCBITaxon:'.lower()
    species_to_ncbi = {s.split(ncbi_suffix)[0]: s.split(ncbi_suffix)[1] for s in species.unique().tolist() if isinstance(s, str) and ncbi_suffix in s}
    species = species.apply(lambda s: s if s not in species_to_ncbi else s + ncbi_suffix + species_to_ncbi[s])

    # Manually merge similar species
    species_merged = [
        (['Homo sapiens (NCBITaxon:9606)', 'homo sapiens', 'human', 'Human'], 'Human'),
        (['Mus musculus domesticus', 'Mus musculus (NCBITaxon:10090)', 'Rattus norvegicus (NCBITaxon:10116)', 'Rattus (NCBITaxon:10114)', 'C57BL/6N', 'Mus sp. (NCBITaxon:10095)', 'mice', 'Mice'], 'Mice'),
        (['Ocean Environmental Samples', 'environmental samples <Bacillariophyta> (NCBITaxon:33858)', 'environmental samples <Verrucomicrobiales> (NCBITaxon:48468)', 'environmental samples <delta subdivision> (NCBITaxon:34033)'], 'Environmental')
    ]
    species_merge_map = {}
    for k, v in species_merged:
        for s in k:
            species_merge_map[s.lower()] = v.lower()
    species = species.apply(lambda s: species_merge_map[s] if s in species_merge_map else s)

    # Other
    species = species.rename({'': 'other'})

    return species