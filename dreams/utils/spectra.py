import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
# import pyopenms as pyms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import rdkit.Chem.Descriptors as rdkitDescriptors
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import dreams.utils.misc as utils
import plotly.graph_objs as go
from typing import List, Union, Iterable
from dreams.utils.misc import get_closest_values, contains_similar
from dreams.utils.mols import mol_to_formula, formula_to_dict
from dreams.utils.plots import init_plotting, save_fig, get_nature_hex_colors

# Terms:
# - raw peak list - i.e. '53.0379 0.894101\n54.0335 0.661867\n'
# - peak list - i.e. ([53.0379, 54.0335, 55.0123], [0.894101, 0.661867, 1.0])
#   - high - of shape (num. peaks, 2)
#   - wide - of shape (2, num.peaks)
# - spectrum - instance of MSnSpectrum


# ------------------------------------ #
# Processing of peak lists and spectra #
# ------------------------------------ #


def parse_raw_peak_list(peak_list: str):
    """
    Parses peak list string into numpy arrays of m/z and intensity values
    e.g. '53.0379 0.894101\n54.0335 0.661867\n' -> ([53.0379, 54.0335], [0.894101, 0.661867])
    """
    try:
        peak_list = [peak.split(' ') for peak in peak_list.split('\n')]
        # Select only m/z, intensity pairs (NIST20 may contain additional annotations)
        peak_list = [(peak[0], peak[1]) for peak in peak_list]
        peak_list = np.array(peak_list, dtype=float)
        peak_list = peak_list.T
        return peak_list
    except Exception as e:
        print(f'Invalid peak list {peak_list}')


def is_valid_peak_list(peak_list: np.array, relative_intensities=True, verbose=None, return_problems_list=False):
    """
    Returns True if peak list is valid (Numbers of m/z and intensity values are equal,
    m/z values are sorted in ascending order etc.), else False.
    TODO: consider padded spectra.

    :param peak_list: np.array of shape (2, n), where n is a number of peaks.
    :param relative_intensities: if True, performs additional checks for the intensities to be relative.
    :param return_problems_list: if True, list of strings describing problems (invalid causes) will be returned
                                 (e.g. ['#mzs != #intensities', 'Exists m/z < 0.0']).
    :param verbose: for 'problems' the reasons why peak list is not valid will be printed, for 'problems_and_peak_list',
                    the peak list will be printed as well.
    """
    mzs = peak_list[0]
    intensities = peak_list[1]

    def predicate_eval(predicate_res, problem_str):
        if not predicate_res:
            return problem_str
        else:
            return None

    predicates_evals = [
        # 1) Peak least must have peaks
        predicate_eval(mzs.size != 0 and intensities.size != 0, 'No peaks'),
        # 2) Numbers of m/z and intensity values must be equal
        predicate_eval(len(mzs) == len(intensities), '#mzs != #intensities'),
        # 3) All intensities must be non-zero
        predicate_eval((intensities != 0.0).all(), 'Exists intensity = 0.0'),
        # 4) All intensities must be positive (overlaps with 3 but better for more verbose output)
        predicate_eval((intensities >= 0.0).all(), 'Exists intensity < 0.0'),
        # 5) M/z values must be positive
        predicate_eval((mzs > 0).all(), 'Exists m/z <= 0.0'),
        # 6) M/z values must be sorted in ascending order
        predicate_eval((mzs[:-1] <= mzs[1:]).all(), 'M/z values are not sorted'),
        # 7) All m/z values must be unique
        predicate_eval(np.unique(mzs).size == mzs.size, 'M/z values are not unique')
    ]

    if relative_intensities:
        predicates_evals.extend([
            # 8) All intensities must be <= 1. (relative intensities)
            predicate_eval((intensities <= 1.).all(), 'Exists intensity > 1.'),
            # 9) Spectrum must have base peak
            predicate_eval((intensities == 1.).any(), 'No base peak'),
        ])

    problems_list = [e for e in predicates_evals if e]

    if problems_list:
        if verbose == 'problems' or verbose == 'problems_and_peak_list':
            print('Problems:', problems_list)
        if verbose == 'problems_and_peak_list':
            print(peak_list)

    if return_problems_list:
        return problems_list
    else:
        if len(problems_list) == 0:
            return True
        else:
            return False


# TODO: refactor. It is old and is used only in MassIVE process_ms_file and spectra libs analysis.
def process_peak_list(
        peak_list,
        n_highest=None,
        sort_mzs=False,
        to_rel_intens=False
):
    mzs = peak_list[0]
    intensities = peak_list[1]

    # a) Select n_highest highest peaks (idempotent)
    if n_highest:
        highest_peaks_idx = intensities.argsort()[-n_highest:]
        mzs = mzs[highest_peaks_idx]
        intensities = intensities[highest_peaks_idx]

    # b) Sort peaks by m/z values (idempotent)
    if sort_mzs:
        sorted_peaks_idx = mzs.argsort()
        mzs = mzs[sorted_peaks_idx]
        intensities = intensities[sorted_peaks_idx]

    # c) Make intensities relative (idempotent)
    if to_rel_intens:
        intensities = (intensities / max(intensities)) * 100.0

    return np.array([mzs, intensities])


def pad_peak_list(pl: np.ndarray, target_len: int, pad_val: float = 0, axis: int = -1) -> np.ndarray:
    """
    Pads peak list to the `target_len` with `pad_val` or performs this for a batch of peak lists.
    :param pl: Peak list of shape (2, num_peaks) or a batch of peak lists of shape (batch_size, 2, num_peaks).
    :param target_len: Target num. of peaks of the peak list.
    :param pad_val: Value used for padding.
    :param axis: Axis along which the padding is performed.
    """
    pad_size = target_len - pl.shape[axis]
    if pad_size <= 0:
        return pl

    npad = [(0, 0)] * pl.ndim
    npad[axis] = (0, pad_size)
    return np.pad(pl, pad_width=npad, mode='constant', constant_values=pad_val)


def unpad_peak_list(peak_list: np.array, pad_val=0.0):
    return peak_list[:, peak_list[0] != pad_val]


def trim_peak_list(peak_list: np.array, n_highest: int):
    """
    Trims peak list by selecting `n_highest` highest peaks or performs this for a batch of peak lists.
    :param peak_list: np.array of shape (2, num_peaks) or (num_spectra, 2, num_peaks).
    :param n_highest: Number of highest peaks to be selected.
    """
    if len(peak_list.shape) == 2:  # Single spectrum case
        intensities = peak_list[1]
        highest_peaks_idx = np.argsort(intensities)[-n_highest:]
        return peak_list[:, np.sort(highest_peaks_idx)]

    # Determine indices of highest peaks for each spectrum
    intensities = peak_list[:, 1, :]
    highest_peaks_idx = np.argsort(intensities, axis=-1)[:, -n_highest:]

    # Sort the indices to match the original order (ascending m/z values)
    highest_peaks_idx = np.sort(highest_peaks_idx, axis=-1)

    # Use advanced indexing to select the highest peaks
    selected_peak_lists = peak_list[np.arange(len(peak_list))[:, np.newaxis], :, highest_peaks_idx]

    # Transpose the output to match the original shape (ChatGPT cannot come up with better solution for indexing)
    selected_peak_lists = selected_peak_lists.transpose(0, 2, 1)

    return selected_peak_lists


def get_base_peak(peak_list: np.array, return_i=False):
    mzs = peak_list[0]
    intensities = peak_list[1]
    bp_i = intensities.argmax()
    if return_i:
        return mzs[bp_i], intensities[bp_i], bp_i
    return mzs[bp_i], intensities[bp_i]


def get_highest_peaks(peak_list: np.array, n):
    """
    Returns n highest peaks.
    """
    mzs = peak_list[0]
    intensities = peak_list[1]

    highest_idx = np.flip(intensities.argsort()[-n:])
    return np.array((mzs[highest_idx], intensities[highest_idx])).T


def get_closest_mz_peaks(peak_list: np.array, query_mz, n):
    """
    Returns list of pairs (mz, intensity) of length n containing peaks having m/z
    values closest to the query_mz sorted ascending by the difference.
    """
    mzs = peak_list[0]
    intensities = peak_list[1]

    closest_peak_idx = get_closest_values(mzs, query_mz, n=n, return_idx=True)
    return np.array((mzs[closest_peak_idx], intensities[closest_peak_idx])).T


def get_closest_mz_peak(peak_list: np.array, query_mz):
    return get_closest_mz_peaks(peak_list, query_mz, 1)[0]


def has_peak_at(peak_list: np.array, query_mz, epsilon):
    mzs = peak_list[0]
    return contains_similar(mzs, query_mz, epsilon)


def get_num_peaks(peak_list):
    return peak_list.shape[1]


def get_peak_intens_nbhd(peak_list, peak_i, intens_thld, intens_thld_below=True):
    """
    Returns indices determining the range of the neighbour around peak at peak_i. The neighbourhood is defined as all
    consecutive peaks above (or below if intens_thld_below=False) the intens_thld intensity.
    """

    if (intens_thld_below and peak_list[1][peak_i] > intens_thld) or \
            (not intens_thld_below and peak_list[1][peak_i] < intens_thld):
        raise ValueError('Intensity at peak_i does not satisfy intens_thld.')

    thld_mask = np.concatenate([
        [False],
        peak_list[1] < intens_thld if intens_thld_below else peak_list[1] > intens_thld,
        [False]
    ])
    bp_min_i = peak_i - np.argmin(np.flip(thld_mask[:peak_i + 1]))  # np.argmin detects the first False element
    bp_max_i = peak_i + np.argmin(thld_mask[peak_i + 2:])
    return bp_min_i, bp_max_i


def intens_amplitude(peak_list):
    return max(peak_list[1]) / min(peak_list[1])


def num_high_peaks(peak_list, high_intensity_thld):
    return (peak_list[1] > high_intensity_thld).sum()


def max_mz(peak_list):
    return peak_list[0].max()

@njit()
def _bin_peak_list(peak_list: np.array, max_mz: float, bin_step: float) -> list:
    mzs, intensities = peak_list

    bin_ub = bin_step
    num_bins = int(max_mz / bin_step)

    binned_pl = []
    for _ in range(num_bins):  # Iterate over number of bins to avoid floating point errors
        bin_intensity = 0.
        for i, mz in enumerate(mzs):
            if bin_ub - bin_step <= mz < bin_ub:
                bin_intensity += intensities[i]
        binned_pl.append(bin_intensity)
        bin_ub += bin_step

    return binned_pl


def bin_peak_list(peak_list: np.array, max_mz: float, bin_step: float) -> np.array:
    return np.array(_bin_peak_list(peak_list, max_mz, bin_step))


@njit()
def _bin_peak_lists(peak_lists: np.array, max_mz: float, bin_step: float) -> list:
    return [_bin_peak_list(pl, max_mz, bin_step) for pl in peak_lists]


def bin_peak_lists(peak_lists: np.array, max_mz: float, bin_step: float) -> np.array:
    return np.array(_bin_peak_lists(peak_lists, max_mz, bin_step))


def to_rel_intensity(peak_list: np.array, scale_factor=None):
    pl = np.array([peak_list[0], peak_list[1] / peak_list[1].max()])
    if scale_factor:
        pl[1] *= scale_factor
    return pl


def merge_peak_lists(peak_lists: List[np.array], eps=1e-2, n_highest_peaks=None) -> np.array:
    """
    Merges peak lists without creating new "artificial" m/z values. The algorithm traverses all peaks (from all spectra)
    descendingly ordered by their intensities and create merged peaks by summing up intensities of all peaks in the
    range m/z ± `eps`. Each peak is used exactly once (either as the one determining the range or the one belonging to
    the range), and final peaks are not transitively connected.
    Notice, that the complexity is O(n^2), where n is a total num. of peaks within all spectra.
    :param peak_lists: List of NumPy arrays of shape (2, num. of peaks).
    :param eps: Epsilon determining the range of m/z values of peaks which are aggregated.
    :param n_highest_peaks: If not None, `n_highest_peaks` highest peaks are selected from each peak list.
    """

    if len(peak_lists) == 1:
        return peak_lists[0]

    if n_highest_peaks:
        peak_lists = [trim_peak_list(pl, n_highest_peaks) for pl in peak_lists]

    # Merge all peaks
    pls = np.hstack(peak_lists)

    # Sort peaks by intensities (descending order)
    pls = pls[:, np.argsort(pls[1])[::-1]]

    merged_pl = []
    visited = np.zeros(pls.shape[1])
    for i in range(pls.shape[1]):
        if visited[i]:
            continue
        visited[i] = 1

        # Collect indices of peaks having same m/z's up to the difference of `eps`
        peak_idx = [i]
        for j in range(i + 1, pls.shape[1]):
            if abs(pls[0, i] - pls[0, j]) < eps:
                peak_idx.append(j)
                visited[j] = 1

        # Create merged peak
        merged_pl.append([
            pls[0, peak_idx[0]],  # M/z of the highest peak
            sum(pls[1, k] for k in peak_idx)  # Sum of intensities
        ])

    # Sort merged peak list by m/z values
    merged_pl = np.array(merged_pl).T
    merged_pl = merged_pl[:, np.argsort(merged_pl[0])]

    return merged_pl

# ----------------------------- #
# Processing of high peak lists #
# ----------------------------- #
# TODO: Everything should be refactored to use only high form (since torch forward is applied along the last dim).


def prepend_precursor_peak(peak_list: np.array, prec_mz, prec_in=1.1, high=False):
    if not high:
        raise NotImplementedError
    return np.vstack([np.array([prec_mz, prec_in]), peak_list])


def normalize_mzs(peak_list: np.array, max_mz: float, in_place=True, high=False):
    if not high:
        raise NotImplementedError
    if not in_place:
        return np.array([peak_list[:, 0] / max_mz, peak_list[:, 1]])
    peak_list[:, 0] /= max_mz
    return peak_list


# --------------------------- #
# Data structures for spectra #
# --------------------------- #


class MSnSpectrum:
    def __init__(
            self,
            peak_list,
            precursor_mol=None,
            precursor_mz=None,
            precursor_charge=None,
            ionization_mode=None,
            collision_energy=None,
            assert_is_valid=True
    ):

        if assert_is_valid:
            assert is_valid_peak_list(peak_list, verbose='problems_and_peak_list'), 'Peak list is not valid'

        super(MSnSpectrum, self).__init__()

        self.peak_list = peak_list
        self.precursor_mol = precursor_mol
        self.precursor_mz = precursor_mz
        self.precursor_charge = precursor_charge
        self.ionization_mode = ionization_mode
        self.collision_energy = collision_energy

    def get_peak_list(self):
        return self.peak_list

    def get_mzs(self):
        return self.peak_list[0]

    def get_intensities(self):
        return self.peak_list[1]

    def get_precursor_mol(self):
        return self.precursor_mol

    def get_precursor_formula(self, to_dict=False):
        formula = mol_to_formula(self.precursor_mol)
        if to_dict:
            formula = formula_to_dict(formula)

        return formula

    def get_collision_energy(self):
        return self.collision_energy

    def get_ionization_mode(self):
        return self.ionization_mode

    def get_precursor_mz(self):
        return self.precursor_mz

    def get_precursor_charge(self):
        return self.precursor_charge

    def get_precursor_mass(self):
        return rdkitDescriptors.ExactMolWt(self.precursor_mol)

    def get_peaks_n(self):
        return len(self.get_mzs())

    # def to_TIC(self):
    #     """
    #     Returns new spectrum with intensities normalized to sum up to 1.
    #     """
    #     pass

    #     # Create new instance of MSnSpectrum
    #     # TODO
    #     #tic = MSnSpectrum(self.get_peaks(), self.precursor_mol)

    #     # Define normalizer
    #     normalizer = pyms.Normalizer()
    #     param = normalizer.getParameters()
    #     param.setValue('method', 'to_TIC')
    #     normalizer.setParameters(param)

    #     # Transform spectrum
    #     normalizer.filterSpectrum(tic)

    #     return tic

    # def is_TIC(self):
    #     return math.isclose(sum(self.get_peaks()[1]), 1.0, abs_tol=1e-4)

    # TODO: rewrite to plotly
    # def plot(self, highl_peaks=[]):
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
    #
    #     # Plot every peak in spectrum and annotate m/z for peaks with
    #     for peak_i, (mz, i) in enumerate(zip(*self.get_peak_list())):
    #         color = 'red' if peak_i in highl_peaks else 'navy'
    #         ax1.plot([mz, mz], [0, i], color=color)
    #         if i >= 10.0 or peak_i == self.get_peaks_n() - 1 or peak_i == 0:
    #             ax1.text(mz, i, '{:.4f}'.format(mz))
    #
    #     ax1.set_ylabel('Intensity')
    #     ax1.set_xlabel('M/z')
    #     ax1.set_ylim(bottom=0)
    #
    #     # Plot precursor structure
    #     if self.precursor_mol:
    #         img = Draw.MolToImage(self.precursor_mol)
    #         ax2.imshow(img, origin='upper')
    #
    #         # X axis label
    #         formula_str = 'Formula: {}\n'.format(self.get_precursor_formula())
    #         mz_str = 'M/z: {:.4f}\n'.format(self.precursor_mz) if self.precursor_mz else ''
    #         mass_str = 'Mass: {:.4f}\n'.format(self.get_precursor_mass())
    #         ax2_xlabel = ''.join([formula_str, mz_str, mass_str])
    #         ax2.set_xlabel(ax2_xlabel)
    #
    #         ax2.set_xticks([])
    #         ax2.set_yticks([])
    #
    #     plt.show()
    def plot(self):
        pass


def plot_spectrum(spec, hue=None, xlim=None, ylim=None, mirror_spec=None, highl_idx=None, high_peaks_at=None,
                  figsize=(6, 2), colors=None, save_pth=None):

    if colors == 'nature':
        colors = get_nature_hex_colors()
        colors = [colors[1], colors[0], colors[2]]
    elif colors is not None:
        assert len(colors) >= 3
    else:
        colors = ['blue', 'green', 'red']

    # Normalize input spectrum
    def norm_spec(spec):
        assert len(spec.shape) == 2
        if spec.shape[0] != 2:
            spec = spec.T
        spec = unpad_peak_list(spec)
        mzs, ins = to_rel_intensity(spec, scale_factor=100.0)
        return mzs, ins
    mzs, ins = norm_spec(spec)

    # Initialize plotting
    init_plotting(figsize=figsize)
    fig, ax = plt.subplots(1, 1)
    if high_peaks_at:
        highl_idx = [utils.get_closest_values(mzs, query_val=m, return_idx=True).item() for m in high_peaks_at]

    # Setup color palette
    if hue is not None:
        norm = matplotlib.colors.Normalize(vmin=min(hue), vmax=max(hue), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)
        plt.colorbar(mapper, ax=ax)

    # Plot spectrum
    for i in range(len(mzs)):
        if highl_idx is not None and i in highl_idx:
            color = colors[1]
        elif hue is not None:
            color = mcolors.to_hex(mapper.to_rgba(hue[i]))
        else:
            color = colors[0]
        plt.plot([mzs[i], mzs[i]], [0, ins[i]], color=color, marker='o', markevery=(1, 2), mfc='white', zorder=2)

    # Plot mirror spectrum
    if mirror_spec is not None:
        mzs_m, ins_m = norm_spec(mirror_spec)

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            label = str(round(-x)) if x < 0 else str(round(x))
            return label

        for i in range(len(mzs_m)):
            plt.plot([mzs_m[i], mzs_m[i]], [0, -ins_m[i]], color=colors[2], marker='o', markevery=(1, 2), mfc='white',
                     zorder=1)
        ax.yaxis.set_major_formatter(major_formatter)

    # Setup axes
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    else:
        plt.xlim(0, max(mzs) + 10)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel('m/z')
    plt.ylabel('Intensity [%]')

    if save_pth is not None:
        save_fig(save_pth)
    plt.show()


def df_to_MSnSpectra(df, assert_is_valid=True, as_new_column=False):
    """
    Processes NIST-like DataFrame to Series of MSnSpectra.
    # TODO: include more columns.
    """

    msn_spectra = df.apply(lambda row: MSnSpectrum(
        peak_list=row['PARSED PEAKS'],
        precursor_mol=row['ROMol'] if 'ROMol' in df.columns else None,
        precursor_mz=row['PRECURSOR M/Z'],
        precursor_charge=row['CHARGE'],
        assert_is_valid=assert_is_valid
    ), axis=1)

    if as_new_column:
        df['MSnSpectrum'] = msn_spectra
        return df
    else:
        return msn_spectra


# ------------------ #
# PyTorch operations #
# ------------------ #


def num_hot_classes(max_val: float, bin_size: float) -> int:
    num_classes = max_val / bin_size
    assert num_classes.is_integer()
    return int(num_classes)


def to_classes(vals: torch.Tensor, max_val: float, bin_size: float, special_vals=(), return_num_classes=False):
    """ Assumes that last dimension of mzs is singleton. """
    special_masks = [vals == v for v in special_vals]
    num_classes = num_hot_classes(max_val, bin_size)
    classes = torch.round(vals / bin_size).long()
    classes = classes.clamp(max=num_classes - 1)  # clamp not to have a separate class for max_mz
    for i, m in enumerate(special_masks):
        classes[m] = num_classes + i
    if return_num_classes:
        return classes, num_classes + len(special_vals)
    return classes


def to_hot(vals: torch.Tensor, max_val: float, bin_size: float, dtype=torch.double):
    """ Assumes that last dimension of mzs is singleton. """
    classes, num_classes = to_classes(vals, max_val, bin_size, return_num_classes=True)
    hots = F.one_hot(classes, num_classes=num_classes)
    return hots.squeeze(dim=-2).to(dtype)


def from_hot(hots: torch.Tensor, bin_size: float, dtype=torch.double) -> torch.Tensor:
    """ Makes the last dimension singleton. """
    vals = torch.argmax(hots, dim=-1) * bin_size
    return vals.unsqueeze(-1).to(dtype)


def from_hot_logits(vals: torch.Tensor, bin_size: float) -> torch.Tensor:
    hots = (vals == vals.max(dim=-1, keepdim=True)[0]).long()
    return from_hot(hots, bin_size)


class PeakListModifiedCosine:
    def __init__(self, mz_tolerance: float = 0.05, unpad: bool = True):
        self.cos_sim = ModifiedCosine(tolerance=mz_tolerance)
        self.unpad = unpad

    def _peak_list_to_matchms(self, peak_list: np.ndarray, prec_mz: float) -> Spectrum:
        assert peak_list.shape[0] == 2, 'Peak list must have shape (2, num. of peaks).'
        if self.unpad:
            peak_list = unpad_peak_list(peak_list)
        return Spectrum(mz=peak_list[0], intensities=peak_list[1], metadata={'precursor_mz': prec_mz})

    def compute(self, spec1: np.ndarray, spec2: np.ndarray, prec_mz1: float, prec_mz2: float) -> float:
        spec1 = self._peak_list_to_matchms(spec1, prec_mz1)
        spec2 = self._peak_list_to_matchms(spec2, prec_mz2)
        return self.cos_sim.pair(spec1, spec2)['score'].item()

    def __call__(self, spec1: np.ndarray, spec2: np.ndarray, prec_mz1: float, prec_mz2: float) -> float:
        return self.compute(spec1, spec2, prec_mz1, prec_mz2)

    def compute_pairwise(self, specs: np.ndarray, prec_mzs: np.ndarray, avg=True) -> Union[np.ndarray, float]:
        specs = [self._peak_list_to_matchms(spec, float(prec_mz)) for spec, prec_mz in zip(specs, prec_mzs)]
        sims = self.cos_sim.matrix(specs, specs, is_symmetric=True)['score']
        if avg:
            return sims.mean().item()
        return sims


# ------------------------------------------------------------------------- #
# Transformations of spectra for PyTorch models (similar to PyTorch vision) #
# ------------------------------------------------------------------------- #
# WARNING: operations below are non-idempotent -> do not call in e.g. __getitem__ of torch.Dataset
# without preliminary tensor.clone()
# DEPRECATED
# class ComposeTransforms:
#     def __init__(self, transforms):
#         self.transforms = transforms
#
#     def __call__(self, x):
#         """
#         :param x: spectrum or batch of spectra
#         """
#         for transform in self.transforms:
#             x = transform(x)
#         return x
#
#
# class ToPairsOfPeaks:
#     def __call__(self, x):
#         return x.transpose(-2, -1)
#
#
# class FlattenSpectrum:
#     def __call__(self, x):
#         return x.flatten(-2, -1)
#
#
# class NormalizeMzs:
#     def __init__(self, const_factor=1000.0, pairs_of_peaks=False):
#         self.const_factor = const_factor
#         self.pairs_of_peaks = pairs_of_peaks
#
#     def __call__(self, sample):
#         if self.pairs_of_peaks:
#             sample[..., -2] = sample[..., -2] / self.const_factor
#         else:
#             sample[..., -2, :] = sample[..., -2, :] / self.const_factor
#         return sample
#
#
# class NormalizeIntensities:
#     def __init__(self, const_factor=100.0, pairs_of_peaks=False):
#         self.const_factor = const_factor
#         self.pairs_of_peaks = pairs_of_peaks
#
#     def __call__(self, sample):
#         if self.pairs_of_peaks:
#             sample[..., -1] = sample[..., -1] / self.const_factor
#         else:
#             sample[..., -1, :] = sample[..., -1, :] / self.const_factor
#         return sample
#
#
# class PrependPrecursor:
#     # TODO: for non-batched dims
#     def __init__(self, prec_mz: torch.Tensor, prec_in=1.1, pairs_of_peaks=False):
#         self.prec_mz = prec_mz
#         self.prec_in = prec_in
#         self.pairs_of_peaks = pairs_of_peaks
#
#     def __call__(self, sample):
#         prec_token = F.pad(input=self.prec_mz.unsqueeze(-1), pad=(0, 1, 0, 0), mode='constant', value=self.prec_in)
#         prec_token_dim = -2 if self.pairs_of_peaks else -1
#         return torch.cat([prec_token.unsqueeze(prec_token_dim), sample], dim=prec_token_dim)


# def prepend_prec(spec, prec_mz, prec_in=1.1):
#     prec_token = F.pad(input=prec_mz.unsqueeze(-1), pad=(0, 1, 0, 0), mode='constant', value=prec_in)
#     return torch.cat([prec_token.unsqueeze(-2), spec], dim=-2)
