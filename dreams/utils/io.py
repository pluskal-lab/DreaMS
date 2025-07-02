import logging
import sys
import pickle
import json
import os
import h5py
import click
import re
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import pyteomics
import pyopenms as oms
import matplotlib.pyplot as plt
import urllib.parse as urlparse
import contextlib
import time
import io as std_io
import wandb
import pyopenms as pyms
import traceback
from collections import Counter
from functools import cache
from matchms.importing import load_from_mgf
from pathlib import Path
from matchms import Spectrum
from typing import Tuple, List, Optional, Union, Iterable
from itertools import groupby
from tqdm import tqdm
import dreams.utils.spectra as su
import dreams.utils.misc as utils
import dreams.utils.lcms as lcms
import dreams.utils.dformats as dformats
from dreams.algorithms.lsh import BatchedPeakListRandomProjection
from dreams.definitions import *


def setup_logger(log_file_path=None, log_name='log'):

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TqdmToLogger(std_io.StringIO):
    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None, mininterval=5):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
        self.mininterval = mininterval
        self.last_time = 0

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        if len(self.buf) > 0 and time.time() - self.last_time > self.mininterval:
            self.logger.log(self.level, self.buf)
            self.last_time = time.time()


def append_to_stem(pth: Path, s, sep='_'):
    """
    path/to/file.txt -> path/to/file{sep}{s}.txt
    """
    return pth.parents[0] / f'{pth.stem}{sep}{s}{pth.suffix}'


def prepend_to_stem(pth: Path, s, sep='_'):
    """
    path/to/file.txt -> path/to/{s}{sep}file.txt
    """
    return pth.parents[0] / f'{s}{sep}{pth.stem}{pth.suffix}'


@cache
def cache_pkl(pth):
    return pd.read_pickle(pth)


def list_to_txt(lst, txt_pth, sep='\n'):
    with open(txt_pth, 'w') as f:
        f.write(sep.join(lst))
        if sep == '\n':
            f.write('\n')


def list_from_txt(txt_pth, sep='\n', apply_lambda=None, progress_bar=False):
    if sep != '\n':
        raise NotImplementedError
    with open(txt_pth, 'r') as f:
        lines = tqdm(f.readlines()) if progress_bar else f.readlines()
        lst = [line.rstrip() for line in lines]
        if apply_lambda is not None:
            lst = [apply_lambda(e) for e in lst]
        return lst


def read_pickle(pth):
    with open(pth, 'rb') as f:
        return pickle.load(f)


def write_pickle(obj, pth):
    with open(pth, 'wb') as f:
        pickle.dump(obj, f)


def read_json(pth):
    with open(pth, 'r') as f:
        return json.load(f)


def write_json(obj, pth):
    with open(pth, 'w') as f:
        json.dump(obj, f)


def bytes_to_human_str(size_bytes, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            break
        size_bytes /= 1024
    return f'{size_bytes:.{decimal_places}f} {unit}'


def bytes_to_units(size_bytes, unit='MB'):
    if unit == 'B':
        power = 0
    elif unit == 'KB':
        power = 1
    elif unit == 'MB':
        power = 2
    elif unit == 'GB':
        power = 3
    elif unit == 'TB':
        power = 4
    else:
        raise ValueError('Given unit is not one of "B", "KB", "MB", "GB" or "TB".')
    return size_bytes / 1024 ** power


def merge_ms_hdfs(in_hdf_pths, out_pth, group='MSn data', max_peaks_n=512, del_in=False, show_tqdm=True, logger=None,
                  add_file_name_dataset=True, mzs_dataset='mzs', intensities_dataset='intensities'):
    """
    TODO: This should be remove after MSData is completely implemented (inclduing .merge method).

    NOTE: currently ignores MS1 data and file-level metadata.
    NOTE: assumes identical keys in all files.
    TODO: when ms_hdfs are created with process_ms_file, mzs and intensities should be refactored to a single
          spectrum dataset. Here, args and body should be refactored to reflect this change.
    TODO: recursively merge groups?
    TODO: max_peaks_n=None
    :param group: str: Merge only datasets withing the given group.
    :param del_in: bool: Delete input files after merging.
    :param show_tqdm: bool: Show tqdm progress bar.
    :param logger: logging.Logger: Logger to log the progress.
    :param add_file_name_dataset: bool: Add a new dataset constantly filled with the file names of merged files.
    """
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    print(f'Using {os.environ["HDF5_USE_FILE_LOCKING"]} file locking.')

    with h5py.File(out_pth, 'w') as f_out:
        for i, hdf_pth in enumerate(tqdm(in_hdf_pths, desc='Merging hdfs')) if show_tqdm else enumerate(in_hdf_pths):
            hdf_pth = Path(hdf_pth)
            with h5py.File(hdf_pth, 'r') as f:

                if group:
                    f = f[group]

                f_len = f[list(f.keys())[0]].shape[0]
                if logger:
                    logger.info(f'Appending {i+1}th / {len(in_hdf_pths)} hdf5 ({f_len} samples) file to {out_pth}.')

                for k in list(f.keys()) + ['file_name'] if add_file_name_dataset else f.keys():

                    # New columns with constant file names
                    if k == 'file_name':
                        data = np.array([hdf_pth.stem] * f_len, dtype='S')

                    # Mzs and intensities to spectra (TODO: refactor (see docstring))
                    elif k in [mzs_dataset, intensities_dataset]:
                        if k == intensities_dataset:
                            continue
                        k = 'spectrum'
                        mzs = f[mzs_dataset][:]
                        intensities = f[intensities_dataset][:]
                        # (n_spec, n_peaks) + (n_spec, n_peaks) -> (n_spec, n_peaks, 2)
                        spectra = np.stack([mzs, intensities], axis=-1)
                        spectra = np.transpose(spectra, (0, 2, 1))
                        spectra = su.trim_peak_list(spectra, n_highest=max_peaks_n)
                        spectra = su.pad_peak_list(spectra, target_len=max_peaks_n)
                        data = spectra

                    # Other metadata datasets
                    else:
                        data = f[k][:]

                    if i == 0:
                        f_out.create_dataset(
                            k, data=data, shape=data.shape, maxshape=(None, *data.shape[1:]), dtype=data.dtype
                        )
                    else:
                        f_out[k].resize(f_out[k].shape[0] + data.shape[0], axis=0)
                        f_out[k][-data.shape[0]:] = data

                # # Add a new dataset constantly filled with the file name
                # if add_file_name_dataset:
                #     data = np.array([hdf_pth.stem] * f_len, dtype='S')
                #     print(data.shape, data[0])
                #     if i == 0:
                #         f_out.create_dataset('file_name', data=data)
                #     else:
                #         f_out['file_name'].resize(f_out['file_name'].shape[0] + data.shape[0], axis=0)
                #         f_out['file_name'][-data.shape[0]:] = data

            if del_in:
                os.remove(hdf_pth)
                if logger:
                    logger.info(f'{hdf_pth} ({i+1}th) deleted.')


def parse_sirius_ms(spectra_file: str) -> Tuple[dict, List[Tuple[str, np.ndarray]]]:
    """
    Parses spectra from the SIRIUS .ms file.

    Copied from the code of Goldman et al.:
    https://github.com/samgoldman97/mist/blob/4c23d34fc82425ad5474a53e10b4622dcdbca479/src/mist/utils/parse_utils.py#LL10C77-L10C77.
    :return Tuple[dict, List[Tuple[str, np.ndarray]]]: metadata and list of spectra tuples containing name and array
    """
    lines = [i.strip() for i in open(spectra_file, "r").readlines()]

    group_num = 0
    metadata = {}
    spectras = []
    my_iterator = groupby(
        lines, lambda line: line.startswith(">") or line.startswith("#")
    )

    for index, (start_line, lines) in enumerate(my_iterator):
        group_lines = list(lines)
        subject_lines = list(next(my_iterator)[1])
        # Get spectra
        if group_num > 0:
            spectra_header = group_lines[0].split(">")[1]
            peak_data = [
                [float(x) for x in peak.split()[:2]]
                for peak in subject_lines
                if peak.strip()
            ]
            # Check if spectra is empty
            if len(peak_data):
                peak_data = np.vstack(peak_data)
                # Add new tuple
                spectras.append((spectra_header, peak_data))
        # Get meta data
        else:
            entries = {}
            for i in group_lines:
                if " " not in i:
                    continue
                elif i.startswith("#INSTRUMENT TYPE"):
                    key = "#INSTRUMENT TYPE"
                    val = i.split(key)[1].strip()
                    entries[key[1:]] = val
                else:
                    start, end = i.split(" ", 1)
                    start = start[1:]
                    while start in entries:
                        start = f"{start}'"
                    entries[start] = end

            metadata.update(entries)
        group_num += 1

    metadata["_FILE_PATH"] = spectra_file
    metadata["_FILE"] = Path(spectra_file).stem
    return metadata, spectras


def read_ms(pth, peaks_tag='>ms2peaks', charge_tag='#Charge', prec_mz_tag='#Precursor_MZ'):
    with open(pth, 'r') as f:
        data = {
            'PARSED PEAKS': [],
            'CHARGE': None,
            'PRECURSOR M/Z': None
        }
        peaks_data = False
        for line in f.readlines():
            if peaks_data:
                data['PARSED PEAKS'].append([float(v) for v in line.strip().split(' ')])
            if line.startswith(charge_tag):
                data['CHARGE'] = int(line[len('#Charge'):])
            if line.startswith(prec_mz_tag):
                data['PRECURSOR M/Z'] = float(line[len('#Precursor_MZ'):])
            if line.startswith(peaks_tag):
                peaks_data = True
        data['PARSED PEAKS'] = np.array(data['PARSED PEAKS']).T
        return data


def read_textual_ms_format(
        pth,
        spectrum_end_line,
        name_value_sep,
        prec_mz_name,
        charge_name='CHARGE',
        adduct_name='ADDUCT',
        smiles_name='SMILES',
        ignore_line_prefixes=(),
        encoding='utf-8',
    ):
    # TODO: this is very raw and dirty.

    # Two numbers separated with a white space
    peak_pattern = re.compile(r'\b([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\b')
    # A word followed by an arbitrary string separated with `name_value_sep`
    attr_pattern = re.compile(rf'^\s*([A-Z_]+){name_value_sep}(.*)\s*$')
    attr_mapping = {prec_mz_name: PRECURSOR_MZ, charge_name: CHARGE, adduct_name: ADDUCT, smiles_name: SMILES}

    data = []
    with open(pth, 'r', encoding=encoding) as f:
        lines = f.readlines()

        # TODO?
        # if lines[-1] != spectrum_end_line:
        #     lines.append(spectrum_end_line)

    for i, line in enumerate(lines):

        if any([line.startswith(p) for p in ignore_line_prefixes]):
            continue
        elif line.rstrip() == spectrum_end_line or i == 0:
            if i != 0:
                spec[SPECTRUM] = np.array(spec[SPECTRUM])
                data.append(spec)
            spec = {SPECTRUM: [[], []]}
            if i != 0:
                continue

        # Attributes parsing
        match = attr_pattern.match(line)
        if match:
            k, v = match.group(1), match.group(2)
            if utils.is_float(v):
                v = float(v)
                if v.is_integer():
                    v = int(v)
            if k in attr_mapping:
                k = attr_mapping[k]
            spec[k] = v
            continue

        # Peaks parsing
        match = peak_pattern.match(line)
        if match:
            mz, intensity = float(match.group(1)), float(match.group(2))
            spec[SPECTRUM][0].append(mz)
            spec[SPECTRUM][1].append(intensity)
            continue

    return pd.DataFrame(data)


def read_msp(pth, **kwargs):
    return read_textual_ms_format(
        pth=pth,
        spectrum_end_line='',
        name_value_sep=': ',
        prec_mz_name='PRECURSORMZ',
        ignore_line_prefixes=('#',),
        **kwargs
    )


def read_mgf(pth, **kwargs):
    return read_textual_ms_format(
        pth=pth,
        spectrum_end_line='END IONS',
        name_value_sep='=',
        prec_mz_name='PEPMASS',
        ignore_line_prefixes=('#',),
        **kwargs
    )


def read_mzml(pth: Union[Path, str], verbose: bool = False, scan_range: Optional[Tuple[int, int]] = None):
    if isinstance(pth, str):
        pth = Path(pth)
    exp = oms.MSExperiment()
    if pth.suffix.lower() == '.mzml':
        oms.MzMLFile().load(str(pth), exp)
    elif pth.suffix.lower() == '.mzxml':
        oms.MzXMLFile().load(str(pth), exp)
    else:
        raise ValueError(f'Unsupported file extension: {pth.suffix}.')

    df = []
    automatic_scans_message = False
    for i, spec in enumerate(tqdm(exp, desc=f'Reading {pth.name}', disable=not verbose)):

        # Skip spectra that are not MS2
        mslevel = spec.getMSLevel()
        if mslevel != 2:
            continue

        # Get or assign scan numbers
        scan_i = re.search(r'scan=(\d+)', spec.getNativeID())
        if scan_i:
            scan_i = int(scan_i.group(1))
        else:
            if verbose and not automatic_scans_message:
                print(f'Assigning scan numbers automatically (no "scan=" in the file).')
            scan_i = i + 1
            automatic_scans_message = True

        if scan_range and (scan_i < scan_range[0] or scan_i > scan_range[1]):
            continue

        # Get peak list and check for problems
        peak_list = np.stack(spec.get_peaks())
        spec_problems = su.is_valid_peak_list(peak_list, relative_intensities=False, return_problems_list=True, verbose=verbose)
        if spec_problems:
            if verbose:
                print(f'Skipping spectrum {i} in {pth.name} with problems: {spec_problems}.')
            continue

        # Get precursor metadata
        prec = spec.getPrecursors()
        if len(prec) != 1:
            continue
        prec = prec[0]

        df.append({
            FILE_NAME: pth.name,
            SCAN_NUMBER: scan_i,
            SPECTRUM: peak_list,
            PRECURSOR_MZ: prec.getMZ(),
            RT: spec.getRT(),
            CHARGE: prec.getCharge(),
        })
    
    if scan_range and verbose:
        print(f'Read {len(df)} valid MS2 spectra with scan numbers in the range {scan_range}.'
              f' Max scan number in the file: {scan_i}.')

    df = pd.DataFrame(df)
    return df


def lcmsms_to_hdf5(
    input_path,
    output_path=None,
    num_peaks=None,
    num_prec_peaks=None,
    store_precursors=True,
    compress_peaks_lvl=0,
    compress_full_lvl=0,
    pwiz_stats=False,
    del_in=False,
    assign_dformats=True,
    log_path=None,
    verbose=False
    ):
    """
    Convert LC-MS/MS data from an input file (.mzML or .mzXML) to an output file (.hdf5).

    Args:
    input_path (str): Path to the input file (.mzML or .mzXML).
    output_path (str, optional): Path to the output file (.hdf5). If not provided, the output is stored as the
        input file name with .hdf5 extension.
    num_peaks (int, optional): The number of peaks to pad the MSn peak lists with zeros. If not specified, it
        will be set to the maximum number of peaks within the spectra that are to be stored.
    num_prec_peaks (int, optional): The number of peaks to pad the MS1 peak lists with zeros. If not specified,
        it will be set to the maximum number of peaks within the spectra that are to be stored.
    store_precursors (bool, optional): Whether to store the data of precursor spectra (peak list and scan id) for
        each MSn spectrum as a separate hdf5 dataset. Defaults to True.
    compress_peaks_lvl (int, optional): The compression level for peak lists in the output .hdf5 file. Should be an
        integer from 0 to 9. Defaults to 0.
    compress_full_lvl (int, optional): The compression level for all stored attributes (e.g. RTs, polarities, etc.)
        except for peak lists. Should be an integer from 0 to 9. Defaults to 0.
    pwiz_stats (bool, optional): Whether to collect ProteoWizard msconvert statistics, including the histogram of
        types of spectra converted by msconvert and the number of spectra centroided by msconvert but having zero
        intensities. Defaults to False.
    del_in (bool, optional): Whether to delete the input .mzML or .mzXML file. Defaults to False.
    assign_dformats (bool, optional): Whether to assign data formats to MSn spectra. Defaults to True.
    log_path (str, optional): Path to the log file containing errors during opening of files and flaws of invalid
        spectra. If set to None, the log file is stored as the input file name with .hdf5 extension.
    verbose (bool, optional): Whether to log the scan number for each invalid spectrum and log additional
        statistics. The statistics are redundant in a sense that they can be calculated from the output .hdf5 file
        but are helpful for the fast analysis of the input file and debugging. Defaults to False.
    """
    input_path = str(input_path)
    output_path = str(output_path) if output_path else None

    # Create a logger
    if not log_path:
        log_path = os.path.splitext(output_path)[0] + '.log'
    logger = setup_logger(log_path)

    # Parse the input file
    df_msn_data, df_prec_data, file_props = read_lcmsms(
        input_path=input_path,
        store_precursors=store_precursors,
        pwiz_stats=pwiz_stats,
        verbose=verbose,
        assign_dformats=assign_dformats,
        logger=logger
    )

    # Write the parsed data to the output .hdf5 file
    if df_msn_data is not None:
        if not output_path:
            output_path = os.path.splitext(input_path)[0] + '.hdf5'

        parsed_lcmsms_to_hdf(
            output_path=output_path,
            file_props=file_props,
            df_msn_data=df_msn_data,
            df_prec_data=df_prec_data,
            logger=logger,
            num_peaks=num_peaks,
            num_prec_peaks=num_prec_peaks,
            compress_peaks_lvl=compress_peaks_lvl,
            compress_full_lvl=compress_full_lvl
        )

    # Delete input file
    if del_in:
        os.remove(input_path)


def read_lcmsms(
        input_path,
        logger,
        store_precursors=True,
        pwiz_stats=False,
        assign_dformats=True,
        verbose=False
    ):

    logger.info(f'Started processing {input_path}')

    # File properties that are to be stored in .hdf5
    file_props = {'name': os.path.basename(input_path)}

    # Load ms data file
    msdata = pyms.MSExperiment()
    try:
        pyms.FileHandler().loadExperiment(input_path, msdata)
    except Exception:
        logger.error('\nPARSING ERROR\n' + traceback.format_exc() + '\nPARSING ERROR')
        logger.info('INPUT FILE')
        with open(input_path, 'r') as f:
            for line in f.readlines():
                logger.info(line)
        logger.info('INPUT FILE')
        #write_to_hdf(args, file_props=None, df_msn_data=None, prec_spectra_data=None, logger=logger)
        return None, None, None

    file_props['Ordered RT'] = lcms.sorted_by_rt(msdata)
    if not file_props['Ordered RT']:
        lcms.sort_by_rt(msdata)

    # Get order of spectra property
    order_of_spectra = lcms.get_order_of_spectra(msdata)
    file_props['MSLevelOrder'] = order_of_spectra.name

    # If order of spectra is invalid or insufficient do not continue the processing
    if order_of_spectra in [lcms.MSLevelsOrder.INVALID, lcms.MSLevelsOrder.EMPTY,
                            lcms.MSLevelsOrder.SINGLE_MS1, lcms.MSLevelsOrder.UNIFORM_MS1]:
        logger.info(f'Not processing the file because of {order_of_spectra}')
        #write_to_hdf(args, file_props=file_props, df_msn_data=None, prec_spectra_data=None, logger=logger)
        return None, None, None

    # Update file_props dict with instrument properties
    file_props.update(lcms.get_instrument_props(msdata))

    # Dict mapping ms_level to the last spectrum of this level
    # e.g. {1: pyopenms.Spectrum, 2: pyopenms.Spectrum}
    prev_spectra = {}
    prev_spectrum = None
    prec_spectra_data = {'peak list': [], 'RT': [], 'scan id': [], 'ion injection time': []}

    ms1_n, msn_n = 0, 0
    spectra_data = []
    problems, prec_problems = Counter(), Counter()
    for scan_id, spectrum in enumerate(msdata):

        spectrum_data = {}

        ms_level = spectrum.getMSLevel()
        rt = spectrum.getRT()

        # Ion injection time
        acq_info = spectrum.getAcquisitionInfo()
        inject_t = acq_info[0].getMetaValue('MS:1000927') if acq_info is not None and acq_info.size() > 0 else -1

        if ms_level == 1:
            ms1_n += 1

        # Get precursor spectrum (last spectrum with lower MS level)
        if prev_spectrum is not None:
            prev_spectra[prev_spectrum.getMSLevel()] = {
                # id - position in prec_spectra_data
                'peak list': prev_spectrum, 'RT': rt, 'scan id': scan_id, 'id': None, 'ion injection time': inject_t
            }
        prev_spectrum = spectrum

        # Consider only MSn spectra, where n > 1
        if ms_level > 1:

            msn_n += 1

            # I. Process peak list

            peaks = spectrum.get_peaks()
            if not peaks:
                continue
            mzs = peaks[0]
            intensities = peaks[1]

            peak_list = su.process_peak_list(np.array([mzs, intensities]), sort_mzs=True)

            # II. Check that the peak list is valid

            spec_type = lcms.get_spectrum_type(spectrum)
            problems_list = su.is_valid_peak_list(peak_list, return_problems_list=True, relative_intensities=False)
            if problems_list:
                if verbose:
                    logger.error('Errors in MSn scan {}: {} ({})'.format(
                        scan_id,
                        json.dumps(problems_list),
                        spec_type
                    ))
                for p in problems_list:
                    problems[p] += 1
                continue
            else:
                spectrum_data['peak list'] = peak_list

            # III. Collect peak list metadata

            # > MS level
            spectrum_data['MS level'] = ms_level

            # > Retention time
            spectrum_data['RT'] = rt

            # > Ion injection time
            spectrum_data['ion injection time'] = inject_t

            # > Polarity
            polarity = spectrum.getInstrumentSettings().getPolarity()
            if polarity == pyms.IonSource.Polarity.POSITIVE:
                spectrum_data['positive polarity'] = 1
            elif polarity == pyms.IonSource.Polarity.NEGATIVE:
                spectrum_data['positive polarity'] = 0
            else:
                spectrum_data['positive polarity'] = -1

            # > Spectrum type (profile/centroid)
            spectrum_data['type'] = spec_type.value
            # spectrum_data['type estim'] = lcms.estimate_peak_list_type(peak_list, to_int=True)

            # > Scan definition string (for Thermo instruments)
            scan_def = spectrum.getMetaValue('filter string')
            spectrum_data['def str'] = scan_def if scan_def is not None else ''

            # Analyze precursor properties and precursor spectrum
            precursors = spectrum.getPrecursors()
            # NOTE: is there at least one case where len(precursors) > 1?
            if len(precursors) > 1:
                logger.info(f'len(precursors) > 1 at scan {scan_id}')
            if precursors and precursors[0] is not None:

                # > Charge
                spectrum_data['charge'] = precursors[0].getCharge()

                # > Precursor m/z
                spectrum_data['precursor mz'] = precursors[0].getMZ()

                # > Isolation window offsets
                spectrum_data['window lo'] = precursors[0].getIsolationWindowLowerOffset()
                spectrum_data['window uo'] = precursors[0].getIsolationWindowUpperOffset()

                # > Activation energy
                spectrum_data['energy'] = precursors[0].getActivationEnergy()

                # > Precursor spectrum (peak list, scan id, RT)
                if store_precursors and ms_level - 1 in prev_spectra:

                    prec_spectrum = prev_spectra[ms_level - 1]['peak list']

                    if prev_spectra[ms_level - 1]['id'] is None:
                        prec_peaks = prec_spectrum.get_peaks()
                        prec_pl = su.process_peak_list(np.array([prec_peaks[0], prec_peaks[1]]), sort_mzs=True)
                        prec_problems_list = su.is_valid_peak_list(prec_pl, return_problems_list=True,
                                                                   relative_intensities=False)
                        if prec_problems_list:
                            if verbose:
                                logger.error('Errors in precursor scan {}: {} ({})'.format(
                                    prev_spectra[ms_level - 1]["scan id"],
                                    json.dumps(prec_problems_list),
                                    spec_type
                                ))
                            for p in prec_problems_list:
                                prec_problems[p] += 1
                            continue
                        else:
                            prev_spectra[ms_level - 1]['id'] = len(prec_spectra_data['peak list'])
                            prec_spectra_data['peak list'].append(prec_pl)
                            prec_spectra_data['scan id'].append(prev_spectra[ms_level - 1]['scan id'])
                            prec_spectra_data['RT'].append(prev_spectra[ms_level - 1]['RT'])
                            prec_spectra_data['ion injection time'].append(
                                prev_spectra[ms_level - 1]['ion injection time']
                            )

                    spectrum_data['precursor id'] = prev_spectra[ms_level - 1]['id']

            if assign_dformats:
                spectrum_data['dformat'] = dformats.assign_dformat(
                    spec=peak_list,
                    prec_mz=spectrum_data['precursor mz'],
                    tbxic_stdev=file_props['TBXICs median stdev'],
                    charge=spectrum_data['charge'],
                    mslevel=ms_level,
                )
            spectra_data.append(spectrum_data)

    if pwiz_stats:
        logger.info(f'ProteoWizard conversion statistics: {json.dumps(lcms.get_pwiz_stats(msdata))}')

    logger.info(f'Num. of MS1 spectra: {ms1_n}')
    logger.info('Collected {} from {} total num. of MSn spectra'.format(
        len(spectra_data),
        msn_n
    ))

    # Create dataframe out of MSn and MS1 data
    df_msn_data = pd.DataFrame(spectra_data)
    df_prec_data = pd.DataFrame(prec_spectra_data)

    if verbose:
        logger.info(f'File properties: {json.dumps(file_props)}')

    logger.info(f'Spectra problems: {json.dumps(problems)}')
    logger.info(f'Precursor spectra problems: {json.dumps(prec_problems)}')

    if len(spectra_data) > 0:
        if verbose:
            logger.info(f'Positive polarity: {df_msn_data["positive polarity"].value_counts().to_json()}')
            logger.info(f'Charge: {df_msn_data["charge"].value_counts().to_json()}')
            logger.info(f'Type hist: {df_msn_data["type"].value_counts().to_json()}')
            # logger.info(f'Type estim hist: {df_msn_data["type estim"].value_counts().to_json()}')
        return df_msn_data, df_prec_data, file_props
    else:
        logger.warning('No MSn spectra collected, not storing .hdf5 file.')
        return None, None, None

    logger.info(f'Finished processing {input_path}')


def parsed_lcmsms_to_hdf(
        output_path,
        file_props,
        df_msn_data,
        df_prec_data,
        logger,
        num_peaks=None,
        num_prec_peaks=None,
        compress_peaks_lvl=3,
        compress_full_lvl=3,
    ):
    # Create a new .hdf5 file to store the data
    with h5py.File(output_path, 'w') as hdf_file:

        # Fill hdf5 attributes with the file properties
        if file_props is not None:
            for k, v in file_props.items():
                hdf_file.attrs[k] = v if v is not None else -1

        # Create hdf5 datasets for peak lists and their metadata
        if df_msn_data is not None:

            # Fill NaN values with -1
            pd.set_option('future.no_silent_downcasting', True)  # TODO: is there a better solution?
            df_msn_data = df_msn_data.fillna(-1)

            # Find maximum number of peaks in peak list
            peaks_n = df_msn_data['peak list'].apply(lambda pl: pl.shape[1])
            max_peaks_n = peaks_n.max()

            # Trim peak lists to num_peaks peaks if num_peaks is set
            if num_peaks and max_peaks_n > num_peaks:
                logger.info('Trimming MSn peaks to {} peaks (max was {}, mean was {}, median was {}).'.format(
                    num_peaks,
                    max_peaks_n,
                    peaks_n.mean(),
                    peaks_n.median()
                ))
                max_peaks_n = num_peaks
                df_msn_data['peak list'] = df_msn_data['peak list'].apply(lambda pl: su.trim_peak_list(pl, max_peaks_n))

            # Pad peak lists with 0 up to the maximum
            df_msn_data['peak list'] = df_msn_data['peak list'].apply(lambda pl: su.pad_peak_list(pl, max_peaks_n))
            peak_lists = np.stack(df_msn_data['peak list'])

            # Create datasets for MSn data
            msn_group = hdf_file.create_group('MSn data')
            msn_group.create_dataset('mzs', data=peak_lists[:, 0, :], dtype='f8', compression='gzip',
                                     compression_opts=compress_peaks_lvl)
            msn_group.create_dataset('intensities', data=peak_lists[:, 1, :], dtype='f4', compression='gzip',
                                     compression_opts=compress_peaks_lvl)
            msn_group.create_dataset('MS level', data=df_msn_data['MS level'], dtype='i1',
                                     compression=compress_full_lvl)
            msn_group.create_dataset(RT, data=df_msn_data['RT'], dtype='f4', compression=compress_full_lvl)
            msn_group.create_dataset(CHARGE, data=df_msn_data['charge'], dtype='i1',
                                     compression=compress_full_lvl)
            msn_group.create_dataset('positive polarity', data=df_msn_data['positive polarity'], dtype='i1',
                                     compression=compress_full_lvl)
            msn_group.create_dataset(PRECURSOR_MZ, data=df_msn_data['precursor mz'], dtype='f4',
                                     compression=compress_full_lvl)
            msn_group.create_dataset('window lo', data=df_msn_data['window lo'], dtype='f4',
                                     compression=compress_full_lvl)
            msn_group.create_dataset('window uo', data=df_msn_data['window uo'], dtype='f4',
                                     compression=compress_full_lvl)
            msn_group.create_dataset('energy', data=df_msn_data['energy'], dtype='f4',
                                     compression=compress_full_lvl)
            msn_group.create_dataset('ion injection time', data=df_msn_data['ion injection time'], dtype='f4',
                                     compression=compress_full_lvl)
            msn_group.create_dataset('type', data=df_msn_data['type'], dtype='i1', compression=compress_full_lvl)
            # msn_group.create_dataset('type estim', data=df_msn_data['type estim'], dtype='i1',
            #                          compression=compress_full_lvl)
            msn_group.create_dataset('def str', data=df_msn_data['def str'], dtype=h5py.string_dtype('utf-8', None),
                                     compression=compress_full_lvl)
            if 'dformat' in df_msn_data:
                # Char dtype
                msn_group.create_dataset('dformat', data=df_msn_data['dformat'], dtype='S1',
                                         compression=compress_full_lvl)

            # Create hdf5 datasets for the data of precursor spectra
            if df_prec_data is not None:

                # Create dataset with pointers from MSn spectra to their precursors
                msn_group.create_dataset('precursor id', data=df_msn_data['precursor id'], dtype='i4',
                                         compression=compress_full_lvl)

                # Find maximum number of peaks in peak list and pad peak lists with 0 up to the maximum
                prec_peak_lists = pd.Series(df_prec_data['peak list'])
                prec_peaks_n = prec_peak_lists.apply(lambda pl: pl.shape[1])
                max_peaks_n = prec_peaks_n.max()
                if num_prec_peaks and max_peaks_n > num_prec_peaks:
                    logger.info('Trimming precursor peaks to {} (max was {}, mean was {}, median was {}).'.format(
                        num_prec_peaks,
                        max_peaks_n,
                        prec_peaks_n.mean(),
                        prec_peaks_n.median()
                    ))
                    max_peaks_n = num_prec_peaks
                    prec_peak_lists = prec_peak_lists.apply(lambda pl: su.trim_peak_list(pl, max_peaks_n))
                prec_peak_lists = prec_peak_lists.apply(lambda pl: su.pad_peak_list(pl, max_peaks_n))
                prec_peak_lists = np.stack(prec_peak_lists)

                # Create datasets
                prec_group = hdf_file.create_group('precursor data')
                prec_group.create_dataset('mzs', data=prec_peak_lists[:, 0, :], dtype='f8', compression='gzip',
                                          compression_opts=compress_peaks_lvl)
                prec_group.create_dataset('intensities', data=prec_peak_lists[:, 1, :], dtype='f4', compression='gzip',
                                          compression_opts=compress_peaks_lvl)
                prec_group.create_dataset('RT', data=df_prec_data['RT'], dtype='f4',
                                          compression=compress_full_lvl)
                prec_group.create_dataset('ion injection time', data=df_prec_data['ion injection time'],
                                          dtype='f4', compression=compress_full_lvl)
                prec_group.create_dataset('scan id', data=df_prec_data['scan id'], dtype='i4',
                                          compression=compress_full_lvl)
                prec_group.create_dataset('type', data=df_msn_data['type'], dtype='i1',
                                          compression=compress_full_lvl)
                prec_group.create_dataset('def str', data=df_msn_data['def str'],
                                          dtype=h5py.string_dtype('utf-8', None), compression=compress_full_lvl)


def downloadpublicdata_to_hdf5s(downloads_log: Path, del_in=True, verbose=False) -> None:
    """
    Convert downloaded LC-MS/MS data (e.g., .mzML or .mzXML) to .hdf5 format.

    Args:
    downloads_log (Path): Path to the log file from `downloadpublicdata` containing information about downloaded files.
    del_in (bool, optional): Whether to delete the input files after conversion. Defaults to True.
    verbose (bool, optional): Whether to print additional information during the conversion. Defaults to True.
    """
    # Open `downloadpublicdata` downloads log
    df = pd.read_csv(downloads_log, sep='\t')

    # Covert downloaded files to .hdf5 format
    for _, row in df.iterrows():
        in_pth = Path(row['target_path'])
        if 'ERROR' not in row['status'] and in_pth.exists():
            if verbose:
                print(f"{row['usi']} was successfully downloaded.")
            out_pth = prepend_to_stem(in_pth, row['usi'].split(':')[1]).with_suffix('.hdf5')
            lcmsms_to_hdf5(input_path=in_pth, output_path=out_pth, verbose=verbose, del_in=del_in)
        else:
            if verbose:
                print(f"Skipping {row['usi']} because it was not downloaded.")


def merge_lcmsms_hdf5s(
        in_pths: Union[Path, Iterable[Path]],
        out_pth: Path,
        dformat: str = 'A',
        store_acc_est: bool = True,
        verbose: bool = True
    ):
    """
    Merge .hdf5 files generated with `lcmsms_to_hdf5`.

    Args:
    in_pths (Union[Path, Iterable[Path]]): Path to the directory with .hdf5 files or an iterable of .hdf5 files.
    out_pth (Path): Path to the output .hdf5 file.
    store_acc_est (bool, optional): Whether to store the instrument accuracy estimate in the output file. Defaults to True.
    verbose (bool, optional): Whether to print additional information during the merging. Defaults to True.
    """

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # Needed to avoid huge file closing bug from h5py

    if not isinstance(in_pths, Path):
        in_pths = Path(in_pths)

    if in_pths.is_dir():
        in_pths = in_pths.glob('*.hdf5')

    dformat_filters = dformats.DataFormatBuilder(dformat).get_dformat()

    f_out = h5py.File(out_pth, 'w')
    first_file = True
    for in_pth in tqdm(list(in_pths), desc='Merging .hdf5 files', disable=not verbose):
        try:
            with h5py.File(in_pth, 'a') as f_in:

                # Check that the input file has a correct order of spectra and enough number of spectra
                n_spectra = f_in['MSn data']['mzs'].shape[0]
                if not f_in.attrs['Ordered RT']:
                    continue
                if n_spectra < dformat_filters.min_file_spectra:
                    continue

                # Compute dformats for MS/MS spectra if needed
                if 'dformat' not in f_in['MSn data']:
                    # TODO: in dformat calculation report invoked filters for further logging
                    # Calculate dformat if not present
                    dformat_array = np.empty(n_spectra, dtype=h5py.string_dtype())
                    specs = np.stack([f_in['MSn data']['mzs'][:], f_in['MSn data']['intensities'][:]], axis=1)
                    prec_mzs = f_in['MSn data']['precursor mz'][:]
                    for i in range(n_spectra):
                        dformat_array[i] = dformats.assign_dformat(specs[i], prec_mz=prec_mzs[i])
                    f_in['MSn data'].create_dataset('dformat', data=dformat_array)

                # Subset spectra accordign to the dformat
                idx = np.where(f_in['MSn data']['dformat'][:].astype(str) == dformat)[0]
                n_spectra = idx.shape[0]

                # Check that the input file has enough spectra after subsetting
                if idx.shape[0] < dformat_filters.min_file_spectra:
                    continue

                # Trim or pad spectra to the maximum number of peaks
                spectra = np.stack([f_in['MSn data']['mzs'][:][idx], f_in['MSn data']['intensities'][:][idx]], axis=1)
                if spectra.shape[2] > dformat_filters.max_peaks_n:
                    spectra = su.trim_peak_list(spectra, dformat_filters.max_peaks_n)
                elif spectra.shape[2] < dformat_filters.max_peaks_n:
                    spectra = su.pad_peak_list(spectra, dformat_filters.max_peaks_n)

                # Define datasets to store
                datasets = [
                    (SPECTRUM, spectra, f_in['MSn data']['mzs'].dtype),
                    (FILE_NAME, [in_pth.stem] * n_spectra, h5py.string_dtype())
                ]
                datasets.extend(
                    [(n, f_in['MSn data'][n][:][idx], f_in['MSn data'][n].dtype) for n in [CHARGE, 'precursor mz', RT]]
                )
                if store_acc_est:
                    datasets.append(
                        ('instrument accuracy est.', np.repeat(f_in.attrs['TBXICs median stdev'], n_spectra), np.float32)
                    )

                # Store datasets in the output file
                for name, data, dtype in datasets:
                    if first_file:
                        f_out.create_dataset(
                            name, data=data, shape=data.shape if isinstance(data, np.ndarray) else (len(data),),
                            maxshape=(None, *data.shape[1:]) if isinstance(data, np.ndarray) else (None,),
                            dtype=dtype
                        )
                    else:
                        data_len = data.shape[0] if isinstance(data, np.ndarray) else len(data)
                        f_out[name].resize(f_out[name].shape[0] + data_len, axis=0)
                        f_out[name][-data_len:] = data
                first_file = False
        except:
            print(f'Skipping {f_in}.')



def lsh_subset(in_pth, dformat, n_hplanes=None, bin_size=1, max_specs_per_lsh=None, seed=333):
    """
    Subset the input .hdf5 file using Locality Sensitive Hashing (LSH) algorithm.

    Args:
        input_path (str): Path to the input file.
        dformat (DataFormatBuilder): Data format builder object.
        n_hplanes (int, optional): Number of hyperplanes for LSH. Defaults to None.
        bin_size (float, optional): Bin size for LSH. Defaults to 1.
        max_specs_per_lsh (int, optional): Maximum number of spectra per LSH. Defaults to None.
        seed (int, optional): Random seed for LSH initialization and selection. Defaults to 333.
    """

    assert n_hplanes is not None or max_specs_per_lsh is not None
    out_suffix = ''
    if max_specs_per_lsh is not None:
        out_suffix += str(max_specs_per_lsh)
    if n_hplanes is not None:
        out_suffix += '_hplanes' + str(n_hplanes)
    out_pth = append_to_stem(in_pth, out_suffix, sep='')

    dformat = dformats.DataFormatBuilder(dformat).get_dformat()
    logger = setup_logger(out_pth.with_suffix('.log'))
    tqdm_logger = TqdmToLogger(logger)

    with h5py.File(in_pth, 'r') as f_in:

        logger.info('Opening input file...')

        data = {}
        for k in f_in.keys():
            logger.info(f'Loading dataset "{k}" of shape {f_in[k].shape} into memory...')
            data[k] = f_in[k][:]

        with h5py.File(out_pth, 'w') as f_out:

            if 'lsh' not in data.keys():

                logger.info(f'Computing LSHs for {data[SPECTRUM].shape}...')

                lsh = BatchedPeakListRandomProjection(
                    subbatch_size=10_000, max_mz=dformat.max_mz,
                    bin_step=bin_size, n_hyperplanes=n_hplanes, seed=seed
                )

                data['lsh'] = lsh.compute(data[SPECTRUM], logger=logger)

            if max_specs_per_lsh is not None:

                logger.info('Deduplicating spectra by LSHs...')
                lshs = data['lsh']
                np.random.seed(seed)

                if max_specs_per_lsh == 1:
                    logger.info('Keeping only unique LSHs...')
                    _, filtered_idx = np.unique(lshs, return_index=True)
                else:
                    logger.info(f'Keeping {max_specs_per_lsh} spectra per LSHs...')
                    lshs_unique, lshs_counts = np.unique(lshs, return_counts=True)

                    filtered_idx = []
                    non_filtered_lshs = []
                    for lsh, count in tqdm(zip(lshs_unique, lshs_counts), file=tqdm_logger):
                        if count > max_specs_per_lsh:
                            idx = np.where(lshs == lsh)[0]
                            filtered_idx.extend(np.random.choice(idx, max_specs_per_lsh, replace=False))
                        else:
                            non_filtered_lshs.append(lsh)
                    filtered_idx.extend(np.where(np.isin(lshs, non_filtered_lshs))[0])

                    filtered_idx = np.array(filtered_idx)
                logger.info(f'Keeping {filtered_idx.shape[0]} / {data["lsh"].shape[0]} deduplicated spectra.')
            else:
                filtered_idx = np.arange(data['lsh'].shape[0])
                logger.info(f'Keeping all {filtered_idx.shape[0]} spectra.')

            for k in data.keys():
                logger.info(f'Adding subdataset "{k}" corresponding to unique LSHs to {out_pth}...')
                f_out.create_dataset(name=k, data=data[k][filtered_idx], shape=(filtered_idx.shape[0], *data[k].shape[1:]),
                                     dtype=data[k].dtype)

    logger.info('Done.')
    return out_pth


def read_json_spec(pth, peaks_key="peaks", prec_mz_key="precursor_mz"):
    spec = json.load(open(pth))
    return {
        SPECTRUM: np.array(spec[peaks_key]).T,
        PRECURSOR_MZ: spec[prec_mz_key]
    }


def save_nist_like_df_to_mgf(df, out_pth: Path, remove_mol_info=False, all_mplush_adducts=False):
    spectra = []
    for i, row in tqdm(df.iterrows()):
        spec = {
            'm/z array': row['PARSED PEAKS'][0],
            'intensity array': row['PARSED PEAKS'][1],
            'params': {
                'adduct': row['PRECURSOR TYPE'] if not all_mplush_adducts else '[M+H]+',
                'collision energy': int(row['COLLISION ENERGY']),
                'pepmass': row['PRECURSOR M/Z'],
                'charge': row['CHARGE'],
            }
        }

        spec_name = str(i) + '_' + row['NAME'] if 'NAME' in row else str(i)
        if not remove_mol_info:
            spec['params'].update({
                'name': spec_name,
                'formula': row['FORMULA'],
                'smiles': row['SMILES']
            })
        else:
            spec['params']['name'] = f'{spec_name}_{row["FORMULA"]}_{row["SMILES"]}'

        spectra.append(spec)

    pyteomics.mgf.write(spectra, str(out_pth))


def savefig(name, path, extension='pdf'):
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    path = path / name
    path = path.with_suffix(extension)

    plt.savefig(path, bbox_inches='tight')


@contextlib.contextmanager
def suppress_output():
    new_stdout = std_io.StringIO()
    new_stderr = std_io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = new_stdout, new_stderr
    try:
        yield new_stdout, new_stderr
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def wandb_import(project_name, entity_name='roman-bushuiev', tags={}, run_name_suffixes=None):
    if isinstance(run_name_suffixes, str):
        run_name_suffixes = [run_name_suffixes]
    api = wandb.Api()
    val_dfs = []
    cfgs = []
    for r in tqdm(api.runs(entity_name + '/' + project_name, include_sweeps=False)):
        if any([s in r.name for s in run_name_suffixes]):
            cfgs.append(r.config)
            val_dfs.append(pd.DataFrame(list(r.scan_history())))
    cfgs = pd.DataFrame(cfgs)
    return val_dfs, cfgs


def ftp_to_msv_id(ftp):
    return ftp.split('/')[0]


def clean_ftps(ftps: dict, verbose=True):
    """
    Cleans the dict of MassIVE ftps (see code comments 1., 2., 3. for details).
    :param ftps: keys - ftps, values - corresponding file sizes.
    """

    if verbose:
        print('Original num. of ftps:', len(ftps))

    # 1. Only .mzMLs and .mzXMLs (e.g. no archives)
    ftps = {f: ftps[f] for f in ftps if os.path.splitext(f)[1].lower() in ['.mzml', '.mzxml']}
    if verbose:
        print('Only .mzMLs and .mzXMLs (no archives):', len(ftps))

    # 2. No duplicate filenames within the same MassIVE ID
    # Sort ftps by extension and belonging to "update folder", so all .mzXMLs go after .mzMLs and .mzMLs from
    # 'update' folders are prioritized
    ftps_keys = sorted(ftps.keys(), key=lambda f: os.path.splitext(f)[1].lower() + str('update' not in f))
    # Iterate over sorted ftps and keep only unique ones (up to MSV ID and file base name)
    ftp_fps = set()
    ftps_clean = []
    for f in ftps_keys:
        ftp_fp = ftp_to_msv_id(f) + os.path.splitext(os.path.basename(f))[0]
        if ftp_fp not in ftp_fps:
            ftp_fps.add(ftp_fp)
            ftps_clean.append(f)
    if verbose:
        print('No duplicate filenames:', len(ftps_clean))

    # 3. Percent encode reserved URI characters (see https://en.wikipedia.org/wiki/Percent-encoding)
    ftps_clean = {urlparse.quote(f): ftps[f] for f in ftps_clean}

    return ftps_clean


def compress_hdf(hdf_pth, out_pth=None, compression='gzip', compression_opts=4):
    if not isinstance(hdf_pth, Path):
        hdf_pth = Path(hdf_pth)
    if out_pth is None:
        out_pth = append_to_stem(hdf_pth, 'compressed')
    with h5py.File(hdf_pth, 'r') as f:
        with h5py.File(out_pth, 'w') as f_out:
            for k in f.keys():
                print(f'Compressing "{k}" dataset...')
                f_out.create_dataset(
                    k, data=f[k][:], shape=f[k].shape, dtype=f[k].dtype,
                    compression=compression, compression_opts=compression_opts
                )


def sample_hdf(hdf_pth, n_samples, out_pth=None, seed=333, compression='gzip', compression_opts=4):

    with h5py.File(hdf_pth, 'r') as f:

        dataset_len = set(len(f[k]) for k in f.keys())
        if len(dataset_len) != 1:
            raise ValueError("Not all datasets have the same length")
        dataset_len = dataset_len.pop()

        if n_samples > dataset_len:
            raise ValueError(f"Cannot sample {n_samples} samples from a dataset of length {dataset_len}")

        if out_pth is None:
            out_pth = append_to_stem(hdf_pth, f'rand{n_samples}')

        print(f'Sampling {n_samples} random spectra from {hdf_pth}...')
        np.random.seed(seed)
        sample_idx = np.random.choice(dataset_len, n_samples, replace=False)
        sample_idx = np.sort(sample_idx)

        with h5py.File(out_pth, 'w') as f_out:
            for k in f.keys():
                print(f'Sampling "{k}" dataset...')
                f_out.create_dataset(
                    k, data=f[k][:][sample_idx], shape=(n_samples, *f[k].shape[1:]), dtype=f[k].dtype,
                    compression=compression, compression_opts=compression_opts
                )

        return out_pth


class ChunkedHDF5File:
    def __init__(self, file_paths):
        """
        Initialize the ChunkedHDF5File with a list of HDF5 file paths.

        Args:
        - file_paths (list of Path): Paths to the HDF5 files.
        """
        self.file_paths = sorted(file_paths)
        self.files = [h5py.File(p, 'r') for p in self.file_paths]
        self.datasets = list(self.files[0].keys())

        # Assume all datasets have the same length along the first dimension
        self.dataset_lengths = [f[self.datasets[0]].shape[0] for f in self.files]
        self.total_length = sum(self.dataset_lengths)

    def keys(self):
        """Return a list of dataset names."""
        return self.datasets

    def close(self):
        """Close all files."""
        try:
            for f in self.files:
                f.close()
        except Exception as e:
            print(f'Ignored error while closing chunked HDF5 files: {e}')

    def __del__(self):
        """Ensure all files are properly closed when the object is deleted."""
        self.close()

    def __getitem__(self, key):
        """
        Allows for standard indexing notation to access datasets and their elements.

        Args:
        - key (str): The name of the dataset.

        Returns:
        - An object to access elements of the specified dataset.
        """
        if key not in self.datasets:
            raise ValueError(f"Dataset {key} not found in the files.")
        return ChunkedDatasetAccessor(self, key)

class ChunkedDatasetAccessor:
    def __init__(self, parent, dataset_name):
        self.parent = parent
        self.dataset_name = dataset_name

    @property
    def shape(self):
        """Return the shape of the dataset."""
        first_shape = self.parent.files[0][self.dataset_name].shape
        return (self.parent.total_length,) + first_shape[1:]

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.parent.total_length)
            if step != 1:
                raise ValueError("Step size other than 1 is not supported.")
            return self._get_slice(start, stop)
        elif isinstance(index, int) or isinstance(index, np.integer):
            if index < 0:
                index += self.parent.total_length
            return self._get_element(index)
        else:
            raise TypeError("Invalid argument type.")

    def _get_element(self, index):
        """Get a specific element from the dataset by index."""
        if index < 0 or index >= self.parent.total_length:
            raise IndexError(f"Index {index} is out of bounds for dataset with length {self.parent.total_length}.")

        cumulative_length = 0
        for i, length in enumerate(self.parent.dataset_lengths):
            if cumulative_length + length > index:
                chunk_index = i
                internal_index = index - cumulative_length
                break
            cumulative_length += length

        return self.parent.files[chunk_index][self.dataset_name][internal_index]

    def _get_slice(self, start, stop):
        """Get a slice of the dataset."""
        if start < 0 or stop > self.parent.total_length or start >= stop:
            raise IndexError(f"Invalid slice range [{start}:{stop}] for dataset with length {self.parent.total_length}.")

        result = []
        cumulative_length = 0
        for i, length in enumerate(self.parent.dataset_lengths):
            if cumulative_length + length > start:
                chunk_index = i
                internal_start = start - cumulative_length
                while start < stop:
                    if chunk_index >= len(self.parent.files):
                        break
                    chunk_end = min(stop - start, self.parent.dataset_lengths[chunk_index] - internal_start)
                    result.append(self.parent.files[chunk_index][self.dataset_name][internal_start:internal_start + chunk_end])
                    start += chunk_end
                    chunk_index += 1
                    internal_start = 0
                break
            cumulative_length += length

        return np.concatenate(result)
