import logging
import sys
import pickle
import json
import os
import h5py
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
from functools import cache
from matchms.importing import load_from_mgf
from pathlib import Path
from matchms import Spectrum
from typing import Tuple, List, Optional
from itertools import groupby
from tqdm import tqdm
import dreams.utils.spectra as su
import dreams.utils.misc as utils
from dreams.definitions import *


LOG_INFO_PREF_LEN = len('2022-09-20 21:04:45,234 | INFO |')


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


# def read_mgf(pth, skip_empty_spectra=False, charge_col='charge', prec_mz_col='precursor_mz', smiles_col='smiles'):
#     spectra = load_from_mgf(str(pth))
#     df = []
#     for s in tqdm(spectra):
#
#         # Read peak list data
#         row = {'PARSED PEAKS': s.peaks.to_numpy.T}
#         if skip_empty_spectra and row['PARSED PEAKS'].size == 0:
#             continue
#
#         # Read metadata
#         row.update(s.metadata)
#
#         # Rename major columns
#         if charge_col:
#             row['CHARGE'] = row.pop(charge_col)
#         if prec_mz_col:
#             row['PRECURSOR M/Z'] = row.pop(prec_mz_col)
#
#         df.append(row)
#     df = pd.DataFrame(df)
#
#     # Obtain rdkit molecules from SMILES
#     if smiles_col:
#         df['ROMol'] = df[smiles_col].apply(Chem.MolFromSmiles)
#
#     return df


def read_textual_ms_format(pth, spectrum_end_line, name_value_sep, prec_mz_name, charge_name=None, adduct_name=None,
                           ignore_line_prefixes=()):
    # TODO: this is very raw and dirty.

    # Two numbers separated with a white space
    peak_pattern = re.compile(r'\b([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\b')
    # A word followed by an arbitrary string separated with `name_value_sep`
    attr_pattern = re.compile(rf'^\s*([A-Z_]+){name_value_sep}(.*)\s*$')
    attr_mapping = {prec_mz_name: PRECURSOR_MZ, charge_name: CHARGE, adduct_name: ADDUCT}

    data = []
    with open(pth, 'r') as f:
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


def read_msp(pth):
    return read_textual_ms_format(
        pth=pth,
        spectrum_end_line='',
        name_value_sep=': ',
        prec_mz_name='PRECURSORMZ',
        ignore_line_prefixes=('#',)
    )


def read_mgf(pth):
    return read_textual_ms_format(
        pth=pth,
        spectrum_end_line='END IONS',
        name_value_sep='=',
        prec_mz_name='PEPMASS',
        charge_name='CHARGE',
        ignore_line_prefixes=('#',)
    )


def read_mzml(pth: Path, progress_bar=True, ):
    exp = oms.MSExperiment()
    if pth.suffix.lower() == '.mzml':
        oms.MzMLFile().load(str(pth), exp)
    elif pth.suffix.lower() == '.mzxml':
        oms.MzXMLFile().load(str(pth), exp)
    else:
        raise ValueError(f'Unsupported file extension: {pth.suffix}.')

    df = []
    for i, spec in enumerate(tqdm(exp, desc=f'Reading {pth.name}', disable=not progress_bar)):
        if spec.getMSLevel() != 2:
            continue

        scan_i = re.search(r'scan=(\d+)', spec.getNativeID())
        scan_i = int(scan_i.group(1)) if scan_i else i + 1

        peak_list = np.stack(spec.get_peaks())
        spec_problems = su.is_valid_peak_list(peak_list, relative_intensities=False, return_problems_list=True, verbose=progress_bar)
        if spec_problems:
            print(f'Skipping spectrum {i} in {pth.name} with problems: {spec_problems}.')
            continue

        prec = spec.getPrecursors()
        if len(prec) != 1:
            continue
        prec = prec[0]

        df.append({
            'FILE NAME': pth.name,
            'SCAN NUMBER': scan_i,
            'PARSED PEAKS': peak_list,
            'PRECURSOR M/Z': prec.getMZ(),
            'RT': spec.getRT(),
            'CHARGE': prec.getCharge(),
        })

    df = pd.DataFrame(df)
    return df


def save_nist_like_df_to_mgf(df, out_pth: Path, remove_mol_info=False):
    spectra = []
    for i, row in tqdm(df.iterrows()):
        spec = {
            'm/z array': row['PARSED PEAKS'][0],
            'intensity array': row['PARSED PEAKS'][1],
            'params': {
                'adduct': row['PRECURSOR TYPE'],
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


def wandb_import(project_name, entity_name='roman-bushuiev', tags={}, run_name_suffix=None, block_list_names=[],
                 include_list_names=[]):
    api = wandb.Api()
    val_dfs = []
    cfgs = []
    for r in tqdm(api.runs(entity_name + '/' + project_name, include_sweeps=False)):
        if run_name_suffix not in r.name: #not r.name in include_list_names:
            continue
            # if tags and not set(r.tags).intersection(tags):
            #     continue
            # if run_name_suffix and run_name_suffix not in r.name:
            #     continue
            # if r.name in block_list_names:
            #     continue
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


def compress_hdf(hdf_pth, out_pth=None, compression='gzip', compression_opts=9):
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
