import pandas as pd
import numpy as np
import typing as T
import dreams.utils.spectra as su
from abc import ABC


class DataFormat(ABC):
    """
    Abstract class for DataFormats.
    """

    # LC-MS quality filters
    min_file_spectra: int = NotImplementedError
    max_tbxic_stdev: float = NotImplementedError

    # MS/MS quality filters 
    max_ms_level: int = NotImplementedError
    min_peaks_n: int = NotImplementedError
    max_peaks_n: int = NotImplementedError
    min_charge: int = NotImplementedError
    max_charge: int = NotImplementedError
    min_intensity_ampl: float = NotImplementedError
    max_prec_mz: float = NotImplementedError
    max_mz: float = NotImplementedError
    high_intensity_thld: float = NotImplementedError

    # LSH parameters
    lsh_n_hplanes: int = NotImplementedError
    lsh_bin_size: float = NotImplementedError

    def val_spec(
            self,
            spec: np.ndarray,
            prec_mz: float,
            tbxic_stdev: T.Optional[float] = None,
            charge: T.Optional[int] = None,
            mslevel: T.Optional[int] = None,
            verbose: bool = False
        ) -> bool:

        if spec.shape[0] != 2:
            raise ValueError('The input array should have 2 rows (m/z and intensity).')

        # Check metadata
        if mslevel is not None and mslevel > self.max_ms_level:
            if verbose:
                print(f"MS level check failed. Expected <= {self.max_ms_level}, got {mslevel}")
            return False
        if prec_mz is not None and prec_mz > self.max_prec_mz:
            if verbose:
                print(f"Precursor m/z check failed. Expected <= {self.max_prec_mz}, got {prec_mz}")
            return False
        if tbxic_stdev is not None and tbxic_stdev > self.max_tbxic_stdev:
            if verbose:
                print(f"TBXIC stdev check failed. Expected <= {self.max_tbxic_stdev}, got {tbxic_stdev}")
            return False
        if charge is not None and (charge < self.min_charge or charge > self.max_charge):
            if verbose:
                print(f"Charge check failed. Expected between {self.min_charge} and {self.max_charge}, got {charge}")
            return False

        # Check number of peaks
        if spec.shape[1] < self.min_peaks_n or spec.shape[1] > self.max_peaks_n:
            if verbose:
                print(f"Number of peaks check failed. Expected between {self.min_peaks_n} and {self.max_peaks_n}, got {spec.shape[1]}")
            return False
        
        # Check mz range
        if su.max_mz(spec) > self.max_mz:
            if verbose:
                print(f"m/z range check failed. Expected <= {self.max_mz}, got {su.max_mz(spec)}")
            return False
        
        # Make relative intensities for the following checks
        spec = su.to_rel_intensity(spec)

        # Check intensity amplitude
        if su.intens_amplitude(spec) < self.min_intensity_ampl:
            if verbose:
                print(f"Intensity amplitude check failed. Expected >= {self.min_intensity_ampl}, got {su.intens_amplitude(spec)}")
            return False
        
        # Check number of high intensity peaks
        if su.num_high_peaks(spec, self.high_intensity_thld) < self.min_peaks_n:
            if verbose:
                print(f"Number of high intensity peaks check failed. Expected >= {self.min_peaks_n}, got {su.num_high_peaks(spec, self.high_intensity_thld)}")
            return False
        
        if verbose:
            print('All checks passed')

        return True


class DataFormatA(DataFormat):
    min_file_spectra = 3
    max_ms_level = 2
    min_peaks_n = 3
    max_peaks_n = 128
    min_charge = 1
    max_charge = 1
    min_intensity_ampl = 20.
    max_tbxic_stdev = 1e-4
    max_prec_mz = 1000.
    max_mz = 1000.
    high_intensity_thld = 0.1


class DataFormatA1(DataFormatA):
    min_peaks_n = 2


class DataFormatA2(DataFormatA):
    min_peaks_n = 2
    high_intensity_thld = 0.05
    min_intensity_ampl = 15


class DataFormatA3(DataFormatA):
    min_peaks_n = 2
    high_intensity_thld = 0.05
    min_intensity_ampl = 15
    min_charge = -1


class DataFormatB(DataFormat):
    min_file_spectra = 3
    max_ms_level = 2
    min_peaks_n = 3
    max_peaks_n = 128
    min_charge = 1
    max_charge = 1
    min_intensity_ampl = 20.
    max_tbxic_stdev = 1e-3
    max_prec_mz = 1500.
    max_mz = 1500.
    high_intensity_thld = 0.1


class DataFormatC(DataFormat):
    min_file_spectra = 3
    max_ms_level = 10
    min_peaks_n = 3
    max_peaks_n = 128
    min_charge = -1
    max_charge = 1
    min_intensity_ampl = 18.
    max_tbxic_stdev = 1e-3
    max_prec_mz = 1500.
    max_mz = 1500.
    high_intensity_thld = 0.1


class DataFormatBuilder:
    def __init__(self, dformat_name):
        if dformat_name == 'A':
            self.dformat = DataFormatA()
        elif dformat_name == 'B':
            self.dformat = DataFormatB()
        elif dformat_name == 'C':
            self.dformat = DataFormatC()
        elif dformat_name == 'A1':
            self.dformat = DataFormatA1()
        elif dformat_name == 'A2':
            self.dformat = DataFormatA2()
        elif dformat_name == 'A3':
            self.dformat = DataFormatA3()
        else:
            raise ValueError(f'{dformat_name} is not a valid data format.')

    def get_dformat(self):
        return self.dformat


def assign_dformat(spec: np.ndarray, **kwargs) -> str:
    for e in ['A', 'B', 'C']:
        dformat = DataFormatBuilder(e).get_dformat()
        if dformat.val_spec(spec, **kwargs):
            return e
    return '-'


### Legacy code

def to_A_format(df: pd.DataFrame, filter=True, trimming=False, reset_index=True, verbose=True, add_msn_col=True,
                filter_block_mask=None):
    return to_format(df, DataFormatA(), filter=filter, trimming=trimming, reset_index=reset_index, verbose=verbose,
                     add_msn_col=add_msn_col, filter_block_mask=filter_block_mask)


def to_format(df: pd.DataFrame, dformat: DataFormat, filter=True, trimming=False, reset_index=True, verbose=True,
              add_msn_col=True, filter_block_mask=None):
    """
    :param df: df with NIST-like columns
    :param filter_block_mask: if not None, a boolean mask with the same length as df. True entries are not filtered.
    """

    # Assert NIST-like columns
    assert all([c in df.columns for c in ['PARSED PEAKS', 'CHARGE', 'PRECURSOR M/Z']])

    # To relative intensities
    df['PARSED PEAKS'] = df['PARSED PEAKS'].apply(su.to_rel_intensity)

    # Filters
    if filter:
        if filter_block_mask is not None:
            df_filter_block = df[filter_block_mask].copy()
            df = df[~filter_block_mask].copy()
            if verbose: print(f'Num. of block list entries:', len(df_filter_block))

        if verbose: print('Initial size:', len(df))
        df = df[(df['CHARGE'].astype(int) >= dformat.min_charge) & (df['CHARGE'].astype(int) <= dformat.max_charge)]
        if verbose: print(f'{dformat.min_charge} <= charge <= {dformat.max_charge}:', len(df))
        df = df[df['PRECURSOR M/Z'].astype(float) <= dformat.max_prec_mz]
        if verbose: print(f'Precursor m/z <= {dformat.max_prec_mz}:', len(df))
        df = df[df['PARSED PEAKS'].apply(lambda pl: su.intens_amplitude(pl) >= dformat.min_intensity_ampl)]
        if verbose: print(f'Intensity amplitude >= {dformat.min_intensity_ampl}:', len(df))
        df = df[df['PARSED PEAKS'].apply(lambda pl: su.num_high_peaks(pl, dformat.high_intensity_thld) >= dformat.min_peaks_n)]
        if verbose: print(f'Num. high (>= {dformat.high_intensity_thld}) peaks >= {dformat.min_peaks_n}:', len(df))

        if filter_block_mask is not None:
            df = pd.concat([df, df_filter_block])

        if verbose: print('Filtered size:', len(df))

    # Trimming
    if trimming:
        df['PARSED PEAKS'] = df['PARSED PEAKS'].apply(lambda pl: su.trim_peak_list(pl, dformat.max_peaks_n))
    # elif filter:
    #     df = df[df['PARSED PEAKS'].apply(su.get_num_peaks) <= dformat.max_peaks_n]
    #     if verbose: print(f'Num. peaks <= {dformat.max_peaks_n}:', len(df))

    # Padding
    df['PARSED PEAKS'] = df['PARSED PEAKS'].apply(lambda pl: su.pad_peak_list(pl, dformat.max_peaks_n))

    # Add the column with MSnSpectrum objects
    if add_msn_col:
        df['MSnSpectrum'] = su.df_to_MSnSpectra(df, assert_is_valid=False)

    return df.reset_index(drop=True) if reset_index else df
