# TODO: All click help to docstrings


import click
import dreams.utils.io as io


@click.command()
@click.option('-i', '--input_path', required=True, help='Path to an input file (.mzML or .mzXML).')
@click.option('-o', '--output_path', help='Path to an output file (.hdf5). If not provided, the output is '
                    'stored as --input_path file name with .hdf5 extension.')
@click.option('--store_precursors', is_flag=True, default=True, help='Store the data of precursor spectra (peak '
                    'list and scan id) for each MSn spectrum as a separate hdf5 dataset.')
@click.option('--num_peaks', type=int, help='M/z values and intensities of MSn peak lists will be padded '
                    'with zeros at the right side up to the length of n_peaks. If num_peaks is not specified it '
                    'would be set to a max num. of peaks within spectra that are to be stored.')
@click.option('--num_prec_peaks', type=int, help='M/z values and intensities of MS1 peak lists will be '
                    'padded with zeros at the right side up to the length of n_peaks. If num_peaks is not ' 
                    'specified it would be set to a max num. of peaks within spectra that are to be stored.')
@click.option('--compress_peaks_lvl', type=int, default=0, help='Compression level for peak lists in output '
                    '.hdf5 (integer from 0 to 9).')
@click.option('--compress_full_lvl', type=int, default=0, help='Compression level for all stored attributes '
                    '(e.g. RTs, polarities, etc.) except for peak lists.')
@click.option('--pwiz_stats', is_flag=True, help='Collect ProteoWizard msconvert statistics: '
                    'histogram of types of spectra converted by msconvert and number of spectra centroided by '
                    'msconvert but having zero intensities.')
@click.option('--del_in', is_flag=True, help='Delete the input .mzML or .mzXML file.')
@click.option('--assign_dformats', is_flag=True, help='Assign data formats to MSn spectra.')
@click.option('--log_path', help='Input to the log file containing errors during opening of files and flaws '
                    'of invalid spectra. If set to None, the log file is stored as --input_path file name with '
                    '.hdf5 extension.')
@click.option('--verbose', is_flag=True, help='Log scan number for each invalid spectrum and log '
                    'additional statistics. The statistics are redundant in a sense that they can be calculated '   
                    'from the output .hdf5 file but are helpful for the fast analysis of the input file and '
                    'debugging.')
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
    io.lcmsms_to_hdf5(
        input_path=input_path,
        output_path=output_path,
        num_peaks=num_peaks,
        num_prec_peaks=num_prec_peaks,
        store_precursors=store_precursors,
        compress_peaks_lvl=compress_peaks_lvl,
        compress_full_lvl=compress_full_lvl,
        pwiz_stats=pwiz_stats,
        del_in=del_in,
        assign_dformats=assign_dformats,
        log_path=log_path,
        verbose=verbose
    )


@click.group()
def cli():
    pass


cli.add_command(lcmsms_to_hdf5)


if __name__ == '__main__':
    cli()
