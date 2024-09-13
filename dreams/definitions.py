import sys
import inspect
import pathlib

# Root path
DREAMS_DIR = pathlib.Path(__file__).parent.absolute()

# Paths to datasets and their dirs
DATA_DIR = DREAMS_DIR.parent / 'data'
NIST20 = DATA_DIR / 'NIST20'
NIST20_PKL = NIST20 / 'nist20_clean.pkl'
MONA = DATA_DIR / 'MoNA'
MONA_PKL = MONA / 'mona_clean.pkl'
CASMI = DATA_DIR / 'CASMI'
CASMI_PKL = CASMI / 'casmi16.pkl'
MASSIVE_DIR = DATA_DIR / 'MassIVE'
MASSIVE_DATA = MASSIVE_DIR / 'data'
MASSIVE_SCRIPTS = MASSIVE_DIR / 'scripts'
MIST = DATA_DIR / 'MIST' / 'datasets'
MERGED_DATASETS = DATA_DIR / 'merged'
GEMS_DIR = DATA_DIR / 'GeMS'

# Molecules
PUBCHEM = DATA_DIR / 'molecules' / 'pubchem'
PUBCHEM_FPS = PUBCHEM / 'cid_smiles_index.h5'

# Experiments
EXPERIMENTS_DIR = DREAMS_DIR.parent / 'experiments'

# Scripts
SCRIPTS_DIR = DREAMS_DIR.parent / 'scripts'

# Misc
MISC = DREAMS_DIR.parent / 'misc'

# Figures
FIGURES = MISC / 'figures'

# Pre-trained models
MODELS = DREAMS_DIR / 'models'
PRETRAINED = MODELS / 'pretrained'

# Data column names
SPECTRUM = 'spectrum'
PRECURSOR_MZ = 'precursor_mz'
CHARGE = 'charge'
ADDUCT = 'adduct'
SMILES = 'smiles'
MSDATA_COLUMNS = [SPECTRUM, PRECURSOR_MZ, CHARGE, ADDUCT]
MSDATA_LABELED_COLUMNS = MSDATA_COLUMNS + [SMILES]
FOLD = 'fold'

DATASET = 'dataset'
ID = 'id'
DREAMS_EMBEDDING = 'DreaMS_embedding'
RT = 'RT'
SCAN_NUMBER = 'scan_number'
FILE_NAME = 'file_name'
NAME = 'name'

def export():
    # Export paths to environment variables
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if isinstance(obj, pathlib.PosixPath):
            path = str(obj)
            print(f'export {name}={path}')
