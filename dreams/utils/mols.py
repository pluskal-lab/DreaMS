import numpy as np
import itertools
import urllib
import json
import time
import ase
import rdkit
import base64
from io import BytesIO
from tqdm import tqdm
from rdkit import DataStructs, RDLogger
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdchem, Draw, rdMolDescriptors, QED, Crippen, Lipinski
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem.Descriptors import ExactMolWt
from collections import defaultdict
from typing import List, Optional
from pathlib import Path
import dreams.utils.misc as utils


def show_mols(mols, legends='new_indices', smiles_in=False, svg=False, sort_by_legend=False, max_mols=500,
              legend_float_decimals=4, mols_per_row=6, save_pth: Optional[Path] = None):
    """
    Returns svg image representing a grid of skeletal structures of the given molecules

    :param mols: list of rdkit molecules
    :param smiles_in: True - SMILES inputs, False - RDKit mols
    :param legends: list of labels for each molecule, length must be equal to the length of mols
    :param svg: True - return svg image, False - return png image
    :param sort_by_legend: True - sort molecules by legend values
    :param max_mols: maximum number of molecules to show
    :param legend_float_decimals: number of decimal places to show for float legends
    :param mols_per_row: number of molecules per row to show
    :param save_pth: path to save the .svg image to
    """
    disable_rdkit_log()

    if smiles_in:
        mols = [Chem.MolFromSmiles(e) for e in mols]

    if legends == 'new_indices':
        legends = list(range(len(mols)))
    elif legends == 'masses':
        legends = [ExactMolWt(m) for m in mols]
    elif callable(legends):
        legends = [legends(e) for e in mols]

    if sort_by_legend:
        idx = np.argsort(legends).tolist()
        legends = [legends[i] for i in idx]
        mols = [mols[i] for i in idx]

    legends = [f'{l:.{legend_float_decimals}f}' if isinstance(l, float) else str(l) for l in legends]

    img = Draw.MolsToGridImage(mols, maxMols=max_mols, legends=legends, molsPerRow=min(max_mols, mols_per_row),
                         useSVG=svg, returnPNG=False)

    if save_pth:
        with open(save_pth, 'w') as f:
            f.write(img.data)

    return img


def mol_to_formula(mol, as_dict=False):
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return formula_to_dict(formula) if as_dict else formula


def smiles_to_formula(s, as_dict=False, invalid_mol_smiles=''):
    mol = Chem.MolFromSmiles(s)
    if not mol and invalid_mol_smiles is not None:
        f = invalid_mol_smiles
    else:
        f = rdMolDescriptors.CalcMolFormula(mol)
    if as_dict:
        f = formula_to_dict(f)
    return f


class MolPropertyCalculator:
    def __init__(self):
        # Estimates of min and max values from the training part of MoNA and NIST20 Murcko histograms split
        self.min_maxs = {
            'AtomicLogP': {'min': -13.054800000000025, 'max': 26.849200000000053},
            'NumHAcceptors': {'min': 0.0, 'max': 36.0},
            'NumHDonors': {'min': 0.0, 'max': 20.0},
            'PolarSurfaceArea': {'min': 0.0, 'max': 585.0300000000002},
            'NumRotatableBonds': {'min': 0.0, 'max': 68.0},
            'NumAromaticRings': {'min': 0.0, 'max': 8.0},
            'NumAliphaticRings': {'min': 0.0, 'max': 22.0},
            'FractionCSP3': {'min': 0.0, 'max': 1.0},
            'QED': {'min': 0.0, 'max': 1.0},  # 'QED': {'min': 0.008950206972239864, 'max': 0.9479380820623227},
            'SyntheticAccessibility': {'min': 1.0, 'max': 10.0},  # 'SyntheticAccessibility': {'min': 1.0549172379947862, 'max': 8.043981630210263},
            'BertzComplexity': {'min': 2.7548875021634682, 'max': 3748.669248605835}
        }
        self.prop_names = list(self.min_maxs.keys())

    def mol_to_props(self, mol, min_max_norm=False):
        props = {
            'AtomicLogP': Crippen.MolLogP(mol),
            'NumHAcceptors': Lipinski.NumHAcceptors(mol),
            'NumHDonors': Lipinski.NumHDonors(mol),
            'PolarSurfaceArea': rdMolDescriptors.CalcTPSA(mol),
            'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
            'NumAromaticRings': Lipinski.NumAromaticRings(mol),
            'NumAliphaticRings': Lipinski.NumAliphaticRings(mol),
            'FractionCSP3': Lipinski.FractionCSP3(mol),
            'QED': QED.qed(mol),
            'SyntheticAccessibility': sascorer.calculateScore(mol),
            'BertzComplexity': rdkit.Chem.GraphDescriptors.BertzCT(mol)
        }
        if min_max_norm:
            props = self.normalize_props(props)
        return props

    def normalize_prop(self, prop, prop_name):
        return (prop - self.min_maxs[prop_name]['min']) / (self.min_maxs[prop_name]['max'] - self.min_maxs[prop_name]['min'])

    def denormalize_prop(self, prop, prop_name, do_not_add_min=False):
        res = prop * (self.min_maxs[prop_name]['max'] - self.min_maxs[prop_name]['min'])
        if not do_not_add_min:
            res = res + self.min_maxs[prop_name]['min']
        return res

    def normalize_props(self, props):
        return {k: self.normalize_prop(v, k) for k, v in props.items()}

    def denormalize_props(self, props):
        return {k: self.denormalize_prop(v, k) for k, v in props.items()}

    def __len__(self):
        return len(self.prop_names)


def formula_to_dict(formula):
    """
    Transforms chemical formula string to dictionary mapping elements to their frequencies
    e.g. 'C15H24' -> {'C': 15, 'H': 24}
    """
    elem_count = defaultdict(int)
    #try:
    formula = formula.replace('+', '').replace('-', '').replace('[', '').replace(']', '')
    formula_counts = ase.formula.Formula(formula)
    formula_counts = formula_counts.count().items()
    for k, v in formula_counts:
        elem_count[k] += v
    #except Exception as e:
    #    print(f'Invalid formula: {formula} ({e.__class__.__name__})')

    return elem_count


def rdkit_fp(mol, fp_size=4096):
    """Default RDKit fingerprint."""
    return Chem.RDKFingerprint(mol, fpSize=fp_size)


def tanimoto_sim(fp1, fp2):
    """Default RDKit Tanimoto distance."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def rdkit_mol_sim(m1, m2, fp_size=4096):
    """Default RDKit Tanimoto distance on default RDKit fingerprint."""
    return tanimoto_sim(rdkit_fp(m1, fp_size=fp_size), rdkit_fp(m2, fp_size=fp_size))


def rdkit_smiles_sim(s1, s2, fp_size=4096):
    """Default RDKit Tanimoto distance on default RDKit fingerprint."""
    return rdkit_mol_sim(Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2), fp_size=fp_size)


def morgan_fp(mol, binary=True, fp_size=4096, radius=2, as_numpy=True):
    if binary:
        fp = Chem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    else:
        fp = Chem.GetHashedMorganFingerprint(mol, radius=radius, nBits=fp_size)

    if as_numpy:
        return rdkit_fp_to_np(fp)
    return fp


def maccs_fp(mol, as_numpy=True):
    """
    NOTE: Since indexing of MACCS keys starts from 1, when converting to numpy array with `as_numpy`, the first element
          is removed, so the resulting array has 166 elements instead of 167.
    """
    fp = GenMACCSKeys(mol)
    if as_numpy:
        return rdkit_fp_to_np(fp)[1:]
    return fp


def fp_func_from_str(s):
    """
    :param s: E.g. "fp_rdkit_2048", "fp_rdkit_2048" or "fp_maccs_166".
    """
    _, fp_type, n_bits = s.split('_')
    n_bits = int(n_bits)
    if fp_type == 'rdkit':
        return lambda mol: np.array(rdkit_fp(mol, fp_size=n_bits), dtype=float)
    elif fp_type == 'morgan':
        return lambda mol: morgan_fp(mol, fp_size=n_bits).astype(float, copy=False)
    elif fp_type == 'maccs':
        return lambda mol: maccs_fp(mol).astype(float, copy=False)
    else:
        raise ValueError(f'Invalid fingerprint function name: "{s}".')


def morgan_mol_sim(m1, m2, fp_size=4096, radius=2):
    return tanimoto_sim(
        morgan_fp(m1, fp_size=fp_size, radius=radius, as_numpy=False),
        morgan_fp(m2, fp_size=fp_size, radius=radius, as_numpy=False)
    )


def morgan_smiles_sim(s1, s2, fp_size=4096, radius=2):
    return morgan_mol_sim(Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2), fp_size=fp_size, radius=radius)


def rdkit_fp_to_np(fp):
    fp_np = np.zeros((0,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, fp_np)
    return fp_np


def np_to_rdkit_fp(fp):
    fp = fp.round().astype(int, copy=False)
    bitstring = ''.join(fp.astype(str))
    return DataStructs.cDataStructs.CreateFromBitString(bitstring)


def mol_to_inchi14(mol: Chem.Mol):
    return Chem.MolToInchiKey(mol).split('-')[0]


def smiles_to_inchi14(s):
    return mol_to_inchi14(Chem.MolFromSmiles(s))


def generate_fragments(mol: Chem.Mol, max_cuts: int = None):
    """
    Generates all possible fragments of a molecule up to a certain number of bond cuts or without the restriction if
    `max_cuts` is not specified.

    :param mol: an RDKit molecule object
    :param max_cuts: the maximum number of bonds to cut
    :return a set of RDKit Mol objects representing all possible fragments
    """

    bonds = mol.GetBonds()
    # bonds = [bond for bond in bonds if bond.GetBondType() in [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE]]
    fragments = set()
    for i in range(1, len(bonds) + 1):

        if max_cuts and i > max_cuts:
            break

        for combination in itertools.combinations(bonds, i):
            new_mol = rdchem.RWMol(mol)
            for bond in combination:
                new_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

            # Update properties such as ring membership after changing the molecule's structure.
            for fragment in Chem.GetMolFrags(new_mol, asMols=True, sanitizeFrags=False):
                fragments.add(Chem.MolToSmiles(fragment))

    fragments = [Chem.MolFromSmiles(f) for f in fragments]
    return [f for f in fragments if f is not None]


def generate_spectrum(mol: Chem.Mol, prec_mz: float = None, fragments: List = None, max_cuts: int = None):
    """
    Generates an MS/MS spectrum by exhaustively simulating the m/z values of theoretical fragments of a given molecule.
    The algorithm is very simplistic since it considers only subgraph-like fragments, does not consider isotopes, etc.

    :param mol: An RDKit molecule object.
    :param prec_mz: The m/z value of a molecule. If not specified, it is calculated as the sum of the
                    exact molecular weight of the molecule and 1.
    :param fragments: A list of RDKit Mol objects representing pre-generated fragments of the molecule. If not specified,
                     the function will generate the fragments automatically.
    :param max_cuts: The maximum number of bonds to cut when generating fragments. If not specified, all possible
                     fragments will be generated without any restriction on the number of cuts.
    :return: A spectrum represented as a numpy array with two columns: m/z values and their respective intensities.
    """

    # Simulate the m/z of "protonated adduct"
    if not prec_mz:
        prec_mz = ExactMolWt(mol) + 1

    # Fragment molecule
    if not fragments:
        fragments = generate_fragments(mol, max_cuts=max_cuts)

    # Simulate spectrum
    masses = np.round(np.array([prec_mz - ExactMolWt(f) for f in fragments]))
    ins, mzs = np.histogram(masses, bins=np.arange(0, np.ceil(max(masses)), 1))
    spec = np.stack([mzs[1:], ins]).T

    return spec


def closest_mz_frags(query_mz, frags, n=1, mass_shift=1, return_masses=False, print_masses=True):
    masses = [ExactMolWt(f) + mass_shift for f in frags]
    idx = utils.get_closest_values(masses, query_mz, n=n, return_idx=True)
    frags, masses = [frags[i] for i in idx], [masses[i] for i in idx]
    if n == 1:
        frags, masses = frags[0], masses[0]
    if print_masses:
        print(masses)
    if return_masses:
        return frags, masses
    return frags


def disable_rdkit_log():
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)


def np_classify(smiles: List[str], progress_bar=True, sleep_each_n_requests=100):
    np_classes = []
    for i, s in enumerate(tqdm(smiles) if progress_bar else smiles):
        if i % sleep_each_n_requests == 0 and i > 0:
            time.sleep(1)
        print(s)
        with urllib.request.urlopen(f'https://npclassifier.ucsd.edu/classify?smiles={urllib.parse.quote(s)}') as url:
            res = json.load(url)
            for k in list(res.keys()):
                if 'fp' in k:
                    res.pop(k)
            np_classes.append(res)
    return np_classes


def mol_to_img_str(mol, svg_size=200):
    """
    Supposed to be used with `pyvis` for showing molecule images as graph nodes.
    """
    buffered = BytesIO()
    d2d = rdMolDraw2D.MolDraw2DSVG(svg_size, svg_size)
    opts = d2d.drawOptions()
    opts.clearBackground = False
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    img_str = d2d.GetDrawingText()
    buffered.write(str.encode(img_str))
    img_str = base64.b64encode(buffered.getvalue())
    img_str = f"data:image/svg+xml;base64,{repr(img_str)[2:-1]}"
    return img_str


def formula_is_carbohydrate(formula):
    return set(formula.keys()) <= {'C', 'H', 'O'}


def formula_is_halogenated(formula):
    return sum([(formula[e] if e in formula else 0) for e in ['F', 'Cl', 'Br', 'I']]) > 0


def formula_type(f):
    if isinstance(f, str):
        f = formula_to_dict(f)

    if not f:
        return 'No formula'
    elif formula_is_carbohydrate(f):
        return 'Carbohydrate'
    elif set(f.keys()) <= {'C', 'H', 'O', 'N'}:
        return 'Carbohydrate with nitrogen'
    elif set(f.keys()) <= {'C', 'H', 'O', 'N', 'S'} and 'N' in f and 'S' in f:
        return 'Carbohydrate with nitrogen and sulfur'
    elif formula_is_halogenated(f):
        return 'Compound with halogens'
    else:
        return 'Other'


def get_mol_mass(mol):
    return ExactMolWt(mol)
