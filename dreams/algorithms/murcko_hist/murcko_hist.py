import numpy as np
import rdkit.Chem.Scaffolds.MurckoScaffold as MurckoScaffold
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from typing import List, Set, Dict, Union, Tuple


def multirings(mol: Mol) -> List[Set[int]]:
    """
    Returns a list of sets of atom indices belonging to "multiring" in a molecule.

    Each "multiring" is a generalized ring, where ordinary fused rings are considered
    to be a single ring/set (hence the "multiring" name).

    Args:
        mol (Mol): The input RDKit molecule.

    Returns:
        List[Set[int]]: A list of sets, where each set contains atom indices belonging to a multiring.
    """
    br = [set(r) for r in mol.GetRingInfo().BondRings()]
    rings_bonds = []
    for i, r_i in enumerate(br):
        ring = r_i
        adj_bonds = [r_j for j, r_j in enumerate(br) if i != j and len(r_i.intersection(r_j)) > 1]
        if adj_bonds:
            ring = r_i.union(*adj_bonds)
        if ring not in rings_bonds:
            rings_bonds.append(ring)

    rings_atoms = []
    for rings_bond in rings_bonds:
        ring_atoms = set()
        for i in rings_bond:
            ring_atoms.add(mol.GetBondWithIdx(i).GetBeginAtomIdx())
            ring_atoms.add(mol.GetBondWithIdx(i).GetEndAtomIdx())
        rings_atoms.append(ring_atoms)

    return rings_atoms

def break_rings(mol: Mol, rings_size: int = 3) -> Mol:
    """
    Breaks all rings of a specified size in a molecule by removing a bond with minimal degree with respect to rings.

    This function is intended to be used prior to computing Murcko scaffolds.
    NOTE: It is not extensively tested and may not be useful for rings_size != 3.

    Args:
        mol (Mol): The input RDKit molecule.
        rings_size (int, optional): The size of rings to break. Defaults to 3.

    Returns:
        Mol: The modified RDKit molecule with specified rings broken.
    """

    def get_ring(mol: Mol, size: int) -> Union[List[int], None]:
        for ring in mol.GetRingInfo().BondRings():
            if len(ring) == size:
                return list(set(ring))
    while (ring := get_ring(mol, size=rings_size)) is not None:
        bonds = mol.GetRingInfo().BondRings()
        degrees = [sum([atom in bond for bond in bonds]) 
                    for atom in ring]
        least_bond = np.argmin(degrees)
        atom = ring[least_bond]
        remove_bond = mol.GetBonds()[atom]
        emol = Chem.EditableMol(mol)
        emol.RemoveBond(remove_bond.GetBeginAtomIdx(), remove_bond.GetEndAtomIdx())
        mol = emol.GetMol()
        mol.ClearComputedProps()
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
    return mol

def murcko_hist(mol: Mol, as_dict: bool = True, show_mol_scaffold: bool = False, 
                no_residue_atom_as_linker: bool = True, break_three_membered_rings: bool = True) -> Union[Dict[str, int], Tuple[np.ndarray, np.ndarray]]:
    """
    Computes the Murcko scaffold histogram for a given molecule.

    This function calculates a histogram of rings in the Murcko scaffold of the input molecule,
    with respect to the number of adjacent rings and linkers.

    Args:
        mol (Mol): The input RDKit molecule.
        as_dict (bool, optional): If True, return the histogram as a dictionary. Otherwise, return as numpy arrays. Defaults to True.
        show_mol_scaffold (bool, optional): If True, display the original molecule and its Murcko scaffold. Defaults to False.
        no_residue_atom_as_linker (bool, optional): If True, do not consider residue atoms as linkers. Defaults to True.
        break_three_membered_rings (bool, optional): If True, break all three-membered rings before processing. Defaults to True.

    Returns:
        Union[Dict[str, int], Tuple[np.ndarray, np.ndarray]]: 
            If as_dict is True, returns a dictionary where keys are string representations of (adjacent rings, adjacent linkers) 
            and values are counts.
            If as_dict is False, returns a tuple of two numpy arrays: unique (adjacent rings, adjacent linkers) pairs and their counts.
    """

    if show_mol_scaffold:
        print('Original molecule:')
        display(mol)

    # Break all three-membered rings
    if break_three_membered_rings:
        mol = break_rings(mol, rings_size=3)

    # Get Murcko scaffold of `mol`
    m = MurckoScaffold.GetScaffoldForMol(mol)
    if show_mol_scaffold:
        print('Murcko scaffold:')
        display(m)

    # Define set of atoms indices belonging to linkers
    link_atoms = set()
    for bond in m.GetBonds():
        if no_residue_atom_as_linker and (bond.GetBeginAtom().GetDegree() < 2 or bond.GetEndAtom().GetDegree() < 2):
            continue
        if not bond.IsInRing():
            link_atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    # Get set of rings (see `multirings` function)
    rings_atoms = multirings(m)

    # For each ring compute
    # (i) Number of adjacent rings
    nums_adj_rings = np.zeros(len(rings_atoms), dtype=int)
    # (ii) Number of adjacent linkers
    nums_adj_links = np.zeros(len(rings_atoms), dtype=int)
    for i, r_i in enumerate(rings_atoms):
        for j, r_j in enumerate(rings_atoms):
            if i == j:
                continue
            nums_adj_rings[i] += len(r_i.intersection(r_j)) // 2
        nums_adj_links[i] = len(r_i.intersection(link_atoms))

    # Compute histogram of rings with respect to the pairs of type ((i), (ii))
    nums_adj = np.stack([nums_adj_rings, nums_adj_links]).T
    hist_adj = np.unique(nums_adj, axis=0, return_counts=True)

    # Return histogram as dictionary
    if as_dict:
        keys = ['_'.join([str(e) for e in hist_adj[0][i]]) for i in range(hist_adj[0].shape[0])]
        values = hist_adj[1].tolist()
        return dict(zip(keys, values))

    # Return histogram as a tuple of two numpy arrays
    return hist_adj


def murcko_hists_dist(h1: Dict[str, int], h2: Dict[str, int]) -> int:
    """
    Computes the distance between two Murcko histogram dictionaries.

    The distance is calculated as the sum of absolute differences between 
    corresponding histogram values, including keys present in only one histogram.

    Args:
        h1 (Dict[str, int]): The first Murcko histogram dictionary.
        h2 (Dict[str, int]): The second Murcko histogram dictionary.

    Returns:
        int: The distance between the two histograms.
    """
    dist = 0
    for k in set(h1.keys()) | set(h2.keys()):
        if k not in h1.keys():
            dist += h2[k]
        elif k not in h2.keys():
            dist += h1[k]
        else:
            dist += abs(h1[k] - h2[k])
    return dist


def are_sub_hists(h1: Dict[str, int], h2: Dict[str, int], k: int = 3, d: int = 4) -> bool:
    """
    Determines if two Murcko histograms are considered sub-histograms of each other.

    This function checks if the histograms are equal when their sums are small,
    or if their distance is within a specified threshold for larger histograms.

    Args:
        h1 (Dict[str, int]): The first Murcko histogram dictionary.
        h2 (Dict[str, int]): The second Murcko histogram dictionary.
        k (int, optional): The threshold for considering small histograms. Defaults to 3.
        d (int, optional): The maximum allowed distance for larger histograms. Defaults to 4.

    Returns:
        bool: True if the histograms are considered sub-histograms, False otherwise.
    """
    if min(sum(list(h1.values())), sum(list(h2.values()))) <= k:
        return h1 == h2
    return murcko_hists_dist(h1, h2) <= d
