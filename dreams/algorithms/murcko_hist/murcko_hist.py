import numpy as np
import rdkit.Chem.Scaffolds.MurckoScaffold as MurckoScaffold
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from collections import Counter


# TODO: docstrings


def multirings(mol):
    """
    Returns a list of sets of atom indices belonging to "multiring"  in a molecule. Each "mutliring" is a generalized
    ring, where ordinary fused rings are considered to be a single ring/set (therefore "mutliring" name).
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

def break_rings(mol: Mol, rings_size=3):
    """
    Breaks all `rings_size`-membered rings in a molecule by removing a bond with minimal degree wrt rings. It is
    supposed to be used prior to computing Murcko scaffolds.
    NOTE: it is not tested and is probably not useful for rings_size != 3.
    """

    def get_ring(mol, size) -> list[int] | None:
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

def murcko_hist(mol, as_dict=True, show_mol_scaffold=False, no_residue_atom_as_linker=True,
                break_three_membered_rings=True):

    if show_mol_scaffold:
        display(mol)

    # Break all three-membered rings
    if break_three_membered_rings:
        mol = break_rings(mol, rings_size=3)

    # Get Murcko scaffold of `mol`
    m = MurckoScaffold.GetScaffoldForMol(mol)
    if show_mol_scaffold:
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


def murcko_hists_dist(h1, h2):
    dist = 0
    for k in set(h1.keys()) | set(h2.keys()):
        if k not in h1.keys():
            dist += h2[k]
        elif k not in h2.keys():
            dist += h1[k]
        else:
            dist += abs(h1[k] - h2[k])
    return dist


def are_sub_hists(h1, h2, k=3, d=4):
    if min(sum(list(h1.values())), sum(list(h2.values()))) <= k:
        return h1 == h2
    return murcko_hists_dist(h1, h2) <= d


# def to_ring_hist(h):
#     c = Counter()
#     for k in h.keys():
#         c[k.split('_')[0]] += h[k]
#     return dict(c)
#
#
# def is_ring_subhist(h1, h2):
#     h1, h2 = to_ring_hist(h1), to_ring_hist(h2)
#     for key in h1.keys():
#         if not key in h2.keys() or h1[key] > h2[key]:
#             return False
#     return True
#
#
# def are_eq_murcko_hists(h1, h2, k=2):
#     if min(sum(list(h1.values())), sum(list(h2.values()))) <= k:
#         return h1 == h2
#     return is_ring_subhist(h1, h2) or is_ring_subhist(h2, h1)
