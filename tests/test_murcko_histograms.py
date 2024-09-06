import pytest

def test_break_rings():
    from dreams.algorithms.murcko_hist.murcko_hist import break_rings
    from rdkit import Chem
    mol = Chem.MolFromInchi('InChI=1S/C28H22O6/c29-21-3-1-9-25-23(31)19-7-5-15(11-17(19)13-27(21,25)33-25)16-6-8-20-18(12-16)14-28-22(30)4-2-10-26(28,34-28)24(20)32/h5-8,11-12H,1-4,9-10,13-14H2')
    mol_broken = break_rings(mol, rings_size=3)
    expect_inchi = 'InChI=1S/C28H26O6/c29-23-5-1-3-21-25(31)19-9-7-15(11-17(19)13-27(21,23)33)16-8-10-20-18(12-16)14-28(34)22(26(20)32)4-2-6-24(28)30/h7-12,21-22,33-34H,1-6,13-14H2'
    assert Chem.MolToInchi(mol_broken) == expect_inchi