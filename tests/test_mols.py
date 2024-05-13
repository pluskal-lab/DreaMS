import rdkit.Chem.AllChem as Chem
import dreams.utils.mols as mu


def test_morgan_fps():
    smiles = [
        'C1C(N=C(S1)C2=NC3=C(S2)C=C(C=C3)O)C(=O)O',
        '[H][C@@]12[C@@H](CC[C@]3(C)C1C(C)=CC[C@@]23[H])C(C)C'
        'C[C@@H]1C[C@]2([C@H]3[C@H]4[C@]1([C@@H]5C=C(C(=O)[C@]5(CC(=C4)COC(=O)CC6=CC(=C(C=C6)O)OC)O)C)OC(O3)(O2)CC7=CC=CC=C7)C(=C)C'
    ]
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp_bin = mu.morgan_fp(m, binary=True, as_numpy=True)
        fp_int = mu.morgan_fp(m, binary=False, as_numpy=True)
        assert (fp_bin == fp_int.astype(bool)).all()
        assert fp_bin.min() == 0
        assert fp_bin.max() == 1
        assert fp_int.min() == 0
        assert fp_int.max() > 1
