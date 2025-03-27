from rdkit import Chem
from utils.utils import check_valency, correct_mol, sanitize_smiles
from utils.kekulize import check_kekulize
from rdkit import rdBase

rdBase.DisableLog('rdApp.*')
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

def add_atom(atom_data, atom_set):
    atom_id = atom_data['atom_id']
    atom_type = atom_data['atom_type']
    if atom_type != "H":
        atom_set.add((atom_id, atom_type))
        for _, bond in enumerate(atom_data['bonds']):
            atom_set = add_atom(bond['atom'], atom_set)
    return atom_set

def add_bond(mol, atom_data, atom_id_map):
    if atom_data['atom_type'] == "H":
        return
    curr_index = atom_data['atom_id']
    for i, bond in enumerate(atom_data['bonds']):
        if bond['atom']['atom_type'] == "H":
            continue
        adj_index = bond['atom']['atom_id']
        # bond_type_str = bond['bond_type'].upper()
        bond_type_str = bond['bond_type']
        if bond_type_str == 'DOUBLE':
            bond_type = Chem.rdchem.BondType.DOUBLE
        elif bond_type_str == 'TRIPLE':
            bond_type = Chem.rdchem.BondType.TRIPLE
        elif bond_type_str == 'SINGLE':
            bond_type = Chem.rdchem.BondType.SINGLE
        elif bond_type_str == "AROMATIC":
            bond_type = Chem.rdchem.BondType.AROMATIC
        else:
            raise ValueError(f"Invalid bond type: {bond_type_str}")

        curr_idx = atom_id_map[curr_index]
        adj_idx = atom_id_map[adj_index]
        if curr_idx != adj_idx and mol.GetBondBetweenAtoms(curr_idx, adj_idx) is None:
            mol.AddBond(curr_idx, adj_idx, bond_type)
        add_bond(mol, bond['atom'], atom_id_map)

def tree2smiles(molecule_data, do_correct=False):
    mol = Chem.RWMol()
    atom_set = set()
    atom_set = add_atom(molecule_data, atom_set)
    atom_set = sorted(atom_set, key=lambda x: x[0])
    atom_id_map = {}
    for i, (atom_id, atom_type) in enumerate(atom_set):
        atom_id_map[atom_id] = i
        mol.AddAtom(Chem.Atom(atom_type))
    add_bond(mol, molecule_data, atom_id_map)

    flag, atomid_valence = check_valency(mol)
    if not flag:
        if do_correct:
            mol, _ = correct_mol(mol)
        else:
            try:
                print(atomid_valence)
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                print(f"idx: {idx}, v: {v}, an: {an}")
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    print("Molecule fixed after adjusting formal charge.")
                    return smiles

            except Exception as e:
                print(f"incorrect valence check")
                pass

    # mol = mol.GetMol()
    # try:
    #     Chem.SanitizeMol(mol)
    #     # Chem.Kekulize(mol, clearAromaticFlags=True)
    #     smiles = Chem.MolToSmiles(mol)
    #     smiles = sanitize_smiles(smiles)
    # except Exception as e:
    #     smiles = check_kekulize(mol)

    # if not smiles:
    #     print("Invalid molecule")
    # return smiles
    try:
        mol = mol.GetMol()
        from rdkit.Chem import AllChem
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        # rdmolops.Kekulize(mol)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        smiles = sanitize_smiles(smiles)
    except Exception as e:
        print(f"Wrong mol to smiles")
        # Clear existing aromaticity flags and redefine aromaticity
        smiles = check_kekulize(mol)


    if not smiles:
        print("Invalid molecule")
    return smiles
