from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')


def check_kekulize(mol):
    try:
        Chem.SanitizeMol(mol)
        print("Molecule is initially valid with no kekulization issues.")
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print("Initial kekulization failed:", e)

    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        Chem.SetAromaticity(mol)
        Chem.SanitizeMol(mol)
        print("Molecule fixed after adjusting aromaticity.")
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception as e:
        print("Kekulization failed after aromaticity adjustment:", e)
    
    # If aromaticity adjustment fails, attempt explicit bond adjustments
    emol = Chem.EditableMol(mol)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            # Try changing aromatic bonds to single or double
            emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            emol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order=Chem.BondType.SINGLE)

    modified_mol = emol.GetMol()
    try:
        Chem.SanitizeMol(modified_mol)
        print("Molecule fixed after modifying bonds.")
        smiles = Chem.MolToSmiles(modified_mol)
        return smiles
    except Exception as e:
        print("Attempt to fix by modifying bonds failed:", e)

    try:
        for atom in modified_mol.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in modified_mol.GetBonds():
            bond.SetIsAromatic(False)
        Chem.SanitizeMol(modified_mol)
        smiles = Chem.MolToSmiles(modified_mol)
        print("Molecule fixed after change aromatic.")
        return smiles
    except Exception as e:
        print("Final attempt to fix by modifying aromatic failed:", e)

    return None