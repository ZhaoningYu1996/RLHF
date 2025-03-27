import re
from rdkit import Chem

bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}

def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible valency

    Return:
        True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence
    
def correct_mol(mol):
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            # Error message is one of the form: 
            # 'Explicit valence for atom # 0 O, 3, is greater than permitted
            # 'Explicit valence for atom # 15 Rn greater than permitted'
            # assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            queue = []

            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder_m[t])

    return mol, no_correct

# Sanitize and kekulize the SMILES
# def sanitize_smiles(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         Chem.Kekulize(mol, clearAromaticFlags=True)
#         return Chem.MolToSmiles(mol)
#     except:
#         return None

def sanitize_smiles(smiles, kekulize=True):
    mol = Chem.MolFromSmiles(smiles)
    mol = sanitize_mol(mol)
    if mol is None:
        return None
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)

    return Chem.MolToSmiles(mol)

def sanitize_mol(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
    except:
        return None
    return mol