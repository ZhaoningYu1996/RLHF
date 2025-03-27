from pydantic import BaseModel
from typing import List
# from utils.config import  BondTypeEnum

def create_mol_format(atom_type_enum, bond_type_enum):
    class Atom(BaseModel):
        atom_id: int
        atom_type: atom_type_enum
        bonds: List['Bond'] = []
    
    class Bond(BaseModel):
        atom: Atom
        bond_type: bond_type_enum
    
    return Atom, Bond

# class Atom_id(BaseModel):
#     atom_id: int
#     atom_name: AtomEnum
#     bonds: List['Bond_id'] = []

# class Bond_id(BaseModel):
#     atom: Atom_id
#     bond_type: BondTypeEnum

# class Molecule(BaseModel):
#     atom_name: AtomEnum
#     adjacency_atoms: List['Molecule'] = []
#     bond_types: List[BondTypeEnum] = []

# class Atom(BaseModel):
#     atom_name: AtomEnum
#     bonds: List['Bond'] = []

# class Bond(BaseModel):
#     atom: Atom
#     bond_type: BondTypeEnum
