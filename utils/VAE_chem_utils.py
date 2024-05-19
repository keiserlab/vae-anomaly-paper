import selfies as sf
from rdkit import Chem
from typing import List, Dict, Tuple, Set, Union, NewType

SMILES = NewType('SMILES', str)
SELFIES = NewType('SELFIES', str)

def rd_canonicalize(smiles: SMILES) -> SMILES:
    
    mol = Chem.MolFromSmiles(smiles)
    smi = Chem.MolToSmiles(mol)
    
    return smi

def rd_canonicalize_smiles_batch(smiles_list: List[SMILES]) -> List[SMILES]:
    
    return [rd_canonicalize(s) for s in smiles_list]

def is_smiles_batch_canonical(smiles_list: List[SMILES]) -> bool:
    
    return all([s == rd_canonicalize(s) for s in smiles_list])
