from rdkit import Chem

def mol_adjust_hpolar(mol):
    mol = Chem.RemoveHs(mol)
    polar_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != 'C']
    if len(polar_atoms):
        mol = Chem.AddHs(mol, onlyOnAtoms=polar_atoms)
        
    return mol

def is_iso_mol(iso_mol, mol):
    #removal of stereochemical hydrogens
    cleaned_iso_mol = Chem.Mol(iso_mol)
    cleaned_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(cleaned_iso_mol)
    Chem.RemoveStereochemistry(cleaned_mol)
    cleaned_iso_mol = Chem.RemoveHs(cleaned_iso_mol)
    cleaned_mol = Chem.RemoveHs(cleaned_mol)
    if Chem.MolToSmiles(cleaned_iso_mol, isomericSmiles=False) != Chem.MolToSmiles(cleaned_mol, isomericSmiles=False):
        return False
    
    iso_mol = Chem.RemoveHs(iso_mol)
    mol = Chem.RemoveHs(mol)
    if not iso_mol.GetSubstructMatch(mol, useChirality=True):
        return False
    
    return True

def is_iso_smiles(iso_smi, smi):
    mol = Chem.MolFromSmiles(smi)
    iso_mol = Chem.MolFromSmiles(iso_smi)
    
    if is_iso_mol(iso_mol, mol):
        return True
    else:
        return False

def get_actual_smiles(mol, kekuleSmiles=True):
    mol_copy = Chem.Mol(mol)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol_copy)
    mol_copy = Chem.RemoveHs(mol_copy)
    
    return Chem.MolToSmiles(mol_copy, canonical=True, isomericSmiles=True, kekuleSmiles=kekuleSmiles)

def is_mol_3d_correct(mol):
    
    if not mol.GetNumConformers():
        return "No conformers"
    if not mol.GetConformer().Is3D():
        return "Not 3D conformer"
    
    mol_copy = Chem.Mol(mol)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol_copy)
    if is_iso_mol(mol_copy, mol):
        return True
    else:
        return False
    
