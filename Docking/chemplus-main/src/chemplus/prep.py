from chemplus import mdl
from rdkit import Chem

def is_salt(mol):
    cations_atoms = ['K', 'Mg', 'Na', 'Li', 'Be', 'Ca']
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in cations_atoms:
            return True
    return False

def delete_cations(mol):
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=True)
    cations = [at for at in rwmol.GetAtoms() if at.GetSymbol() in ['K', 'Mg', 'Na', 'Li', 'Be', 'Ca']]
    for cation in cations:
        for nbr in cation.GetNeighbors():
            nbr.SetFormalCharge(-1)
            rwmol.RemoveBond(nbr.GetIdx(), cation.GetIdx())
        rwmol.RemoveAtom(cation.GetIdx())
    return rwmol.GetMol()

def get_uniqiue_frags(mol):
    if is_salt(mol):
        mol = delete_cations(mol)
    mol_frags = Chem.GetMolFrags(mol, asMols=True)
    unique_frags = []
    smiles_set = set()
    for mol_frag in mol_frags:
        smiles = mdl.get_actual_smiles(mol_frag, kekuleSmiles=False)
        if smiles not in smiles_set:
            smiles_set.add(smiles)
            unique_frags.append(mol_frag)
    unique_frags = [(frag, frag.GetNumAtoms()) for frag in unique_frags]
    unique_frags = sorted(unique_frags, key=lambda x: x[1], reverse=True)
    unique_frags = [frag[0] for frag in unique_frags]
    return unique_frags

def is_organic(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            return True
    return False

def is_druglike(mol):
    permitted_atoms = ['H', 'C', 'O', 'N', 'S', 'F', 'I', 'P', 'Cl', 'Br', 'Se']
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in permitted_atoms:
            return False
    return True

def prep_mol(mol):
    mol = Chem.Mol(mol)
    smiles = mdl.get_actual_smiles(mol, kekuleSmiles=False)
    if "*" in smiles:
        return [(mol, "Undetermined structure")]
    mol_frags = get_uniqiue_frags(mol)
    success_frags = []
    failed_frags = []
    for frag in mol_frags:
        inchi = Chem.inchi.MolToInchi(frag, treatWarningAsError=False)
        if "i" in inchi:
            failed_frags.append((frag, "Non-typical isotope"))
            continue
        if not is_organic(frag):
            failed_frags.append((frag,"Non-ogranic molecule"))
            continue
        if not is_druglike(frag):
            failed_frags.append((frag,"Non-druglike molecule"))
            continue
        success_frags.append((frag, "OK"))
    return success_frags + failed_frags
