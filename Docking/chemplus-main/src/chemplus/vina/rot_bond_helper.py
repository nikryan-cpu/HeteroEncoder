from chemplus import mdl
from rdkit import Chem
#rdDepictor.SetPreferCoordGen(True)

def match(mol, sma):
    pattern = Chem.MolFromSmarts(sma)
    hit_bond_atoms = mol.GetSubstructMatches(pattern)
    hit_bonds = [mol.GetBondBetweenAtoms(atom1, atom2).GetIdx() for atom1, atom2 in hit_bond_atoms]
    
    return hit_bonds

def rdkit_simple(mol):
    sma = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    return match(mol, sma)

def rdkit_strict(mol):
    sma = "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
    return match(mol, sma)

def vina_exactly(mol):
    sma = "[!D1&!$([C,N]#[CD1,ND1,CH,NH])&!$([CD3](-!@[#7])(=!@[#7])-!@[#7])&!$([CH3])]-!@[!D1&!$([C,N]#[CD1,ND1,CH,NH])&!$([CD3](-!@[#7])(=!@[#7])-!@[#7])&!$([CH3])]"
    sma_amide = "[$([CD3](=[O]))]-!@[#7]"
    all_matches = match(mol, sma)
    amides = match(mol, sma_amide)
    return [i for i in all_matches if i not in amides]

def get_rot_bonds(mol, method=vina_exactly, adjusth=False):
    if adjusth:
        mol = mdl.mol_adjust_hpolar(mol)
        
    return method(mol)

if __name__ == '__main__':
    smi = 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'
    mol = Chem.MolFromSmiles(smi)

    print(len(get_rot_bonds(mol)))