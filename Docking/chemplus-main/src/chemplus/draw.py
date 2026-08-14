from chemplus.vina import rot_bond_helper
from chemplus import mdl
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem import AllChem
import base64
from io import BytesIO

def image_base64(im):
    with BytesIO() as buffer:
        im.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()

def draw_mol(mol, png_file=None):
    mol_copy = Chem.RemoveHs(mol)
    if mol_copy.GetNumConformers() and mol_copy.GetConformer().Is3D():
        Chem.rdmolops.AssignStereochemistryFrom3D(mol_copy)
        AllChem.Compute2DCoords(mol_copy)
    
    draw_options = Chem.Draw.MolDrawOptions()
    draw_options.minFontSize = 18
    draw_options.maxFontSize = 90
    draw_options.bondLineWidth = 2
    data = Draw.MolToImage(mol_copy, (600, 400), returnPNG=True, kekulize=True, options = draw_options)
    if png_file is None:
        data_encoded = image_base64(data)
        return f'<img data-content="rdkit/molecule" src="data:image/png;base64,{data_encoded}" width="400" alt="Mol"/>'
    else:
        data.save(png_file)

def draw_mol_bonds(mol, bonds, bond_colors=None, png_file=None):
    mol_copy = Chem.RemoveHs(mol)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol_copy)
    AllChem.Compute2DCoords(mol_copy)
    
    atoms = []
    if bond_colors:
        atom_colors={}
    else:
        atom_colors=None
        
    for bond_idx in bonds:
        bond = mol_copy.GetBondWithIdx(bond_idx)
        atoms.append(bond.GetBeginAtomIdx())
        atoms.append(bond.GetEndAtomIdx())
        if bond_colors:
            atom_colors[bond.GetBeginAtomIdx()] = bond_colors[bond_idx]
            atom_colors[bond.GetEndAtomIdx()] = bond_colors[bond_idx]
    
    draw_options = Chem.Draw.MolDrawOptions()
    draw_options.minFontSize = 18
    draw_options.maxFontSize = 90
    draw_options.bondLineWidth = 2
    
    d = rdMolDraw2D.MolDraw2DCairo(600, 400)
    d.SetDrawOptions(draw_options)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol_copy, highlightAtoms=atoms, highlightBonds=bonds, highlightAtomColors=atom_colors, highlightBondColors=bond_colors)
    d.FinishDrawing()
    
    if png_file is None:
        data_encoded = base64.b64encode(d.GetDrawingText()).decode()
        return f'<img data-content="rdkit/molecule" src="data:image/png;base64,{data_encoded}" width="400" alt="Mol"/>'
    else:
        d.WriteDrawingText(png_file)
    
def draw_mol_rot_bonds(mol, method=rot_bond_helper.vina_exactly, adjusth=True, png_file=None):
    if adjusth:
        mol = mdl.mol_adjust_hpolar(mol)
        
    hit_bonds = method(mol)
    
    return draw_mol_bonds(mol, hit_bonds, png_file)

def draw_bad_rot_bonds(pdb_file, bad_rb, missed_rb, png_file=None):
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=True)
    atoms_idx_pdbnum_dict = {}
    for atom in mol.GetAtoms():
        atoms_idx_pdbnum_dict[atom.GetPDBResidueInfo().GetSerialNumber()] = atom.GetIdx()
    bad_rb = [(atoms_idx_pdbnum_dict[atom1_num], atoms_idx_pdbnum_dict[atom2_num]) for atom1_num, atom2_num in bad_rb]
    missed_rb = [(atoms_idx_pdbnum_dict[atom1_num], atoms_idx_pdbnum_dict[atom2_num]) for atom1_num, atom2_num in missed_rb]
    bad_bonds = [mol.GetBondBetweenAtoms(atom1_idx, atom2_idx).GetIdx() for atom1_idx, atom2_idx in bad_rb]
    missed_bonds = [mol.GetBondBetweenAtoms(atom1_idx, atom2_idx).GetIdx() for atom1_idx, atom2_idx in missed_rb]
    bond_colors = {}
    for bond in bad_bonds:
        bond_colors[bond] = (1,0,0)
    for bond in missed_bonds:
        bond_colors[bond] = (1,1,0)
    return draw_mol_bonds(mol, bad_bonds + missed_bonds, bond_colors=bond_colors, png_file=png_file)