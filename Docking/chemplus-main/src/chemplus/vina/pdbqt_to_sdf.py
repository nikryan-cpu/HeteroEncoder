import os
import glob
import argparse
from rdkit import Chem
from tqdm import tqdm
from chemplus import sdf

def pdbqt_coords_to_pdb(pdbqt_file, pdb_file, silent=True):
    pdb_lines = open(pdb_file).readlines()
    pdbqt_lines = open(pdbqt_file).readlines()

    pdbqt_atoms = {}
    
    for line in pdbqt_lines:
        if line[:6] != "HETATM":
            continue
            
        atom_name = line[12:16].strip()
        if atom_name not in pdbqt_atoms:
            pdbqt_atoms[atom_name] = line[30:54]
        else:
            if not silent:
                print("PDBQT atom names are not unique")
            return
    
    pdb_atoms = set()
    
    for i in range(len(pdb_lines)):
        pdb_line = pdb_lines[i]
        if pdb_line[:6] != "HETATM":
            continue
        atom_name = pdb_line[12:16].strip()
        
        if atom_name not in pdb_atoms:
            pdb_atoms.add(atom_name)
        else:
            if not silent:
                print("PDB atom names are not unique")
            return
                  
        if atom_name in pdbqt_atoms:
            pdb_lines[i] = pdb_line[:30] + pdbqt_atoms[atom_name] + pdb_line[54:]
        else:
            if not silent:
                print("PDBQT does not contain an atom with name:", atom_name)
            return

    open(pdb_file, "w").writelines(pdb_lines)
    return True

t = Chem.rdmolops.RemoveHsParameters()
t.removeMapped = False

def pdbqt_to_sdf(pdbqt_dir, pdb_dir, sdf_file, sdf_format_dir, pdb_format_dir, overwrite=True, silent=True):
    
    if not os.path.exists(pdbqt_dir):
        raise Exception("PDBQT folder is not exists")
        
    if not os.path.exists(pdb_dir):
        raise Exception("PDB folder is not exists")
        
    if not os.path.exists(os.path.dirname(sdf_file)):
        os.makedirs(os.path.dirname(sdf_file))
    
    pdbqt_num = 0
    converted_pdbqt_num = 0
    bad_lig_list = []    
    writer = sdf.SDFWriter(sdf_file)
        
    for pdbqt_file in tqdm(glob.glob(pdbqt_dir + os.sep + "*.pdbqt")):
        pdbqt_num +=1
        lig_id = os.path.basename(pdbqt_file).split(".")[0]
        
        pdb_file = pdb_dir + os.sep + lig_id + ".pdb"
        pdb_format_file = pdb_format_dir + os.sep + lig_id + ".pdb"
        
        if not overwrite and os.path.exists(mol_out_file):
            continue
        
        try:
            pdb_mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
        except:
            bad_lig_list.append(lig_id)
            if not silent:
                print("Unable to open PDB file for ligand", lig_id)
            continue
            
        for atom in pdb_mol.GetAtoms():
            if atom.GetAtomicNum() == 1 and atom.GetNeighbors()[0].GetAtomicNum() != 6:
                atom.SetAtomMapNum(1)
        
        try:
            pdb_mol = Chem.RemoveHs(pdb_mol, params = t, sanitize=True)
        except Chem.AtomValenceException:
            bad_lig_list.append(lig_id)
            if not silent:
                print("Bad atom valence in lig:", lig_id)
            continue
        Chem.MolToPDBFile(pdb_mol, pdb_format_file)

        if not pdbqt_coords_to_pdb(pdbqt_file, pdb_format_file, silent):
            bad_lig_list.append(lig_id)
            if not silent:
                print("Bad lig:", lig_id)
            continue
        
        mol = Chem.MolFromPDBFile(pdb_format_file, removeHs=False, sanitize=True, proximityBonding=False)
        mol = Chem.AddHs(mol, addCoords=True)
        Chem.rdmolops.AssignStereochemistryFrom3D(mol)
        mol.SetProp("_Name", lig_id)
        writer.write(mol)
        Chem.MolToMolFile(mol, sdf_format_dir + os.sep + lig_id + ".sdf")
        converted_pdbqt_num +=1
        
    writer.close()
    return pdbqt_num, converted_pdbqt_num, bad_lig_list
