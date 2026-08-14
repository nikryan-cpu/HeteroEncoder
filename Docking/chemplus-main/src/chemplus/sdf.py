from rdkit import Chem
import gzip
import os
import glob
from tqdm import tqdm
from io import StringIO

#Reading SDF

class SDFReader:
    def __init__(self, filename, sanitize=True, removeHs=True, strictParsing=True):
        self.filename = filename
        self.sanitize = sanitize
        self.removeHs = removeHs
        self.strictParsing = strictParsing
    
    def __iter__(self):
        if ".gz" in self.filename:
            self.inf = gzip.open(self.filename)
        else:
            self.inf = open(self.filename, "rb")
        self.suppl = Chem.ForwardSDMolSupplier(self.inf, self.sanitize, self.removeHs, self.strictParsing)
        return self
    
    def __next__(self):
        try:
            return next(self.suppl)
        except StopIteration:
            self.inf.close()
            raise

def sdf_to_pdb(sdf_file, pdb_dir, name_field="_Name", overwrite=True, silent=False, bond_order=True,
               sanitize=True, removeHs=False, strictParsing=True):
        
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    
    bad_mols_list = []
    converted_mol_num = 0
    
    for count, mol in tqdm(enumerate(SDFReader(sdf_file, sanitize, removeHs, strictParsing))):
        if not mol:
            if not silent:
                print("Bad mol:", count)
            bad_mols_list.append(count)
            continue
        
        mol_name = mol.GetProp(name_field)
        pdb_file_path = pdb_dir + os.sep + mol_name + ".pdb"
        
        if not overwrite and os.path.exists(pdb_file_path):
            continue
        
        if bond_order:
            Chem.MolToPDBFile(mol, pdb_file_path, flavor=20)
        else:
            Chem.MolToPDBFile(mol, pdb_file_path, flavor=28)
        
        converted_mol_num +=1
        
    return count+1, converted_mol_num, bad_mols_list

#Writing SDF

class SDFWriter:
    def __init__(self, filename):
        self.filename = filename
        if isinstance(self.filename, StringIO):
            self.outf = self.filename
        elif ".gz" in self.filename:
            self.outf = gzip.open(self.filename, "wt+")
        else:
            self.outf = open(self.filename, "wt+")
        self.writer = Chem.SDWriter(self.outf)
    
    def write(self, mol):
        self.writer.write(mol)
    
    def set_props(self, properties):
        self.writer.SetProps(properties)
    
    def close(self):
        self.writer.close()
        if not isinstance(self.outf, StringIO):
            self.outf.close()
    

def mols_to_sdf(mols, filename):
    writer = SDFWriter(filename)
    for mol in mols:
        writer.write(mol)
    return writer.close()

def mols_dir_to_sdf(mols_dir, sdf_file):
    if not os.path.exists(mols_dir):
        raise Exception("MOL dir doesn't exists")
    
    writer = SDFWriter(sdf_file)
    for mol_file in tqdm(glob.glob(mols_dir + os.sep + "*.mol")):
        mol_name = os.path.basename(mol_file).split(".")[0]
        mol = Chem.MolFromMolFile(mol_file, removeHs=False)
        
        if mol is None:
            print("Invalid mol:", mol_name)
            continue
        
        mol.SetProp("_Name", mol_name)
        writer.write(mol)
    
    writer.close()