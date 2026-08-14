from chemplus import sdf
from chemplus import draw
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import PandasPatcher
PandasTools.RenderImagesInAllDataFrames()

PandasPatcher.PrintAsImageString = draw.draw_mol
Chem.Mol._repr_html_ = draw.draw_mol


def df_from_sdf(filename, name_field="_Name", new_name_field="Name", mol_column="Mol", mol_names=None,
                properties=None, sanitize=True, removeHs=False, strictParsing=True, raise_nones=True):
    if mol_names is not None and not isinstance(mol_names, (list, tuple)):
        raise Exception("Please provide the parameter mol_names in the format of a Python List or Tuple")
    
    mols_list = []
    
    for mol in sdf.SDFReader(filename, sanitize=sanitize, removeHs=removeHs, strictParsing=strictParsing):
        if mol is None:
            if raise_nones:
                raise Exception("Mol is None")
            else:
                continue
        if mol_names is None or mol.GetProp(name_field) in mol_names:
            if not properties:
                mol_row = mol.GetPropsAsDict()
            else:
                mol_row = dict((prop_name, mol.GetProp(prop_name)) for prop_name in mol.GetPropNames() if prop_name in properties)
            mol_row[name_field] = mol.GetProp(name_field)
            for prop_name in mol.GetPropNames():
                mol.ClearProp(prop_name)
            mol_row[mol_column] = mol
            mols_list.append(mol_row)
    
    if not mol_names is None:
        loaded_names = [mol_row[name_field] for mol_row in mols_list]
        for mol_name in mol_names:
            if mol_name not in loaded_names:
                mols_list.append({name_field : mol_name, mol_column : "Not Found"})
                
    df = pd.DataFrame(mols_list)
    df.rename(columns={name_field : new_name_field}, inplace=True)
    if mol_names is not None:
        df = df.merge(pd.Series(mol_names, name=new_name_field), how='right')
    new_columns_order = [new_name_field, mol_column] + list(set(df.columns.tolist()) - set([new_name_field, mol_column]))
    df = df[new_columns_order]
    return df

def df_to_sdf(df, filename, name_column="Name", mol_column="Mol", properties=None):
    writer = sdf.SDFWriter(filename)
    
    if properties is None:
        properties = []
    else:
        properties = list(properties)
        writer.set_props(properties)
    
    for index, row in df.iterrows():
        if row[mol_column] is None:
            continue
        mol = Chem.Mol(row[mol_column])
        
        for prop in properties:
            value = row[prop]
            mol.SetProp(prop, str(value))
        
        if name_column is None:
            mol.SetProp("_Name", str(index))
        else:
            mol.SetProp("_Name", str(row[name_column]))
        
        writer.write(mol)
    
    writer.close()
