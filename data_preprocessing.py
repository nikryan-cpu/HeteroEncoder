import pandas as pd
import numpy as np
from pandas import DataFrame
from rdkit import Chem
from utilities import get_rdkit_descriptors, replace_atoms



def get_molecular_features(data : DataFrame):
    properties_data = data['SMILES'].apply(get_rdkit_descriptors)
    data = pd.concat([data, properties_data], axis=1).dropna()
    return data

def clean_data(data : DataFrame):
    data["smiles_len"] = data['SMILES'].str.len()
    data = data[(data["smiles_len"] >= 35) & (data["smiles_len"] <= 75)]
    data = data[['SMILES']]

    def canonicalize(s):
        mol = Chem.MolFromSmiles(s)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        else:
            return None
    data['CANONICAL_SMILES'] = data['SMILES'].apply(canonicalize)
    data['SMILES'] = data['SMILES'].apply(replace_atoms)
    data['CANONICAL_SMILES'] = data['CANONICAL_SMILES'].apply(replace_atoms)
    # Delete None from data
    data = data.dropna(subset=['CANONICAL_SMILES'])
    return data

def vectorize_from_smiles(data : DataFrame, charset, char_to_int, max_size : int):
    smiles_list = data['SMILES'].to_list()
    one_hot = np.zeros((len(smiles_list), max_size, len(charset)), dtype=bool)
    for i in range(len(smiles_list)):
        one_hot[i, 0, char_to_int['!']] = True  # Adding Start char
        for j in range(len(smiles_list[i])):
            one_hot[i, j + 1, char_to_int[smiles_list[i][j]]] = True
        for j in range(len(smiles_list[i]) + 1, max_size):  # Adding End char
            one_hot[i, j, char_to_int['E']] = True
    return one_hot[:, :-1, :], one_hot[:, 1:, :]

def preprocess_data(data : DataFrame):
    data = get_molecular_features(data)
    data = clean_data(data)

