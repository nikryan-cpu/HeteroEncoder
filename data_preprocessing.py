import string

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from rdkit import Chem


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
    # Delete None from data
    data = data.dropna(subset=['CANONICAL_SMILES'])
    return data


def split_data(data : DataFrame):
    train = data.sample(frac=0.8, random_state=100)
    test = data.drop(train.index)
    return train, test

def replace_atoms(smiles, reverse=False):
    d = {'Br': 'X',
         'Cl': 'Y',
         'Se': 'Z',
         'br': 'x',
         'cl': 'y',
         'se': 'z',
         '-]': 'V]'
         }
    new_smiles = smiles
    for key in d:
        new_smiles = new_smiles.replace(key, d[key])
    return new_smiles

def reverse_replace_atoms(smiles, reverse=False):
    d = {
    'X': 'Br',
    'Y': 'Cl',
    'Z': 'Se',
    'x': 'br',
    'y': 'cl',
    'z': 'se',
    'V]': '-]'
    }
    new_smiles = smiles
    for key in d:
        new_smiles = new_smiles.replace(key, d[key])
    return new_smiles

def get_charset(data): # Get all possible char's
    unique_chars = set(''.join(data['SMILES']))
    unique_chars.add('!')  # Start token
    unique_chars.add('E')  # End token
    charset = sorted(list(unique_chars))
    return charset

def get_char_to_int(charset):
    char_to_int = dict()
    for i in range(len(charset)):
        char_to_int[charset[i]] = i
    return char_to_int

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
