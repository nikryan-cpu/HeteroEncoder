import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


def preprocess_embeddings(embeds_df_train, embeds_df_test):     # normalize molecular features
    sc = StandardScaler().fit(embeds_df_train)
    embeds_df_train = sc.transform(embeds_df_train)
    embeds_df_test = sc.transform(embeds_df_test)
    return embeds_df_train, embeds_df_test, sc

def preprocess_energy(energy_df_train, energy_df_test):     # normalize molecular binding energy
    sc_energy = StandardScaler().fit(energy_df_train.to_numpy().reshape(-1, 1))
    energy_df_train = sc_energy.transform(energy_df_train.to_numpy().reshape(-1, 1))
    energy_df_test = sc_energy.transform(energy_df_test.to_numpy().reshape(-1, 1))
    return energy_df_train, energy_df_test, sc_energy

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

def split_data(data : DataFrame):
    train = data.sample(frac=0.8, random_state=100)
    test = data.drop(train.index)
    return train, test

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

def get_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return pd.Series({
        'MolWt': Descriptors.MolWt(mol),  # Molecular Weight
        'LogP': Descriptors.MolLogP(mol),  # Lipophilicity
        'NumHDonors': Descriptors.NumHDonors(mol),  # Hydrogen Bond Donors
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),  # Hydrogen Bond Acceptors
        'TPSA': Descriptors.TPSA(mol),  # Polar Surface Area (Permeability)
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),  # Flexibility
        'RingCount': Descriptors.RingCount(mol),  # Number of Rings
        'QED': QED.qed(mol)  # Quantitative Estimation of Drug-likeness (0..1)
    })