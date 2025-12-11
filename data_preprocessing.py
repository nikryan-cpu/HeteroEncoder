import pandas as pd
import numpy as np
from pandas import DataFrame
from rdkit import Chem
from utilities import get_rdkit_descriptors, replace_atoms, is_safe_molecule


def get_molecular_features(data : DataFrame):
    properties_data = data['SMILES'].apply(get_rdkit_descriptors)
    data = pd.concat([data, properties_data], axis=1).dropna()
    return data

def clean_data(data : DataFrame):
    data = data[data['SMILES'].apply(is_safe_molecule)]
    data["smiles_len"] = data['SMILES'].str.len()
    data = data[(data["smiles_len"] >= 35) & (data["smiles_len"] <= 75)]

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
    data = data.drop(columns=['smiles_len'])
    return data


def vectorize_smiles(smiles_list, charset, char_to_int, max_size):
    """
    Принимает список строк (SMILES), а не DataFrame.
    Возвращает float32 тензоры X и Y.
    """
    num_samples = len(smiles_list)
    vocab_size = len(charset)

    # Используем float32 для Keras
    one_hot = np.zeros((num_samples, max_size, vocab_size), dtype='float32')

    for i, smile in enumerate(smiles_list):
        # Start char '!'
        one_hot[i, 0, char_to_int['!']] = 1.0

        # Body
        for j, char in enumerate(smile):
            if j + 1 < max_size:
                if char in char_to_int:
                    one_hot[i, j + 1, char_to_int[char]] = 1.0

        # Padding with 'E' (End char) till the end
        # (В вашей логике вы заполняли всё оставшееся место символами E)
        for j in range(len(smile) + 1, max_size):
            one_hot[i, j, char_to_int['E']] = 1.0

    # X: с 0 по предпоследний, Y: с 1 по последний
    return one_hot[:, :-1, :], one_hot[:, 1:, :]


def preprocess_data(data: DataFrame):
    # Сначала считаем фичи на чистых SMILES (до замены атомов)
    data = get_molecular_features(data)
    # Потом чистим, канонизируем и меняем атомы
    data = clean_data(data)
    return data  # <--- ДОБАВЛЕН RETURN

def preprocess_data(data : DataFrame):
    data = get_molecular_features(data)
    data = clean_data(data)
    return data

