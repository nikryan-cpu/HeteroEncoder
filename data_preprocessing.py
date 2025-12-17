import pandas as pd
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski, QED, Crippen
from tqdm import tqdm
from utilities import replace_atoms, clean_data

tqdm.pandas()
rdBase.DisableLog('rdApp.warning')

CORE_SUBS_SMILES = "NC(=O)c1ccnc2ccccc12"
core_mol = Chem.MolFromSmiles(CORE_SUBS_SMILES)


def preprocess_data(data: pd.DataFrame):
    print("Preprocessing data (calculating physicochemical properties)...")
    data = clean_data(data)

    if 'PLANET_affinity' in data.columns:
        data['Energy'] = data['PLANET_affinity']

    def process_one_row(smiles):
        if not isinstance(smiles, str) or not smiles: return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        if core_mol and not mol.HasSubstructMatch(core_mol): return None

        try:
            canon = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            return [
                canon, canon,  # SMILES, CANONICAL
                Descriptors.MolWt(mol), Crippen.MolLogP(mol), Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol), Descriptors.TPSA(mol),
                Lipinski.NumRotatableBonds(mol), Lipinski.RingCount(mol), QED.qed(mol)
            ]
        except:
            return None

    results = data['SMILES'].progress_apply(process_one_row)
    data = data[results.notna()].copy()

    cols = ['SMILES', 'CANONICAL_SMILES', 'MolWt', 'LogP', 'NumHDonors',
            'NumHAcceptors', 'TPSA', 'NumRotatableBonds', 'RingCount', 'QED']

    features_df = pd.DataFrame(results.dropna().tolist(), columns=cols, index=data.index)

    # Avoid duplicate columns
    data = data.drop(columns=['SMILES', 'CANONICAL_SMILES'], errors='ignore')
    final_data = pd.concat([data, features_df], axis=1)

    # Tokenize
    final_data['SMILES'] = final_data['SMILES'].apply(replace_atoms)
    final_data['CANONICAL_SMILES'] = final_data['CANONICAL_SMILES'].apply(replace_atoms)

    print(f"Done. Total rows: {len(final_data)}")
    return final_data


def vectorize_smiles(smiles_list, charset, char_to_int, max_size):
    num_samples = len(smiles_list)
    vocab_size = len(charset)
    one_hot = np.zeros((num_samples, max_size, vocab_size), dtype='uint8')

    for i, smile in enumerate(smiles_list):
        one_hot[i, 0, char_to_int['!']] = 1  # Start
        for j, char in enumerate(smile):
            if j + 1 < max_size and char in char_to_int:
                one_hot[i, j + 1, char_to_int[char]] = 1
        for j in range(len(smile) + 1, max_size):
            one_hot[i, j, char_to_int['E']] = 1  # End

    return one_hot[:, :-1, :], one_hot[:, 1:, :]