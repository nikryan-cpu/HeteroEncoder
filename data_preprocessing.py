import pandas as pd
import numpy as np
import pickle
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from utilities import SmilesTokenizer

# Target core structure for filtering
SCAFFOLD = "O=C(N)c1ccnc2ccccc12"


def run_preprocessing(input_csv='dataset.csv'):
    # Check if processed files already exist to avoid redundant work
    if os.path.exists('processed_data.pkl') and os.path.exists('vocab.pkl'):
        print("Preprocessing already done. Skipping...")
        return

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    # 1. Remove duplicate SMILES
    df = df.drop_duplicates(subset=['CANONICAL_SMILES'])

    # 2. Filter molecules by scaffold presence
    scaffold_mol = Chem.MolFromSmarts(SCAFFOLD)
    valid_data = []

    print("Filtering molecules and calculating descriptors...")
    for _, row in df.iterrows():
        smi = row['CANONICAL_SMILES']
        mol = Chem.MolFromSmiles(smi)

        if mol and mol.HasSubstructMatch(scaffold_mol):
            # Calculate physical-chemical properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)

            valid_data.append({
                'CANONICAL_SMILES': smi,
                'Energy': row['Energy'],
                'descriptors': [mw, logp, tpsa]
            })

    df_clean = pd.DataFrame(valid_data)

    # 3. Min-Max Normalization for descriptors
    descs = np.array(df_clean['descriptors'].tolist())
    d_min, d_max = descs.min(axis=0), descs.max(axis=0)
    descs_norm = (descs - d_min) / (d_max - d_min + 1e-6)
    df_clean['descriptors_norm'] = list(descs_norm)

    # 4. Tokenization and Vocabulary creation
    tokenizer = SmilesTokenizer()
    tokenizer.fit(df_clean['CANONICAL_SMILES'])

    # Save processed data, tokenizer, and scaler parameters
    df_clean.to_pickle('processed_data.pkl')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    np.save('scaler_params.npy', {'min': d_min, 'max': d_max})

    print(f"Done! Saved {len(df_clean)} molecules. Vocab size: {tokenizer.vocab_size()}")


if __name__ == "__main__":
    run_preprocessing()