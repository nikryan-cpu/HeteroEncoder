import pandas as pd

from data_preprocessing import clean_data

DATASET_PATH = 'PubChem_compound_smiles_substructure_C1=CC=C2C(=C1)C(=CC=N2)C(=O)N.csv'
OUTPUT_PATH = ''


def main():
    data = pd.read_csv(DATASET_PATH)
    preprocessed_data = clean_data(data)
