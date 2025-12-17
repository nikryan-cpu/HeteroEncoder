import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, MolFromSmarts

# Allowed atoms: H, C, N, O, F, P, S, Cl, Br, I
ALLOWED_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
PHARMACOPHORE_SMARTS = MolFromSmarts("O=C(N)c1c2ccccc2ncc1")


def is_safe_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if not mol: return False
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in ALLOWED_ATOMS:
                return False
        return True
    except:
        return False


def preprocess_embeddings(embeds_df_train, embeds_df_test):
    sc = StandardScaler().fit(embeds_df_train)
    return sc.transform(embeds_df_train), sc.transform(embeds_df_test), sc


def preprocess_energy(energy_train, energy_test):
    sc_energy = StandardScaler().fit(energy_train.to_numpy().reshape(-1, 1))
    return sc_energy.transform(energy_train.to_numpy().reshape(-1, 1)), \
        sc_energy.transform(energy_test.to_numpy().reshape(-1, 1)), sc_energy


def get_charset(data):
    unique_chars = set(''.join(data['SMILES']))
    unique_chars.update(['!', 'E'])
    return sorted(list(unique_chars))


def get_char_to_int(charset):
    return {c: i for i, c in enumerate(charset)}


def replace_atoms(smiles, reverse=False):
    # Shorten SMILES by replacing 2-char atoms/groups with single characters
    d = {
    '[C@@H]': '1', '[C@H]': '2', '[C@@]': '3', '[C@]': '4',
    '[S@@]': '5', '[S@]': '6', '[P@@]': '7', '[P@]': '8',

    '[NH3+]': 'a', '[NH2+]': 'b', '[nH+]': 'c', '[NH+]': 'd',
    '[N+]': 'e', '[N-]': 'f', '[n+]': 'g', '[n-]': 'h',

    '[O-]': 'i', '[O+]': 'j', '[OH+]': 'k',
    '[S-]': 'l', '[S+]': 'm', '[s+]': 'n',

    '[nH]': 'o', '[NH]': 'p',
    '[sH]': 'q', '[oH]': 'r',
    '[pH]': 's', '[PH]': 't',

    'Cl': 'X', 'Br': 'Y',
    'Si': 'Z', 'Se': 'W',

    '[C]': 'C', '[N]': 'N', '[O]': 'O', '[S]': 'S', '[P]': 'P',
    '[F]': 'F', '[I]': 'I', '[B]': 'B',
    '[c]': 'c', '[n]': 'n', '[o]': 'o', '[s]': 's',

    '[CH3]': 'C', '[CH2]': 'C', '[CH]': 'C', '[CH4]': 'C',
}
    for key in sorted(d.keys(), key=len, reverse=True):
        smiles = smiles.replace(key, d[key])
    return smiles


def reverse_replace_atoms(smiles, reverse=False):
    d = {
 '1': '[C@@H]',
 '2': '[C@H]',
 '3': '[C@@]',
 '4': '[C@]',
 '5': '[S@@]',
 '6': '[S@]',
 '7': '[P@@]',
 '8': '[P@]',
 'a': '[NH3+]',
 'b': '[NH2+]',
 'c': '[c]',
 'd': '[NH+]',
 'e': '[N+]',
 'f': '[N-]',
 'g': '[n+]',
 'h': '[n-]',
 'i': '[O-]',
 'j': '[O+]',
 'k': '[OH+]',
 'l': '[S-]',
 'm': '[S+]',
 'n': '[n]',
 'o': '[o]',
 'p': '[NH]',
 'q': '[sH]',
 'r': '[oH]',
 's': '[s]',
 't': '[PH]',
 'X': 'Cl',
 'Y': 'Br',
 'Z': 'Si',
 'W': 'Se',
 'N': '[N]',
 'O': '[O]',
 'S': '[S]',
 'P': '[P]',
 'F': '[F]',
 'I': '[I]',
 'B': '[B]'
    }
    for key in d:
        smiles = smiles.replace(key, d[key])
    return smiles


def clean_data(data: DataFrame):
    print("Processing: Filtering molecules...")
    data = data.dropna(subset=['SMILES'])

    def get_representations(s):
        try:
            if isinstance(s, str):
                s = s.replace('[c]', 'c').replace('[C]', 'C')
            mol = Chem.MolFromSmiles(s)
            if not mol or not is_safe_molecule(s): return None, None

            # Return Random (Input) and Canonical (Target/ID)
            return (Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False, doRandom=True),
                    Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))
        except:
            return None, None

    reps = data['SMILES'].apply(get_representations)
    data['SMILES'] = reps.apply(lambda x: x[0])
    data['CANONICAL_SMILES'] = reps.apply(lambda x: x[1])

    data = data.dropna(subset=['SMILES', 'CANONICAL_SMILES'])
    data = data.drop_duplicates(subset=['CANONICAL_SMILES'], keep='first')

    data["len"] = data['SMILES'].str.len()
    data = data[(data["len"] >= 35) & (data["len"] <= 75)]

    print(f"Final dataset size: {len(data)}")
    return data.drop(columns=['len'], errors='ignore')