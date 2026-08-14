import torch
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
import os


class SmilesTokenizer:
    def __init__(self):
        self.stoi = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.itos = {0: '<pad>', 1: '<sos>', 2: '<eos>'}
        self.replacements = {'Cl': 'L', 'Br': 'R'}
        self.reverse_replacements = {'L': 'Cl', 'R': 'Br'}

    def fit(self, smiles_list):
        chars = set()
        for s in smiles_list:
            s = self._replace_two_letter(s)
            chars.update(set(s))
        for i, c in enumerate(sorted(list(chars)), start=3):
            self.stoi[c] = i
            self.itos[i] = c

    def _replace_two_letter(self, smi):
        for k, v in self.replacements.items():
            smi = smi.replace(k, v)
        return smi

    def _restore_two_letter(self, smi):
        for k, v in self.reverse_replacements.items():
            smi = smi.replace(k, v)
        return smi

    def encode(self, smi, max_len):
        smi = self._replace_two_letter(smi)
        tokens = [self.stoi['<sos>']] + [self.stoi[c] for c in smi] + [self.stoi['<eos>']]
        if len(tokens) < max_len:
            tokens += [self.stoi['<pad>']] * (max_len - len(tokens))
        return tokens[:max_len]

    def decode(self, tokens):
        smi = ""
        for t in tokens:
            if t == self.stoi['<eos>']: break
            if t == self.stoi['<sos>']: continue
            if t == self.stoi['<pad>']: continue
            smi += self.itos.get(t, '')
        return self._restore_two_letter(smi)

    def vocab_size(self):
        return len(self.stoi)


def get_reward(smiles, scaffold_smarts):
    """
    Reward: -5 (Invalid), +1 (Valid, no scaffold), +5 (Valid + Scaffold)
    """
    if not smiles: return -5.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return -5.0

    scaffold = Chem.MolFromSmarts(scaffold_smarts)
    if mol.HasSubstructMatch(scaffold):
        return 5.0
    return 1.0


def plot_training_log(log_path, title='Training Log'):
    if not os.path.exists(log_path):
        print(f"Log file {log_path} not found.")
        return

    df = pd.read_csv(log_path)
    plt.figure(figsize=(10, 5))

    if 'train_loss' in df.columns:
        plt.plot(df['epoch'], df['train_loss'], label='Total Loss')
        if 'recon_loss' in df.columns:
            plt.plot(df['epoch'], df['recon_loss'], label='Recon Loss', linestyle='--')
        plt.ylabel('Loss')

    elif 'avg_reward' in df.columns:
        plt.plot(df['epoch'], df['avg_reward'], label='Avg Reward', color='green')
        plt.ylabel('Reward')

    plt.xlabel('Epoch')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(log_path.replace('.csv', '.png'))
    print(f"Plot saved to {log_path.replace('.csv', '.png')}")
    plt.close()