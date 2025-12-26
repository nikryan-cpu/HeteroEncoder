import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem
from rdkit import rdBase

# Import your model architecture
from model import HeteroEncoderCVAE

# Disable RDKit logs
rdBase.DisableLog('rdApp.*')

# ==========================================
# 1. SETTINGS
# ==========================================
SETTINGS = {
    'input_file': 'processed_data.pkl',  # Contains ALL known data (Train + Val)
    'vocab_file': 'vocab.pkl',
    'model_path': 'model_best.pth',
    'model_rl_path': 'model_rl_best.pth',
    'batch_size': 100
}


# ==========================================
# 2. PHARMACOPHORE CHECK
# ==========================================
def check_pharmacophore(mol):
    if mol is None: return False
    # Example SMARTS
    pat = Chem.MolFromSmarts("O=C(N)c1ccnc2ccccc12")
    return mol.HasSubstructMatch(pat) if pat else False


# ==========================================
# 3. GENERATION FUNCTION
# ==========================================
def run_generation(total_attempts=1000, target_energy=-12, seed_energy_threshold=-10, noise=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== STARTING EXPERIMENT: {total_attempts} ATTEMPTS ===")

    # --- A. Load Resources ---
    with open(SETTINGS['vocab_file'], 'rb') as f:
        tokenizer = pickle.load(f)

    model = HeteroEncoderCVAE(vocab_size=tokenizer.vocab_size()).to(device)
    if os.path.exists(SETTINGS['model_rl_path']):
        model.load_state_dict(torch.load(SETTINGS['model_rl_path'], map_location=device))
        print(f"Model loaded from {SETTINGS['model_rl_path']}")

    elif os.path.exists(SETTINGS['model_path']):
        model.load_state_dict(torch.load(SETTINGS['model_path'], map_location=device))
        print(f"Model loaded from {SETTINGS['model_path']}")
    else:
        print(f"Error: Model file not found.")
        return

    model.eval()

    if not os.path.exists(SETTINGS['input_file']):
        print(f"Error: Data file not found.")
        return

    df = pd.read_pickle(SETTINGS['input_file'])

    # --- NOVELTY CHECK PREPARATION ---
    print("Indexing known molecules (Train + Val)...")
    # This set contains ALL molecules previously seen by the model
    known_smiles_set = set(df['CANONICAL_SMILES'].values)
    print(f"Cached {len(known_smiles_set)} known molecules.")

    # --- B. Prepare Seeds ---
    good_seeds = df[df['Energy'] < seed_energy_threshold]

    if len(good_seeds) == 0:
        print("Error: No suitable seeds found.")
        return

    # Sample with replacement to match total_attempts
    selected_seeds = good_seeds.sample(n=total_attempts, replace=True)
    print(f"Selected {len(selected_seeds)} seeds for generation.")

    # Create TensorDataset
    X_smiles = torch.tensor([tokenizer.encode(s, 85) for s in selected_seeds['CANONICAL_SMILES']]).long().to(device)
    descriptors_np = np.stack(selected_seeds['descriptors_norm'].values)
    X_desc = torch.from_numpy(descriptors_np).float().to(device)

    dataset = TensorDataset(X_smiles, X_desc)
    loader = DataLoader(dataset, batch_size=SETTINGS['batch_size'], shuffle=False)

    # --- C. Generation Loop ---
    # Statistics counters
    stats = {
        'valid': 0,
        'pharmacophore': 0,
        'rejected_known': 0,  # Rejected because it's in training data
        'rejected_duplicate': 0,  # Rejected because we just generated it
        'saved': 0
    }

    # Set to track uniqueness WITHIN the current generation batch
    unique_new_molecules = set()
    results_list = []

    print("Generating...")
    with torch.no_grad():
        for smi_in, desc_in in tqdm(loader, desc="Progress"):
            batch_size = smi_in.size(0)

            # 1. Encoder
            embedded = model.embedding(smi_in)
            _, h_n = model.encoder_gru(embedded)
            combined = torch.cat([h_n.squeeze(0), desc_in], dim=1)
            mu = model.fc_mu(combined)

            # 2. Decoder with Noise (Variation)
            z = mu + torch.randn_like(mu) * noise
            target_eng = torch.full((batch_size, 1), target_energy).to(device)
            z_cond = torch.cat([z, target_eng], dim=1)
            hidden = model.fc_z_to_hidden(z_cond).unsqueeze(0)

            # 3. Generate Tokens
            curr = torch.tensor([[tokenizer.stoi['<sos>']]] * batch_size).to(device)
            seqs = torch.zeros(batch_size, 85).long().to(device)

            for t in range(85):
                emb_dec = model.embedding(curr)
                out, hidden = model.decoder_gru(emb_dec, hidden)
                logits = model.fc_out(out.squeeze(1))
                probs = F.softmax(logits / 0.8, dim=1)
                curr = torch.multinomial(probs, 1)
                seqs[:, t] = curr.squeeze(1)

            # 4. Process Batch
            for i in range(batch_size):
                toks = seqs[i].tolist()
                try:
                    toks = toks[:toks.index(tokenizer.stoi['<eos>'])]
                except:
                    pass
                smi = tokenizer.decode(toks)

                mol = Chem.MolFromSmiles(smi)

                if mol is not None:
                    stats['valid'] += 1
                    canon_smi = Chem.MolToSmiles(mol, canonical=True)

                    # --- FILTER 1: Is it in the training/val set? ---
                    if canon_smi in known_smiles_set:
                        stats['rejected_known'] += 1
                        continue  # SKIP

                    # --- FILTER 2: Did we already generate it in this run? ---
                    if canon_smi in unique_new_molecules:
                        stats['rejected_duplicate'] += 1
                        continue  # SKIP

                    # If we are here, the molecule is NOVEL and UNIQUE
                    has_pharma = check_pharmacophore(mol)
                    if has_pharma:
                        stats['pharmacophore'] += 1

                    unique_new_molecules.add(canon_smi)
                    stats['saved'] += 1

                    results_list.append({
                        'SMILES': canon_smi,
                        'Has_Pharmacophore': has_pharma
                    })

    # --- D. Final Report ---
    print("\n" + "=" * 40)
    print("EXPERIMENT RESULTS")
    print("=" * 40)
    print(f"Total attempts:          {total_attempts}")
    print(f"Valid molecules:         {stats['valid']}")
    print("-" * 20)
    print(f"Rejected (Known in DB):  {stats['rejected_known']}")
    print(f"Rejected (Duplicates):   {stats['rejected_duplicate']}")
    print("-" * 20)
    print(f"SAVED (Novel & Unique):  {stats['saved']}")
    print(f"  > With Pharmacophore:  {stats['pharmacophore']}")
    print("=" * 40)

    # Save ONLY novel and unique
    if results_list:
        res_df = pd.DataFrame(results_list)
        filename = 'novel_molecules.csv'
        res_df.to_csv(filename, index=False)
        print(f"Saved {len(res_df)} novel molecules to {filename}")
    else:
        print("No novel molecules found.")


if __name__ == "__main__":
    run_generation(total_attempts=1000)