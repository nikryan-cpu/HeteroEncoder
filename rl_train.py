import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import os  # Добавлено для проверки существования файла
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem, rdBase

from model import HeteroEncoderCVAE

rdBase.DisableLog('rdApp.*')

# Target core structure (Scaffold)
SCAFFOLD = "O=C(N)c1ccnc2ccccc12"
BEST_MODEL_TRAIN_PATH = 'pre-trained/model_best.pth'
BEST_MODEL_RL_PATH = 'pre-trained/model_rl_best.pth'
LAST_MODEL_RL_PATH = 'pre-trained/model_rl_last.pth'
INPUT_FILE = 'pre-trained/processed_data.pkl'
VOCAB_FILE = 'pre-trained/vocab.pkl'


# ==========================================
# 1. REWARD FUNCTION
# ==========================================
def get_reward_diversity(smiles, scaffold_smarts, known_db_set, epoch_history_set):
    """Calculates reward with a penalty for duplicates (self-repetition)."""
    if not smiles: return -5.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return -5.0

    try:
        canon_smi = Chem.MolToSmiles(mol, canonical=True)
    except:
        return -5.0

    scaffold = Chem.MolFromSmarts(scaffold_smarts)
    has_scaffold = mol.HasSubstructMatch(scaffold) if scaffold else False

    is_in_db = canon_smi in known_db_set
    is_in_epoch = canon_smi in epoch_history_set

    if has_scaffold:
        # High reward for novel molecules; lower for duplicates/known ones
        if is_in_db or is_in_epoch:
            return 2.0
        else:
            return 10.0

    return 0.5


# ==========================================
# 2. STATISTICS TRACKING
# ==========================================
def check_stats(smiles, scaffold_smarts, known_db_set, epoch_history_set):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return False, False, False, False

    canon = Chem.MolToSmiles(mol, canonical=True)
    has_scaf = False
    pat = Chem.MolFromSmarts(scaffold_smarts)
    if pat: has_scaf = mol.HasSubstructMatch(pat)

    is_new_db = canon not in known_db_set
    is_unique_epoch = canon not in epoch_history_set

    return True, has_scaf, is_new_db, is_unique_epoch


# ==========================================
# 3. RL TRAINING LOOP
# ==========================================
def run_rl(
        epochs=10,
        batch_size=128,
        noise_scale=0.2,
        target_energy=-12.0,
        train_energy_threshold=-10.0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running RL with Diversity Penalty ---")

    # Load resources
    with open(VOCAB_FILE, 'rb') as f:
        tokenizer = pickle.load(f)

    model = HeteroEncoderCVAE(tokenizer.vocab_size()).to(device)

    # Freeze Encoder: Only fine-tune the Decoder
    for param in model.encoder_gru.parameters(): param.requires_grad = False
    for param in model.embedding.parameters(): param.requires_grad = False
    for param in model.fc_mu.parameters(): param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    start_epoch = 1
    best_reward = -float('inf')

    if os.path.exists(LAST_MODEL_RL_PATH):
        print(f"Found checkpoint: {LAST_MODEL_RL_PATH}. Resuming training...")
        checkpoint = torch.load(LAST_MODEL_RL_PATH, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_reward = checkpoint.get('best_reward', -float('inf'))
            print(f"Resuming from Epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("Warning: Loaded weights only from RL checkpoint (no optimizer state).")

    elif os.path.exists(BEST_MODEL_TRAIN_PATH):
        print("No RL checkpoint found. Loading pre-trained base model...")
        model.load_state_dict(torch.load(BEST_MODEL_TRAIN_PATH, map_location=device))
    else:
        print("Warning: No models found! Starting from scratch.")

    # Dataset preparation
    df_all = pd.read_pickle(INPUT_FILE)
    known_db_set = set(df_all['CANONICAL_SMILES'].values)

    # Use high-affinity molecules as training seeds
    df_train = df_all[df_all['Energy'] <= train_energy_threshold].copy()
    if len(df_train) == 0: return

    X_smiles = torch.tensor([tokenizer.encode(s, 85) for s in df_train['CANONICAL_SMILES']]).long().to(device)
    descriptors_np = np.stack(df_train['descriptors_norm'].values)
    X_desc = torch.from_numpy(descriptors_np).float().to(device)

    dataset = TensorDataset(X_smiles, X_desc)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(start_epoch, epochs + 1):
        total_reward, batches_processed, total_mols = 0, 0, 0
        stat_valid, stat_scaf, stat_novel_db, stat_unique_epoch = 0, 0, 0, 0

        # Reset uniqueness history for the new epoch
        epoch_history_set = set()

        loop = tqdm(loader, desc=f"Ep {epoch}")

        for batch in loop:
            smi_in, desc_in = batch
            current_bs = smi_in.size(0)
            batches_processed += 1
            total_mols += current_bs

            optimizer.zero_grad()

            # 1. Latent space encoding
            with torch.no_grad():
                embedded = model.embedding(smi_in)
                _, h_n = model.encoder_gru(embedded)
                h_n = h_n.squeeze(0)
                combined = torch.cat([h_n, desc_in], dim=1)
                mu = model.fc_mu(combined)

            # 2. Sampling with Noise and Conditioning
            z = mu + torch.randn_like(mu) * noise_scale
            target_tensor = torch.full((current_bs, 1), target_energy).float().to(device)
            z_cond = torch.cat([z, target_tensor], dim=1)
            hidden = model.fc_z_to_hidden(z_cond).unsqueeze(0)

            # 3. Autoregressive Molecule Generation
            inp = torch.tensor([[tokenizer.stoi['<sos>']]] * current_bs).to(device)
            log_probs = []
            tokens_batch = [[] for _ in range(current_bs)]

            for _ in range(85):
                emb = model.embedding(inp)
                out, hidden = model.decoder_gru(emb, hidden)
                logits = model.fc_out(out.squeeze(1))
                probs = F.softmax(logits, dim=1)
                m = torch.distributions.Categorical(probs)
                action = m.sample()
                log_probs.append(m.log_prob(action))
                inp = action.unsqueeze(1)
                for i, t in enumerate(action):
                    tokens_batch[i].append(t.item())

            # 4. Reward calculation and stats tracking
            rewards = []
            for i in range(current_bs):
                smi = tokenizer.decode(tokens_batch[i])
                r = get_reward_diversity(smi, SCAFFOLD, known_db_set, epoch_history_set)
                rewards.append(r)

                is_val, has_sc, is_new_db, is_uniq_ep = check_stats(smi, SCAFFOLD, known_db_set, epoch_history_set)
                if is_val:
                    stat_valid += 1
                    canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
                    epoch_history_set.add(canon_smi)

                if has_sc: stat_scaf += 1
                if has_sc and is_new_db: stat_novel_db += 1
                if has_sc and is_uniq_ep: stat_unique_epoch += 1

            # 5. Policy Gradient (REINFORCE) update
            r_tensor = torch.tensor(rewards).float().to(device)
            r_norm = (r_tensor - r_tensor.mean()) / (r_tensor.std() + 1e-8)
            log_probs_stack = torch.stack(log_probs).transpose(0, 1)

            policy_loss = [-log_probs_stack[i].sum() * r_norm[i] for i in range(current_bs)]
            loss = torch.stack(policy_loss).mean()
            loss.backward()
            optimizer.step()

            total_reward += np.mean(rewards)
            loop.set_postfix({'Reward': f"{np.mean(rewards):.2f}", 'Valid': f"{stat_valid / total_mols:.0%}"})

        # Epoch Summary
        avg_reward = total_reward / batches_processed
        print(f"\nEpoch {epoch} Summary: Reward: {avg_reward:.4f}, Novelty: {stat_novel_db / total_mols:.1%}")

        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_reward': best_reward
        }

        if avg_reward > best_reward:
            best_reward = avg_reward
            checkpoint_data['best_reward'] = best_reward
            torch.save(checkpoint_data, BEST_MODEL_RL_PATH)
            print(f"New best model saved to {BEST_MODEL_RL_PATH}")

        torch.save(checkpoint_data, LAST_MODEL_RL_PATH)