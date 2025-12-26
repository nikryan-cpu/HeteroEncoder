import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import pickle
import os
import csv
from tqdm import tqdm
from model import HeteroEncoderCVAE

def loss_function(logits, x, mu, logvar, kld_weight=0.005):
    """
    Standard VAE loss: Reconstruction (CrossEntropy) + KL Divergence.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = x[:, 1:].contiguous()
    B, L, V = shift_logits.shape

    recon_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_targets.view(-1),
        ignore_index=0,
        reduction='mean'
    )

    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon_loss + kld_weight * kld

    return total_loss, recon_loss, kld

def run_training(epochs=50, batch_size=128, lr=1e-3, kld_weight=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # --- 1. DATA LOADING ---
    if not os.path.exists('processed_data.pkl'):
        print("Error: processed_data.pkl not found.")
        return

    df = pd.read_pickle('processed_data.pkl')
    with open('vocab.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    max_len = 85
    X_smiles = torch.tensor([tokenizer.encode(s, max_len) for s in df['CANONICAL_SMILES']]).long()
    descriptors_np = np.stack(df['descriptors_norm'].values)
    X_desc = torch.from_numpy(descriptors_np).float()
    X_energy = torch.tensor(df['Energy'].values).float().unsqueeze(1)

    dataset = TensorDataset(X_smiles, X_desc, X_energy)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 2. INITIALIZATION ---
    model = HeteroEncoderCVAE(tokenizer.vocab_size()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- 3. RESUME CHECKPOINT ---
    checkpoint_path = 'checkpoint_last.pth'
    start_epoch = 1
    best_val_loss = float('inf')

    if os.path.exists(checkpoint_path):
        print(f"--> Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    else:
        print("--> Starting training from scratch.")

    # --- 4. LOGGING SETUP ---
    epoch_log_file = 'training_log.csv'
    detailed_log_file = 'training_log_detailed.csv'

    if start_epoch == 1:
        with open(epoch_log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Val_Loss'])
        with open(detailed_log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['Epoch', 'Batch', 'Loss', 'Recon', 'KLD'])

    # --- 5. TRAINING LOOP ---
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss_accum = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}")

        for batch_idx, batch in enumerate(loop):
            smi, desc, eng = [b.to(device) for b in batch]

            optimizer.zero_grad()
            logits, mu, logvar = model(smi, desc, eng)
            loss, recon, kld = loss_function(logits, smi, mu, logvar, kld_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss_accum += loss.item()

            # Batch logging
            with open(detailed_log_file, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, batch_idx, f"{loss.item():.4f}", f"{recon.item():.4f}", f"{kld.item():.4f}"])

            loop.set_postfix(loss=f"{loss.item():.4f}")

        # --- VALIDATION ---
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for batch in val_loader:
                smi, desc, eng = [b.to(device) for b in batch]
                logits, mu, logvar = model(smi, desc, eng)
                loss, _, _ = loss_function(logits, smi, mu, logvar, kld_weight)
                val_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)
        avg_val_loss = val_loss_accum / len(val_loader)

        print(f"Epoch {epoch} Results | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save epoch stats
        with open(epoch_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{avg_train_loss:.5f}", f"{avg_val_loss:.5f}"])

        # --- SAVE CHECKPOINTS ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model_best.pth')
            print(f"--> New Best Model Saved (Val Loss: {best_val_loss:.4f})")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, 'checkpoint_last.pth')
        torch.save(model.state_dict(), 'model_last.pth')

if __name__ == "__main__":
    run_training()