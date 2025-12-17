import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from data_preprocessing import preprocess_data, vectorize_smiles
from utilities import get_charset, get_char_to_int, preprocess_embeddings, preprocess_energy
from model import build_hetero_encoder

# --- CONFIGURATION ---
DATASET_PATH = 'FINAL_Results_for_Model.csv'  # or your raw file
PROCESSED_DATA_PATH = 'READY_TO_TRAIN_DATA_V3.csv'
ENERGY_COL = 'Energy'
MAX_LEN = 75
BATCH_SIZE = 128
EPOCHS = 50


def train():
    # 1. Load Data
    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"Loading cache: {PROCESSED_DATA_PATH}")
        data = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        print(f"Processing raw file: {DATASET_PATH}")
        if not os.path.exists(DATASET_PATH):
            print("ERROR: Dataset file not found.")
            return
        raw_data = pd.read_csv(DATASET_PATH)

        # Rename column if needed
        if ENERGY_COL not in raw_data.columns and len(raw_data.columns) > 1:
            print(f"Warning: '{ENERGY_COL}' not found. Using 2nd column as Energy.")
            raw_data.rename(columns={raw_data.columns[1]: ENERGY_COL}, inplace=True)

        data = preprocess_data(raw_data)
        data.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Data ready: {len(data)} samples")

    # 2. Vocabulary
    data = data.reset_index(drop=True)
    full_text = data['SMILES'].astype(str).tolist() + data['CANONICAL_SMILES'].astype(str).tolist()

    charset = get_charset(pd.DataFrame({'SMILES': full_text}))
    char_to_int = get_char_to_int(charset)
    vocab_size = len(charset)
    print(f"Vocab size: {vocab_size}")

    # Save vocab for generation
    with open('vocab.pkl', 'wb') as f:
        pickle.dump({'charset': charset, 'char_to_int': char_to_int}, f)

    # 3. Split Data
    train_df, test_df = train_test_split(data, test_size=0.1, random_state=42)

    # 4. Vectorization (One-Hot)
    # Using MAX_LEN + 1 because vectorize_smiles splits into Input (t) and Target (t+1)
    print("Vectorizing SMILES...")
    X_s_train, Y_s_train = vectorize_smiles(train_df['SMILES'].tolist(), charset, char_to_int, MAX_LEN + 1)
    X_s_test, Y_s_test = vectorize_smiles(test_df['SMILES'].tolist(), charset, char_to_int, MAX_LEN + 1)

    X_c_train, Y_c_train = vectorize_smiles(train_df['CANONICAL_SMILES'].tolist(), charset, char_to_int, MAX_LEN + 1)
    X_c_test, Y_c_test = vectorize_smiles(test_df['CANONICAL_SMILES'].tolist(), charset, char_to_int, MAX_LEN + 1)

    # 5. Normalize Features & Energy
    feat_cols = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds', 'RingCount', 'QED']

    X_feat_train, X_feat_test, sc_feat = preprocess_embeddings(train_df[feat_cols], test_df[feat_cols])
    X_en_train, X_en_test, sc_en = preprocess_energy(train_df[ENERGY_COL], test_df[ENERGY_COL])

    # Save scalers for generation
    with open('scalers.pkl', 'wb') as f:
        pickle.dump({'sc_feat': sc_feat, 'sc_energy': sc_en}, f)

    # 6. Build Model
    print("Building model...")
    model = build_hetero_encoder(vocab_size, MAX_LEN, len(feat_cols), 128)

    losses = ['categorical_crossentropy', 'categorical_crossentropy']
    metrics = ['accuracy', 'accuracy']
    model.compile(optimizer='adam', loss=losses, metrics=metrics)

    # 7. Training
    callbacks = [
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    print("Starting training...")
    model.fit(
        x=[X_s_train, X_c_train, X_feat_train, X_en_train],
        y=[Y_s_train, Y_c_train],
        validation_data=(
            [X_s_test, X_c_test, X_feat_test, X_en_test],
            [Y_s_test, Y_c_test]
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print("Training complete!")


if __name__ == "__main__":
    train()