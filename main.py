import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, Callback, ReduceLROnPlateau, EarlyStopping

from data_preprocessing import preprocess_data, vectorize_smiles
from utilities import get_charset, get_char_to_int, preprocess_embeddings, preprocess_energy
from model import build_hetero_encoder

# --- CONFIGURATION ---
DATASET_PATH = 'FINAL_Results_for_Model.csv'
PROCESSED_DATA_PATH = 'READY_TO_TRAIN_DATA_V3.csv'
ENERGY_COL_NAME = 'Energy'
MAX_LEN = 75
BATCH_SIZE = 64
EPOCHS = 150
BEST_MODEL_PATH = 'best_model.h5'
LAST_MODEL_PATH = 'last_model.h5'
LOG_FILE = 'training_log.csv'

# === GENERATOR ===
class MoleculeGenerator(tf.keras.utils.Sequence):
    def __init__(self, smiles, canon, features, energy, weights, charset, char_to_int, batch_size=128, max_len=75, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.smiles = smiles
        self.canon = canon
        self.features = features
        self.energy = energy
        self.charset = charset
        self.char_to_int = char_to_int
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.smiles))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.smiles) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_smiles = [self.smiles[k] for k in indexes]
        batch_canon = [self.canon[k] for k in indexes]

        X_s, Y_s = vectorize_smiles(batch_smiles, self.charset, self.char_to_int, self.max_len + 1)
        X_c, Y_c = vectorize_smiles(batch_canon, self.charset, self.char_to_int, self.max_len + 1)
        X_feat = self.features.iloc[indexes].values
        X_en = self.energy.iloc[indexes].values

        return (X_s, X_c, X_feat, X_en), (Y_s, Y_c)

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)

class ForceFlushLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'a') as f: os.fsync(f.fileno())

def main():
    print("Starting...")

    # 1. Load Data
    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"Loading cache: {PROCESSED_DATA_PATH}")
        data = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        print(f"Processing raw file: {DATASET_PATH}")
        if not os.path.exists(DATASET_PATH):
            print("ERROR: Dataset not found!")
            return
        raw_data = pd.read_csv(DATASET_PATH)
        data = preprocess_data(raw_data)
        data.to_csv(PROCESSED_DATA_PATH, index=False)

    # 2. Vocabulary
    data = data.reset_index(drop=True)
    full_text = data['SMILES'].astype(str).tolist() + data['CANONICAL_SMILES'].astype(str).tolist()
    charset = get_charset(pd.DataFrame({'SMILES': full_text}))
    char_to_int = get_char_to_int(charset)
    vocab_size = len(charset)

    with open('vocab.pkl', 'wb') as f:
        pickle.dump({'charset': charset, 'char_to_int': char_to_int}, f)

    with open('vocab.pkl', 'rb') as f:
        chars = pickle.load(f)
    print(chars)

    # 3. Split & Preprocess
    train_df, test_df = train_test_split(data, test_size=0.1, random_state=42)
    feat_cols = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds', 'RingCount', 'QED']
    X_feat_train, X_feat_test, sc_feat = preprocess_embeddings(train_df[feat_cols], test_df[feat_cols])
    X_en_train, X_en_test, sc_en = preprocess_energy(train_df[ENERGY_COL_NAME], test_df[ENERGY_COL_NAME])

    with open('scalers.pkl', 'wb') as f:
        pickle.dump({'sc_feat': sc_feat, 'sc_energy': sc_en}, f)

    # 4. Generators
    train_gen = MoleculeGenerator(
        train_df['SMILES'].tolist(), train_df['CANONICAL_SMILES'].tolist(),
        pd.DataFrame(X_feat_train, index=train_df.index),
        pd.Series(X_en_train.flatten(), index=train_df.index),
        None, charset, char_to_int, BATCH_SIZE, MAX_LEN
    )
    val_gen = MoleculeGenerator(
        test_df['SMILES'].tolist(), test_df['CANONICAL_SMILES'].tolist(),
        pd.DataFrame(X_feat_test, index=test_df.index),
        pd.Series(X_en_test.flatten(), index=test_df.index),
        None, charset, char_to_int, BATCH_SIZE, MAX_LEN, shuffle=False
    )

    # 5. Model Setup
    losses = ['categorical_crossentropy', 'categorical_crossentropy']
    metrics = ['accuracy', 'accuracy']
    model = None
    initial_epoch = 0

    if os.path.exists(LAST_MODEL_PATH):
        try:
            model = load_model(LAST_MODEL_PATH, compile=False)
            model.compile(optimizer='adam', loss=losses, metrics=metrics)
            print("✅ Loaded last_model.")
        except: pass

    if model is None and os.path.exists(BEST_MODEL_PATH):
        try:
            model = load_model(BEST_MODEL_PATH, compile=False)
            model.compile(optimizer='adam', loss=losses, metrics=metrics)
            print("✅ Loaded best_model.")
        except: pass

    if model is None:
        print("🆕 Building new model.")
        model = build_hetero_encoder(vocab_size, MAX_LEN, len(feat_cols), 128)
        model.compile(optimizer='adam', loss=losses, metrics=metrics)

    if os.path.exists(LOG_FILE) and os.stat(LOG_FILE).st_size > 0:
        try:
            log_df = pd.read_csv(LOG_FILE)
            if not log_df.empty:
                initial_epoch = log_df['epoch'].max() + 1
                print(f"Resuming from epoch: {initial_epoch}")
        except: pass

    # 6. Callbacks
    callbacks = [
        ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        ModelCheckpoint(LAST_MODEL_PATH, save_best_only=False, verbose=0),
        CSVLogger(LOG_FILE, append=True),
        ForceFlushLogger(),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    print(f"🚀 Starting training...")
    model.fit(train_gen, validation_data=val_gen, initial_epoch=initial_epoch, epochs=EPOCHS, callbacks=callbacks)
    print("🏁 Training finished.")

if __name__ == "__main__":
    main()