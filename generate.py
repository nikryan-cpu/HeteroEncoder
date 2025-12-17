import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
import pickle
import os
from rdkit import Chem
from utilities import reverse_replace_atoms

# --- SETTINGS ---
MODEL_PATH = 'best_model.h5'
DATA_PATH = 'READY_TO_TRAIN_DATA_V3.csv'
VOCAB_PATH = 'vocab.pkl'
SCALERS_PATH = 'scalers.pkl'

MAX_LEN = 75
TARGET_ENERGY = -12.0
NUM_MOLECULES = 200
NOISE_LEVEL = 0.2  # Slightly increased for diversity
TEMPERATURE = 0.8  # < 1.0 = More precise, > 1.0 = More random


def sample_with_temperature(logits, temperature=1.0):
    # Logits are log-probabilities from the dense layer
    logits = np.array(logits).astype('float64')
    # Apply temperature
    logits = logits / temperature
    # Softmax
    exp_preds = np.exp(logits)
    preds = exp_preds / np.sum(exp_preds)
    # Draw from distribution
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate():
    print("=== MOLECULE GENERATION ===")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found ({MODEL_PATH})")
        return

    model = load_model(MODEL_PATH, compile=False)

    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    char_to_int = vocab['char_to_int']
    int_to_char = {i: c for c, i in char_to_int.items()}
    vocab_size = len(char_to_int)

    with open(SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
        sc_feat = scalers['sc_feat']
        sc_en = scalers.get('sc_energy', scalers.get('sc_en'))

    # 1. Prepare Seed Data
    df = pd.read_csv(DATA_PATH).sample(NUM_MOLECULES)

    s_in = np.zeros((NUM_MOLECULES, MAX_LEN, vocab_size))
    c_in = np.zeros((NUM_MOLECULES, MAX_LEN, vocab_size))

    for i, smiles in enumerate(df['SMILES']):
        full_s = "!" + str(smiles)
        for t, char in enumerate(full_s):
            if t < MAX_LEN and char in char_to_int:
                s_in[i, t, char_to_int[char]] = 1
                c_in[i, t, char_to_int[char]] = 1

    feat_cols = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds', 'RingCount', 'QED']
    feat_in = sc_feat.transform(df[feat_cols].values)

    en_vals = df[['Energy']].values if 'Energy' in df.columns else np.zeros((NUM_MOLECULES, 1))
    en_in = sc_en.transform(en_vals)

    # 2. Encoder (Get Latent Z)
    try:
        encoder = Model(inputs=model.input, outputs=model.get_layer('z_layer').output)
    except ValueError:
        print("❌ Error: 'z_layer' not found in model.")
        return

    z_gen = encoder.predict([s_in, c_in, feat_in, en_in], verbose=0)

    # 3. Modify Latent Space (Add Noise + Set Target Energy)
    z_gen += np.random.normal(0, NOISE_LEVEL, size=z_gen.shape)
    target_val = sc_en.transform([[TARGET_ENERGY]])[0][0]
    z_gen[:, -1] = target_val

    # 4. Decoder Initialization
    init_h = model.get_layer('dec_s_bn_h')(model.get_layer('dec_s_init_h')(z_gen)).numpy()
    init_c = model.get_layer('dec_s_bn_c')(model.get_layer('dec_s_init_c')(z_gen)).numpy()

    # 5. Generation Loop
    inference_lstm = layers.LSTM(128, return_sequences=True, return_state=True)
    # Warmup to build weights
    dummy = tf.zeros((1, 1, vocab_size))
    inference_lstm(dummy, initial_state=[tf.zeros((1, 128)), tf.zeros((1, 128))])
    inference_lstm.set_weights(model.get_layer('lstm_s').get_weights())

    dense_out = model.get_layer('out_smiles')

    generated_smiles = []
    print(f"Generating {NUM_MOLECULES} molecules (Temp={TEMPERATURE})...")

    for i in range(NUM_MOLECULES):
        curr_h = init_h[i:i + 1]
        curr_c = init_c[i:i + 1]
        curr_input = np.zeros((1, 1, vocab_size))
        curr_input[0, 0, char_to_int['!']] = 1.0

        mol = ""
        for _ in range(MAX_LEN):
            lstm_out, h_out, c_out = inference_lstm(curr_input, initial_state=[curr_h, curr_c])
            logits = dense_out(lstm_out)[0, 0].numpy()

            # --- KEY CHANGE: Temperature Sampling ---
            idx = sample_with_temperature(logits, TEMPERATURE)
            char = int_to_char[idx]

            if char == 'E': break
            mol += char

            curr_input = np.zeros((1, 1, vocab_size))
            curr_input[0, 0, idx] = 1.0
            curr_h, curr_c = h_out, c_out

        generated_smiles.append(mol)

    # 6. Validate & Save
    valid_mols = []
    for s in generated_smiles:
        dec = reverse_replace_atoms(s).replace('[c]', 'c').replace('[C]', 'C')
        if Chem.MolFromSmiles(dec):
            print(f"✅ {dec}")
            valid_mols.append(dec)
        else:
            print(f"❌ {dec}")

    if valid_mols:
        # Save unique valid molecules
        unique_mols = list(set(valid_mols))
        pd.DataFrame({'SMILES': unique_mols}).to_csv('gen_simple.csv', index=False)
        print(f"\nSaved: gen_simple.csv ({len(unique_mols)} unique molecules)")
    else:
        print("\nNo valid molecules generated.")


if __name__ == "__main__":
    generate()