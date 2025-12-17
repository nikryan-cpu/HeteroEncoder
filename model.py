import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def build_hetero_encoder(vocab_size, max_len, num_features=8, latent_dim=128):
    # --- 1. ENCODERS ---

    # A. SMILES Encoder (LSTM)
    # Takes the sequence, processes it, and outputs the final state vectors (h, c)
    input_smiles = Input(shape=(max_len, vocab_size), name='in_smiles')
    _, h_s, c_s = layers.LSTM(128, return_state=True)(input_smiles)
    emb_smiles = layers.Dense(128, activation='relu')(h_s)

    # B. Canonical SMILES Encoder (LSTM)
    # Separate encoder to learn the canonical structure
    input_canon = Input(shape=(max_len, vocab_size), name='in_canon')
    _, h_c, c_c = layers.LSTM(128, return_state=True)(input_canon)
    emb_canon = layers.Dense(128, activation='relu')(h_c)

    # C. Features Encoder (Dense MLP)
    # Processes numerical descriptors (MolWt, LogP, etc.)
    input_features = Input(shape=(num_features,), name='in_features')
    x_f = layers.Dense(64, activation='relu')(input_features)
    x_f = layers.Dense(32, activation='relu')(x_f)
    x_f = layers.BatchNormalization()(x_f)
    emb_features = layers.Dense(16, activation='relu')(x_f)

    # --- 2. LATENT SPACE ---

    # Concatenate all embeddings
    concat = layers.Concatenate()([emb_smiles, emb_canon, emb_features])
    concat = layers.BatchNormalization()(concat)

    # Compress to latent vector
    latent_mol = layers.Dense(latent_dim, activation=None)(concat)

    # Inject Binding Energy
    # This input allows conditioning generation on specific energy levels
    input_energy = Input(shape=(1,), name='in_energy')

    # The Core Z-Vector (Latent Space + Energy)
    # Named 'z_layer' so the Generator script can find it easily
    z = layers.Concatenate(name='z_layer')([latent_mol, input_energy])

    # --- 3. DECODERS ---

    # Helper to create initialization layers for LSTM states (h, c)
    # We need to map the Z-vector back to the LSTM state size (128)
    def make_init_layers(prefix):
        dense_h = layers.Dense(128, activation='relu', name=f'{prefix}_init_h')
        bn_h = layers.BatchNormalization(name=f'{prefix}_bn_h')
        dense_c = layers.Dense(128, activation='relu', name=f'{prefix}_init_c')
        bn_c = layers.BatchNormalization(name=f'{prefix}_bn_c')
        return dense_h, bn_h, dense_c, bn_c

    # Initialize states for SMILES Decoder
    s_h_dense, s_h_bn, s_c_dense, s_c_bn = make_init_layers('dec_s')
    state_h_s = s_h_bn(s_h_dense(z))
    state_c_s = s_c_bn(s_c_dense(z))

    # Initialize states for Canonical Decoder
    c_h_dense, c_h_bn, c_c_dense, c_c_bn = make_init_layers('dec_c')
    state_h_c = c_h_bn(c_h_dense(z))
    state_c_c = c_c_bn(c_c_dense(z))

    # --- DECODING (Teacher Forcing) ---
    # The decoder receives the correct previous character to predict the next one.

    # Decoder 1: SMILES
    lstm_dec_s = layers.LSTM(128, return_sequences=True, name='lstm_s')
    x_dec_s = lstm_dec_s(input_smiles, initial_state=[state_h_s, state_c_s])
    output_smiles = layers.Dense(vocab_size, activation='softmax', name='out_smiles')(x_dec_s)

    # Decoder 2: Canonical
    lstm_dec_c = layers.LSTM(128, return_sequences=True, name='lstm_c')
    x_dec_c = lstm_dec_c(input_canon, initial_state=[state_h_c, state_c_c])
    output_canon = layers.Dense(vocab_size, activation='softmax', name='out_canon')(x_dec_c)

    # --- MODEL ASSEMBLY ---
    model = Model(
        inputs=[input_smiles, input_canon, input_features, input_energy],
        outputs=[output_smiles, output_canon]
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model