import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def build_hetero_encoder(vocab_size, max_len, num_features=8, latent_dim=128):
    # --- 1. ENCODERS (Кодировщики) ---

    # A. SMILES Encoder (LSTM)
    input_smiles = Input(shape=(max_len, vocab_size), name='in_smiles')
    # LSTM возвращает последовательность и последнее состояние
    _, h_s, c_s = layers.LSTM(128, return_state=True)(input_smiles)
    # Пропускаем через Dense (как на схеме "Dense Encoder" после LSTM)
    emb_smiles = layers.Dense(128, activation='relu')(h_s)

    # B. Canonical SMILES Encoder (LSTM)
    input_canon = Input(shape=(max_len, vocab_size), name='in_canon')
    _, h_c, c_c = layers.LSTM(128, return_state=True)(input_canon)
    emb_canon = layers.Dense(128, activation='relu')(h_c)

    # C. Features Encoder (Dense)
    input_features = Input(shape=(num_features,), name='in_features')
    x_f = layers.Dense(64, activation='relu')(input_features)
    x_f = layers.Dense(32, activation='relu')(x_f)
    x_f = layers.BatchNormalization()(x_f)
    emb_features = layers.Dense(16, activation='relu')(x_f)

    # --- 2. LATENT SPACE (Латентное пространство) ---

    # Объединяем выходы всех энкодеров
    concat = layers.Concatenate()([emb_smiles, emb_canon, emb_features])
    concat = layers.BatchNormalization()(concat)

    # Сжимаем до 128 (Вектор свойств молекулы)
    latent_mol = layers.Dense(latent_dim, activation='relu')(concat)

    # Добавляем Энергию (Energy Binding)
    # Это позволяет управлять генерацией: мы подаем желаемую энергию
    input_energy = Input(shape=(1,), name='in_energy')

    # Итоговый вектор Z (128 + 1)
    z = layers.Concatenate()([latent_mol, input_energy])

    # --- 3. DECODERS (Декодировщики) ---

    # Нам нужно превратить вектор Z обратно в начальные состояния (h, c) для LSTM декодеров
    # Создаем слои с именами, чтобы потом найти их при генерации
    def make_init_layers(prefix):
        dense_h = layers.Dense(128, activation='relu', name=f'{prefix}_init_h')
        bn_h = layers.BatchNormalization(name=f'{prefix}_bn_h')
        dense_c = layers.Dense(128, activation='relu', name=f'{prefix}_init_c')
        bn_c = layers.BatchNormalization(name=f'{prefix}_bn_c')
        return dense_h, bn_h, dense_c, bn_c

    # Слои инициализации для SMILES декодера
    s_h_dense, s_h_bn, s_c_dense, s_c_bn = make_init_layers('dec_s')
    # Применяем их к Z
    state_h_s = s_h_bn(s_h_dense(z))
    state_c_s = s_c_bn(s_c_dense(z))

    # Слои инициализации для Canonical декодера
    c_h_dense, c_h_bn, c_c_dense, c_c_bn = make_init_layers('dec_c')
    state_h_c = c_h_bn(c_h_dense(z))
    state_c_c = c_c_bn(c_c_dense(z))

    # Сами декодеры (Teacher Forcing: вход = правильная строка)

    # Decoder 1: SMILES
    lstm_dec_s = layers.LSTM(128, return_sequences=True, name='lstm_s')
    # Подаем input_smiles как "подсказку" (сдвинутую на 1 символ при обучении)
    x_dec_s = lstm_dec_s(input_smiles, initial_state=[state_h_s, state_c_s])
    output_smiles = layers.Dense(vocab_size, activation='softmax', name='out_smiles')(x_dec_s)

    # Decoder 2: Canonical
    lstm_dec_c = layers.LSTM(128, return_sequences=True, name='lstm_c')
    x_dec_c = lstm_dec_c(input_canon, initial_state=[state_h_c, state_c_c])
    output_canon = layers.Dense(vocab_size, activation='softmax', name='out_canon')(x_dec_c)

    # Сборка модели
    model = Model(
        inputs=[input_smiles, input_canon, input_features, input_energy],
        outputs=[output_smiles, output_canon]
    )

    # Компиляция
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model