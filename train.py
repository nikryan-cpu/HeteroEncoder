import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Импортируем ваши функции
from data_preprocessing import preprocess_data, vectorize_smiles
from utilities import get_charset, get_char_to_int, preprocess_embeddings, preprocess_energy
from model import build_hetero_encoder

DATASET_PATH = 'results_100k.csv'  # Ваш файл
ENERGY_COL = 'Energy'  # Проверьте название колонки в CSV!
MAX_LEN = 75  # Размерность вектора (из PDF 35-75)


def train():
    # 1. Загрузка и очистка
    print("Loading data...")
    raw_data = pd.read_csv(DATASET_PATH)

    # Если колонки 'Energy' нет, переименуйте вторую колонку
    if ENERGY_COL not in raw_data.columns:
        print(f"Warning: Column '{ENERGY_COL}' not found. Using the second column as Energy.")
        raw_data.rename(columns={raw_data.columns[1]: ENERGY_COL}, inplace=True)

    data = preprocess_data(raw_data)
    print(f"Data ready: {data.shape[0]} samples")

    # 2. Словарь символов
    # Объединяем текст из обоих колонок, чтобы найти все возможные символы
    full_text = pd.concat([data['SMILES'], data['CANONICAL_SMILES']])
    # Создаем фиктивный DF для вашей функции get_charset, если она требует DF
    charset = get_charset(pd.DataFrame({'SMILES': full_text}))
    char_to_int = get_char_to_int(charset)
    vocab_size = len(charset)
    print(f"Vocab size: {vocab_size}")

    # Сохраняем словарь (нужен для генерации)
    with open('vocab.pkl', 'wb') as f:
        pickle.dump({'charset': charset, 'char_to_int': char_to_int}, f)

    # 3. Split Train/Test
    train_df, test_df = train_test_split(data, test_size=0.1, random_state=42)

    # 4. Векторизация (One-Hot)
    print("Vectorizing...")
    # Обычные SMILES (Вход X, Выход Y)
    X_s_train, Y_s_train = vectorize_smiles(train_df['SMILES'].tolist(), charset, char_to_int, MAX_LEN)
    X_s_test, Y_s_test = vectorize_smiles(test_df['SMILES'].tolist(), charset, char_to_int, MAX_LEN)

    # Canonical SMILES
    X_c_train, Y_c_train = vectorize_smiles(train_df['CANONICAL_SMILES'].tolist(), charset, char_to_int, MAX_LEN)
    X_c_test, Y_c_test = vectorize_smiles(test_df['CANONICAL_SMILES'].tolist(), charset, char_to_int, MAX_LEN)

    # 5. Нормализация Дескрипторов и Энергии
    feat_cols = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds', 'RingCount', 'QED']
    X_feat_train, X_feat_test, sc_feat = preprocess_embeddings(train_df[feat_cols], test_df[feat_cols])

    # Энергию тоже нужно нормализовать (StandardScaler)
    X_en_train, X_en_test, sc_en = preprocess_energy(train_df[ENERGY_COL], test_df[ENERGY_COL])

    # Сохраняем скейлеры (нужны для генерации)
    with open('scalers.pkl', 'wb') as f:
        pickle.dump({'sc_feat': sc_feat, 'sc_en': sc_en}, f)

    # 6. Создание модели
    model = build_hetero_encoder(vocab_size, MAX_LEN, num_features=len(feat_cols))
    model.summary()

    # 7. Обучение
    print("Starting training...")
    # Вход: [SMILES, Canonical, Features, Energy]
    # Выход: [SMILES_Next_Char, Canonical_Next_Char]

    model.fit(
        x=[X_s_train, X_c_train, X_feat_train, X_en_train],
        y=[Y_s_train, Y_c_train],
        validation_data=(
            [X_s_test, X_c_test, X_feat_test, X_en_test],
            [Y_s_test, Y_c_test]
        ),
        batch_size=128,
        epochs=30,  # Можно увеличить до 50-100
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
    )
    print("Training complete!")


if __name__ == "__main__":
    train()