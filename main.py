import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

# Импорты ваших модулей
from data_preprocessing import preprocess_data, vectorize_smiles
from utilities import get_charset, get_char_to_int, preprocess_embeddings, preprocess_energy
from model import build_hetero_encoder

# --- КОНФИГУРАЦИЯ ---
DATASET_PATH = 'results_100k.csv'
ENERGY_COL_NAME = 'Energy'
MAX_LEN = 75
BATCH_SIZE = 128
EPOCHS = 50  # Ставим побольше, так как теперь мы можем дообучать

# Имена файлов для сохранения прогресса
BEST_MODEL_PATH = 'best_model.h5'  # Лучшая по val_loss
LAST_MODEL_PATH = 'last_model.h5'  # Последняя сохраненная (для resume)
LOG_FILE = 'training_log.csv'  # Файл с историей loss


def main():
    # 1. Загрузка данных (как обычно)
    print(f"Загрузка данных...")
    if not os.path.exists(DATASET_PATH):
        print(f"Ошибка: {DATASET_PATH} не найден. Убедитесь, что диск примонтирован.")
        return

    raw_data = pd.read_csv(DATASET_PATH)
    if ENERGY_COL_NAME not in raw_data.columns:
        raw_data.rename(columns={raw_data.columns[1]: ENERGY_COL_NAME}, inplace=True)

    data = preprocess_data(raw_data)

    # 2. Подготовка словаря
    full_text_series = pd.concat([data['SMILES'], data['CANONICAL_SMILES']])
    temp_vocab_df = pd.DataFrame({'SMILES': full_text_series})
    charset = get_charset(temp_vocab_df)
    char_to_int = get_char_to_int(charset)
    vocab_size = len(charset)

    # Сохраняем словарь, если его еще нет (или перезаписываем, не страшно)
    with open('vocab.pkl', 'wb') as f:
        pickle.dump({'charset': charset, 'char_to_int': char_to_int}, f)

    # 3. Разделение и Векторизация
    train_df, test_df = train_test_split(data, test_size=0.1, random_state=42)

    print("Векторизация...")
    X_s_train, Y_s_train = vectorize_smiles(train_df['SMILES'].tolist(), charset, char_to_int, MAX_LEN)
    X_s_test, Y_s_test = vectorize_smiles(test_df['SMILES'].tolist(), charset, char_to_int, MAX_LEN)

    X_c_train, Y_c_train = vectorize_smiles(train_df['CANONICAL_SMILES'].tolist(), charset, char_to_int, MAX_LEN)
    X_c_test, Y_c_test = vectorize_smiles(test_df['CANONICAL_SMILES'].tolist(), charset, char_to_int, MAX_LEN)

    feat_cols = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds', 'RingCount', 'QED']
    X_feat_train, X_feat_test, sc_feat = preprocess_embeddings(train_df[feat_cols], test_df[feat_cols])
    X_en_train, X_en_test, sc_en = preprocess_energy(train_df[ENERGY_COL_NAME], test_df[ENERGY_COL_NAME])

    with open('scalers.pkl', 'wb') as f:
        pickle.dump({'sc_feat': sc_feat, 'sc_energy': sc_energy}, f)

    # --- ЛОГИКА RESUME (Возобновление обучения) ---

    initial_epoch = 0
    model = None

    # Проверяем, есть ли сохраненная "последняя" модель
    if os.path.exists(LAST_MODEL_PATH):
        print(f"\nНАЙДЕН ЧЕКПОИНТ: {LAST_MODEL_PATH}")
        print("Загружаем модель и продолжаем обучение...")

        try:
            model = load_model(LAST_MODEL_PATH)

            # Пытаемся понять, на какой эпохе остановились, читая CSV лог
            if os.path.exists(LOG_FILE):
                log_df = pd.read_csv(LOG_FILE)
                if not log_df.empty:
                    # Последняя эпоха в файле (в файле нумерация с 0)
                    last_epoch_in_file = log_df['epoch'].iloc[-1]
                    initial_epoch = last_epoch_in_file + 1
                    print(f"Возобновляем с эпохи: {initial_epoch}")
            else:
                print("Файл логов не найден, но модель есть. Начнем следующую эпоху условно.")

        except Exception as e:
            print(f"Ошибка загрузки чекпоинта: {e}")
            print("Будет создана новая модель.")
            model = None

    # Если модель не загрузилась (первый запуск), строим с нуля
    if model is None:
        print("\nСоздаем новую модель с нуля...")
        model = build_hetero_encoder(
            vocab_size=vocab_size,
            max_len=MAX_LEN,
            num_features=len(feat_cols),
            latent_dim=128
        )
        # Если начинаем сначала, удаляем старый лог, чтобы не смешивать данные
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)

    model.summary()

    # --- CALLBACKS ---

    # 1. Сохраняет лучшую модель (для использования в generate.py)
    cb_best = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # 2. Сохраняет модель КАЖДУЮ эпоху (для возобновления)
    cb_last = ModelCheckpoint(
        LAST_MODEL_PATH,
        save_best_only=False,  # Сохраняем всегда, даже если результат хуже (это текущее состояние)
        verbose=0
    )

    # 3. Пишет историю в CSV (append=True позволяет дописывать в конец файла)
    cb_logger = CSVLogger(LOG_FILE, append=True)

    print(f"\nСтарт обучения (Epoch {initial_epoch} -> {EPOCHS})...")

    # Если мы уже прошли все эпохи
    if initial_epoch >= EPOCHS:
        print("Обучение уже завершено (initial_epoch >= EPOCHS).")
        print("Если хотите учить дальше, увеличьте константу EPOCHS в main.py")
        return

    history = model.fit(
        x=[X_s_train, X_c_train, X_feat_train, X_en_train],
        y=[Y_s_train, Y_c_train],
        validation_data=(
            [X_smiles_test, X_canon_test, X_feat_test, X_energy_test],
            [Y_smiles_test, Y_canon_test]
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,  # <--- Ключевой параметр для возобновления
        callbacks=[cb_best, cb_last, cb_logger]
    )

    print("Сессия обучения завершена.")


if __name__ == "__main__":
    main()