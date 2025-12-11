import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from rdkit import Chem
from utilities import reverse_replace_atoms  # Вам нужна обратная замена Y->Cl


def generate(target_energy=-12.0, num_molecules=100):
    # 1. Загрузка ресурсов
    print("Loading model and resources...")
    model = load_model('best_model.h5')

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    char_to_int = vocab['char_to_int']
    int_to_char = {i: c for c, i in char_to_int.items()}
    vocab_size = len(char_to_int)

    with open('scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    sc_en = scalers['sc_en']

    # 2. Подготовка вектора Z
    # Мы генерируем из случайного шума (128) + целевая энергия (1)

    # Нормализуем целевую энергию
    target_en_norm = sc_en.transform([[target_energy]])[0][0]

    # Шум (Latent features)
    noise = np.random.normal(0, 1, size=(num_molecules, 128))

    # Объединяем шум и энергию
    energy_vec = np.full((num_molecules, 1), target_en_norm)
    z_vec = np.concatenate([noise, energy_vec], axis=1)  # Shape (N, 129)

    # 3. Получение начальных состояний для LSTM декодера
    # Достаем слои из модели по именам, которые мы дали в model.py

    # Для примера берем SMILES декодер (dec_s)
    h_layer = model.get_layer('dec_s_init_h')
    bn_h_layer = model.get_layer('dec_s_bn_h')
    c_layer = model.get_layer('dec_s_init_c')
    bn_c_layer = model.get_layer('dec_s_bn_c')

    # Рассчитываем начальные h и c
    init_h = bn_h_layer(h_layer(z_vec)).numpy()
    init_c = bn_c_layer(c_layer(z_vec)).numpy()

    lstm_layer = model.get_layer('lstm_s')
    dense_layer = model.get_layer('out_smiles')

    # 4. Посимвольная генерация
    generated_smiles = []
    print(f"Generating {num_molecules} molecules with Energy {target_energy}...")

    for i in range(num_molecules):
        # Текущие состояния для одной молекулы
        curr_h = init_h[i:i + 1]
        curr_c = init_c[i:i + 1]

        # Начальный символ '!'
        start_idx = char_to_int['!']
        curr_input = np.zeros((1, 1, vocab_size))
        curr_input[0, 0, start_idx] = 1.0

        mol_string = ""

        for _ in range(75):  # Max len
            # Шаг LSTM
            lstm_out, h_out, c_out = lstm_layer(curr_input, initial_state=[curr_h, curr_c])

            # Предсказание следующего символа
            probs = dense_layer(lstm_out)[0, 0]
            # Выбор символа (sampling)
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = int_to_char[next_idx]

            if next_char == 'E':  # Конец строки
                break

            mol_string += next_char

            # Обновляем вход и состояния для следующего шага
            curr_input = np.zeros((1, 1, vocab_size))
            curr_input[0, 0, next_idx] = 1.0
            curr_h = h_out
            curr_c = c_out

        generated_smiles.append(mol_string)

    # 5. Пост-обработка и проверка
    valid_mols = []
    for s in generated_smiles:
        # Обратная замена Y->Cl, X->Br
        real_s = reverse_replace_atoms(s)
        if Chem.MolFromSmiles(real_s):
            valid_mols.append(real_s)

    print(f"Valid molecules: {len(valid_mols)} / {num_molecules}")
    print(valid_mols[:5])

    # Сохраняем
    pd.DataFrame({'SMILES': valid_mols}).to_csv('generated_molecules.csv', index=False)


if __name__ == "__main__":
    generate()