import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import os

# Настройки стиля
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

FILES = {
    'log': 'training_log.csv',
    'real': 'PubChem_compound_smiles_substructure_C1=CC=C2C(=C1)C(=CC=N2)C(=O)N (1).csv',
    'gen': 'generated_molecules.csv'
}


def calc_props(smiles_list):
    """Считает простые свойства для списка SMILES"""
    mw = []
    logp = []
    qed = []
    valid_count = 0

    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol:
                mw.append(Descriptors.MolWt(mol))
                logp.append(Descriptors.MolLogP(mol))
                qed.append(QED.qed(mol))
                valid_count += 1
        except:
            pass
    return mw, logp, qed, valid_count


def plot_training_details():
    if not os.path.exists(FILES['log']):
        print("Нет логов обучения.")
        return

    data = pd.read_csv(FILES['log'])

    # Создаем фигуру с 2 графиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Общий Loss
    ax1.plot(data['epoch'], data['loss'], label='Train Total', linewidth=2)
    ax1.plot(data['epoch'], data['val_loss'], label='Val Total', linestyle='--')
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    # 2. Компоненты Loss (если Keras сохранил имена слоев)
    # Обычно имена выглядят как out_smiles_loss и out_canon_loss
    # Проверим, есть ли они в колонках
    cols = data.columns
    smiles_col = [c for c in cols if 'out_smiles_loss' in c and 'val' not in c]
    canon_col = [c for c in cols if 'out_canon_loss' in c and 'val' not in c]

    if smiles_col and canon_col:
        ax2.plot(data['epoch'], data[smiles_col[0]], label='SMILES Decoder')
        ax2.plot(data['epoch'], data[canon_col[0]], label='Canonical Decoder')
        ax2.set_title('Decoder Specific Losses')
        ax2.set_xlabel('Epoch')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Детальные лоссы не найдены в CSV', ha='center')

    plt.tight_layout()
    plt.savefig('graph_training_loss.png')
    print("Сохранен graph_training_loss.png")
    plt.show()


def plot_chemical_space():
    if not os.path.exists(FILES['real']) or not os.path.exists(FILES['gen']):
        print("Нет файлов с молекулами (real или generated).")
        return

    print("Расчет химических свойств (это может занять время)...")

    # Загружаем реальные (берем сэмпл 1000 шт для скорости)
    real_df = pd.read_csv(FILES['real'])
    if len(real_df) > 1000:
        real_df = real_df.sample(1000)
    real_mw, real_logp, real_qed, _ = calc_props(real_df['SMILES'])

    # Загружаем сгенерированные
    gen_df = pd.read_csv(FILES['gen'])
    gen_mw, gen_logp, gen_qed, val_count = calc_props(gen_df['SMILES'])

    print(f"Валидных сгенерированных молекул: {val_count} / {len(gen_df)}")

    # Строим 3 графика распределения
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # MW
    sns.kdeplot(real_mw, fill=True, label='Training Data', ax=axes[0], color='blue', alpha=0.3)
    sns.kdeplot(gen_mw, fill=True, label='Generated', ax=axes[0], color='red', alpha=0.3)
    axes[0].set_title('Molecular Weight Distribution')
    axes[0].legend()

    # LogP
    sns.kdeplot(real_logp, fill=True, label='Training Data', ax=axes[1], color='blue', alpha=0.3)
    sns.kdeplot(gen_logp, fill=True, label='Generated', ax=axes[1], color='red', alpha=0.3)
    axes[1].set_title('LogP (Lipophilicity) Distribution')

    # QED
    sns.kdeplot(real_qed, fill=True, label='Training Data', ax=axes[2], color='blue', alpha=0.3)
    sns.kdeplot(gen_qed, fill=True, label='Generated', ax=axes[2], color='red', alpha=0.3)
    axes[2].set_title('QED (Drug-likeness) Distribution')

    plt.tight_layout()
    plt.savefig('graph_chemical_properties.png')
    print("Сохранен graph_chemical_properties.png")
    plt.show()


def main():
    print("Построение графиков обучения...")
    plot_training_details()

    print("\nПостроение химических графиков...")
    plot_chemical_space()


if __name__ == "__main__":
    main()