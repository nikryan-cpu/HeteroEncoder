"""
Единственное место, где задаются пути ML-пайплайна.

Все модули из src/ читают пути отсюда. Пути привязаны к расположению этого
файла, а не к текущей директории, поэтому пайплайн запускается из любого места:
раньше `python main.py` работал только из корня проекта.
"""

import os
from dataclasses import dataclass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- химия: единственное определение, раньше дублировалось в четырёх модулях ---
SCAFFOLD = "O=C(N)c1ccnc2ccccc12"  # хинолин-4-карбоксамид

# --- архитектура HeteroEncoderCVAE (src/model.py) ---
MODEL_EMBEDDING_DIM = 128
MODEL_HIDDEN_DIM = 256
MODEL_LATENT_DIM = 64


@dataclass
class RewardWeights:
    """
    Веса награды RL (src/rl_train.py: get_reward_diversity).

    Раньше были литералами в теле функции — вынесены сюда, чтобы GUI мог
    их менять без правки кода. Значения по умолчанию воспроизводят прежнее
    поведение один в один.
    """
    invalid: float = -5.0
    no_scaffold: float = 0.5
    scaffold_known: float = 2.0
    scaffold_novel: float = 10.0
    # Штраф за липофильность/массу — раньше отсутствовал вовсе. По умолчанию
    # выключен (lipophilicity_penalty=0), чтобы не менять поведение по молчанию.
    # См. README про reward hacking: LogP медиана 8.12 при пороге Lipinski 5.
    logp_threshold: float = 5.0
    mw_threshold: float = 500.0
    lipophilicity_penalty: float = 0.0

# --- входные данные ---
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DATASET = os.path.join(DATA_DIR, "dataset.csv")

# --- артефакты обучения: веса, словарь, нормировка, логи ---
# Переопределяется через переменную окружения — так GUI (gui/runner.py)
# нацеливает подпроцесс на конкретную модель из models/<имя>/, не трогая
# остальной код: src/*.py читают config.PROCESSED_DATA и т.п. при импорте,
# а этот путь должен быть выставлен ДО их импорта.
PRETRAINED_DIR = os.environ.get("HETEROENCODER_MODEL_DIR") or os.path.join(PROJECT_DIR, "pre-trained")
PROCESSED_DATA = os.path.join(PRETRAINED_DIR, "processed_data.pkl")
VOCAB = os.path.join(PRETRAINED_DIR, "vocab.pkl")
SCALERS = os.path.join(PRETRAINED_DIR, "scaler_params.npy")
CHECKPOINT = os.path.join(PRETRAINED_DIR, "checkpoint_last.pth")
MODEL_BEST = os.path.join(PRETRAINED_DIR, "model_best.pth")
MODEL_LAST = os.path.join(PRETRAINED_DIR, "model_last.pth")
MODEL_RL_BEST = os.path.join(PRETRAINED_DIR, "model_rl_best.pth")
MODEL_RL_LAST = os.path.join(PRETRAINED_DIR, "model_rl_last.pth")
TRAINING_LOG = os.path.join(PRETRAINED_DIR, "training_log.csv")
TRAINING_LOG_DETAILED = os.path.join(PRETRAINED_DIR, "training_log_detailed.csv")
RL_LOG = os.path.join(PRETRAINED_DIR, "rl_log.csv")

# --- библиотека моделей (GUI): новые модели создаются здесь, "default"
#     остаётся сам pre-trained/ и никогда не переименовывается ---
MODELS_ROOT = os.path.join(PROJECT_DIR, "models")

# --- результаты генерации ---
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
NOVEL_MOLECULES = os.path.join(OUTPUTS_DIR, "novel_molecules.csv")

# --- графики ---
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

for _d in (DATA_DIR, PRETRAINED_DIR, OUTPUTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)
