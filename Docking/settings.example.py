"""
Шаблон settings.py — скопируй этот файл в settings.py и впиши свои значения.

settings.py в .gitignore (личный логин на кластере и локальный путь к UCC),
поэтому в репозитории лежит только этот шаблон с плейсхолдерами.

Единственное место, где задаются параметры докинга на кластере.

Все скрипты (MyDocking.py, check_results.py, tools/sync_cluster.py) читают
настройки отсюда, поэтому смена кластера — правка одной строки SITE_NAME,
а не поиск по файлам.

Пути привязаны к расположению этого файла, а не к текущей директории,
поэтому скрипты запускаются из любого места.
"""

import os

# ============================================================
# КЛАСТЕР
# ============================================================
# Путь к ucc.bat (клиент UNICORE), см. Docking/README.md — раздел установки UCC.
UCC_PATH = r"C:\Path\To\ucc-1.3.1-all\ucc-1.3.1\bin\ucc.bat"

# "SKIF_GRID_CIS" (40 CPU/узел) или "SKIF_GEO" (36 CPU/узел).
# Это РАЗНЫЕ машины с РАЗНЫМИ файловыми системами: после смены сайта
# нужно заново прогнать tools/sync_cluster.py — код там не появится сам.
SITE_NAME = "SKIF_GRID_CIS"

# Твой логин на кластере.
CLUSTER_USER = "USERNAME"
SERVER_EXECUTABLE = "/share/bio/data/miniconda3/envs/chem/bin/python"
SERVER_WORK_DIR = "~/projects/docking"
CPU_COUNT = 200

# Сколько CPU на узел даёт каждый сайт — отсюда считается число узлов.
CPUS_PER_NODE = {"SKIF_GRID_CIS": 40, "SKIF_GEO": 36}

# ============================================================
# ЛОКАЛЬНЫЕ ПУТИ
# ============================================================
DOCKING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DOCKING_DIR)
PROTEIN_DIR = os.path.join(DOCKING_DIR, "chemnotes-main", "1. Protein processing", "prepared_protein")

LOCAL_WORK_DIR = os.path.join(DOCKING_DIR, "results")

# Вывод служебной задачи-проверки MGLTools. Отдельно от результатов, потому что
# UCC называет файлы по UUID задачи и в общей папке они копятся при каждом
# запуске. Очищается перед каждой проверкой.
PROBE_DIR = os.path.join(LOCAL_WORK_DIR, "_probe")

# Сколько последних запусков докинга хранить в results/.
# Более старые тройки <uuid>.properties/.stdout/.stderr и их .u/.job удаляются
# при старте нового запуска. dockscore.csv, *_docked.sdf.gz и файлы белка
# не трогаются никогда.
KEEP_RUNS = 5
LOCAL_RECEPTOR = os.path.join(PROTEIN_DIR, "3WZD_cleaned.pdbqt")
LOCAL_PDB_RECEPTOR = os.path.join(PROTEIN_DIR, "3WZD_cleaned.pdb")
LOCAL_CONFIG = os.path.join(PROTEIN_DIR, "config.txt")
# Лиганды приезжают из ML-пайплайна: outputs/ — его папка результатов.
# Впиши имя своего файла, полученного на стадии "Подготовка соединений".
LOCAL_SDF = os.path.join(PROJECT_DIR, "outputs", "protonated_prepared_passed_3D_<дата>.sdf")

# Источник истины для правок и генерируемая из него копия для кластера.
CHEMPLUS_SRC = os.path.join(DOCKING_DIR, "chemplus-main", "src", "chemplus")
CHEMPLUS_LOCAL = os.path.join(DOCKING_DIR, "chemplus_local")

# ============================================================
# ПРОИЗВОДНЫЕ ЗНАЧЕНИЯ — считаются из SITE_NAME, руками не менять
# ============================================================
SERVER_HOME = "/home/" + CLUSTER_USER
SERVER_DOCK_DIR = SERVER_HOME + "/projects/docking"

# Куда tools/sync_cluster.py заливает chemplus_local.
SERVER_CHEMPLUS_URL = f"u6://{SITE_NAME}/Home/projects/docking/chemplus_local"

# PYTHONPATH задачи: перекрывает общий chemplus в /share/bio/data/chemplus.
# Абсолютный путь обязателен — "~" в переменной окружения не раскрывается.
SERVER_PYTHONPATH = SERVER_DOCK_DIR + "/chemplus_local"


def resources_for(cpu_count=CPU_COUNT, site_name=SITE_NAME):
    """Возвращает (nodes, cpus_per_node) — та же арифметика, что в unicore_dock."""
    per_node = CPUS_PER_NODE.get(site_name, 40)
    nodes = cpu_count // per_node + (cpu_count % per_node > 0)
    return nodes, cpu_count // nodes
