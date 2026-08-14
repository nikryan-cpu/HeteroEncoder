"""
Синхронизирует chemplus с кластером: пересобирает chemplus_local из
chemplus-main/src и сразу заливает его на сайт из settings.SITE_NAME.

Заменяет пару build_override.py + upload_override.py — раньше их надо было
звать по очереди, и пропуск второго молча оставлял на кластере старый код.

    python tools/sync_cluster.py            # собрать и залить
    python tools/sync_cluster.py --build    # только собрать локально
"""

import argparse
import os
import shutil
import sys

sys.stdout.reconfigure(errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settings

sys.path.insert(0, os.path.join(settings.DOCKING_DIR, "chemplus-main", "src"))
from chemplus import unicore

# Модули, которые нужны докингу на кластере. Остальное из chemplus-main
# (binana, md, swiss_adme, unicore) там не используется и не заливается.
MODULES = [
    "__init__.py",
    "sdf.py", "mdl.py", "draw.py", "gen3d.py", "mol_df.py", "prep.py",
    "vina/__init__.py",
    "vina/docking.py",
    "vina/pdbqt.py",
    "vina/dockscore.py",
    "vina/pdbqt_to_sdf.py",
    "vina/docking_serial.py",
    "vina/docking_mpi.py",
    "vina/rot_bond_helper.py",
    "vina/sf/nnscore2.0/NNScore2.py",
    "tasks/__init__.py",
    "tasks/docking.py",
]


def build():
    """Копирует нужные модули из chemplus-main в chemplus_local."""
    dst_root = os.path.join(settings.CHEMPLUS_LOCAL, "chemplus")
    print(f"[*] Сборка {os.path.relpath(settings.CHEMPLUS_LOCAL, settings.DOCKING_DIR)} ...")
    for f in MODULES:
        src = os.path.join(settings.CHEMPLUS_SRC, f)
        dst = os.path.join(dst_root, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    root_init = os.path.join(dst_root, "__init__.py")
    if not os.path.exists(root_init):
        open(root_init, "w").close()
    print(f"[+] Собрано модулей: {len(MODULES)}")


def collect_files():
    """Файлы chemplus_local для заливки, без кэша Python."""
    files = []
    for root, dirs, fnames in os.walk(settings.CHEMPLUS_LOCAL):
        # __pycache__ на кластере не нужен и может затенить свежий исходник
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for n in sorted(fnames):
            if n.endswith(".pyc"):
                continue
            p = os.path.join(root, n)
            files.append((p, os.path.relpath(p, settings.CHEMPLUS_LOCAL).replace("\\", "/")))
    return files


def upload():
    """Заливает chemplus_local на кластер по адресу из settings."""
    files = collect_files()
    print(f"[*] Заливка на {settings.SITE_NAME}: файлов {len(files)}")
    for path, rel in files:
        server_dir = os.path.dirname(settings.SERVER_CHEMPLUS_URL + "/" + rel)
        unicore.put_file(settings.UCC_PATH, path, server_dir)
        print(f"    {rel}")
    print(f"[+] Готово: {settings.SERVER_CHEMPLUS_URL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пересборка и заливка chemplus на кластер")
    parser.add_argument("--build", action="store_true",
                        help="только пересобрать chemplus_local, не заливать")
    args = parser.parse_args()

    build()
    if args.build:
        print("[i] Заливка пропущена (--build). На кластере остался прежний код.")
    else:
        upload()
