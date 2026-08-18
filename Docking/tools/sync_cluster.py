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
import time

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


def upload(site_name=None, file_retries=3, outer_passes=8, outer_wait=15):
    """
    Заливает chemplus_local на кластер. По умолчанию — на settings.SITE_NAME;
    --site позволяет залить на другой сайт (например, при подготовке
    параллельного докинга на нескольких сайтах сразу — MyDocking_parallel.py),
    не трогая settings.py.

    UNICORE-соединение временами рвётся посреди заливки (например, из-за VPN) —
    "Can't get protocols", "Connect timed out", разные java-исключения на
    случайном по счёту файле. Не системная блокировка, а именно нестабильная
    связь: та же операция почти всегда проходит при следующей попытке. Поэтому
    два уровня повтора: file_retries — сразу же для одного файла, и если этого
    не хватило — до outer_passes проходов по ВСЕМ ещё не залитым файлам с
    паузой между проходами (уже залитые не трогаются повторно, put_file и так
    перезаписывает, но это лишнее время на неустойчивом канале).
    """
    site_name = site_name or settings.SITE_NAME
    server_url = f"u6://{site_name}/Home/projects/docking/chemplus_local"
    files = collect_files()
    print(f"[*] Заливка на {site_name}: файлов {len(files)}")

    done = set()
    last_exc = None
    for outer_attempt in range(1, outer_passes + 1):
        remaining = [(p, rel) for p, rel in files if rel not in done]
        if not remaining:
            break
        for path, rel in remaining:
            server_dir = os.path.dirname(server_url + "/" + rel)
            for attempt in range(1, file_retries + 1):
                try:
                    unicore.put_file(settings.UCC_PATH, path, server_dir)
                    done.add(rel)
                    print(f"    {rel}")
                    break
                except Exception as e:
                    last_exc = e
                    if attempt < file_retries:
                        time.sleep(3)
        if len(done) < len(files) and outer_attempt < outer_passes:
            print(f"[!] Загружено {len(done)}/{len(files)}, соединение нестабильно — "
                 f"пауза {outer_wait}с, проход {outer_attempt + 1}/{outer_passes}...")
            time.sleep(outer_wait)

    if len(done) < len(files):
        raise Exception(f"Не удалось залить {len(files) - len(done)} из {len(files)} файлов "
                        f"после {outer_passes} проходов. Последняя ошибка: {last_exc}")
    print(f"[+] Готово: {server_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пересборка и заливка chemplus на кластер")
    parser.add_argument("--build", action="store_true",
                        help="только пересобрать chemplus_local, не заливать")
    parser.add_argument("--site", default=None,
                        help="залить на этот сайт вместо settings.SITE_NAME (например SKIF_GEO)")
    args = parser.parse_args()

    build()
    if args.build:
        print("[i] Заливка пропущена (--build). На кластере остался прежний код.")
    else:
        upload(args.site)
