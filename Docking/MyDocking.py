# -*- coding: utf-8 -*-
import time
import sys
import os

# Добавляем в sys.path саму папку Docking (ради settings.py) и chemplus
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
chemplus_path = os.path.join(script_dir, "chemplus-main", "src")
if os.path.exists(chemplus_path):
    sys.path.insert(0, chemplus_path)

# Настройка кодировки для вывода во избежание UnicodeEncodeError на Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

from chemplus.unicore import docking
from chemplus.unicore import get_job_status

# Все настройки живут в settings.py — правь там, не здесь.
from settings import (UCC_PATH, SITE_NAME, SERVER_EXECUTABLE, SERVER_WORK_DIR,
                      SERVER_PYTHONPATH, CPU_COUNT, LOCAL_WORK_DIR, PROBE_DIR,
                      KEEP_RUNS, LOCAL_RECEPTOR, LOCAL_PDB_RECEPTOR, LOCAL_CONFIG,
                      LOCAL_SDF, resources_for)

# Шаги запуска (проверка UCC/MGLTools, папки запусков, диагностика падений)
# общие с MyDocking_parallel.py — вынесены в docking_common.py.
from docking_common import test_ucc, fix_mgltools, new_run_dir, prune_old_runs, print_last_stderr

# ============================================================
# ОСНОВНОЙ ЗАПУСК С ЛОГИРОВАНИЕМ
# ============================================================
def run_docking_with_progress():
    print("=" * 60)
    print(">>> ЗАПУСК МОЛЕКУЛЯРНОГО ДОКИНГА НА SKIF")
    print("=" * 60)
    _nodes, _per_node = resources_for()
    print(f"   Кластер: {SITE_NAME}")
    print(f"   CPU: {CPU_COUNT} ({_nodes} узла(ов) * {_per_node})")
    print(f"   Рецептор: {LOCAL_RECEPTOR}")
    print(f"   Лиганды:  {LOCAL_SDF}")
    print(f"   Конфиг:   {LOCAL_CONFIG}")

    # Каждый запуск пишет в свою папку results/<дата>_<время>/ — вывод UCC
    # и результаты докинга лежат вместе и не смешиваются между запусками.
    os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
    prune_old_runs(LOCAL_WORK_DIR, KEEP_RUNS)  # до создания новой папки, чтобы она точно уцелела
    run_dir = new_run_dir(LOCAL_WORK_DIR)
    print(f"   Результаты: {run_dir}")
    print("-" * 60)

    # 1. Проверка UCC
    if not test_ucc(UCC_PATH):
        print("\n[!] Исправь проблему с UCC перед запуском!")
        sys.exit(1)

    # 1.5. Проверка/починка MGLTools (pythonsh) на кластере
    if not fix_mgltools(UCC_PATH, SITE_NAME, PROBE_DIR, SERVER_EXECUTABLE):
        print("\n[!] Не удалось найти MGLTools на кластере!")
        sys.exit(1)

    # 2. Асинхронный запуск (чтобы мониторить прогресс)
    print("\n[+] Отправка задачи на кластер...")
    start = time.time()

    try:
        job_id_file = docking.unicore_dock(
            ucc_path=UCC_PATH,
            site_name=SITE_NAME,
            server_executable=SERVER_EXECUTABLE,
            server_work_dir=SERVER_WORK_DIR,
            cpu_count=CPU_COUNT,
            local_work_dir=run_dir,
            local_receptor=LOCAL_RECEPTOR,
            local_pdb_receptor=LOCAL_PDB_RECEPTOR,
            local_config=LOCAL_CONFIG,
            local_sdf=LOCAL_SDF,
            sync=False,  # ← АСИНХРОННЫЙ РЕЖИМ для мониторинга
            rewrite=False,  # переиспользуем уже сконвертированные PDB/PDBQT (докинг всё равно пересчитается: _docked пуст)
            pythonpath=SERVER_PYTHONPATH,  # своя копия chemplus перекрывает общую
        )
        print(f"[+] Задача принята в очередь")
        print(f"    ID задачи: {os.path.basename(job_id_file)}")
        print(f"    Папка запуска: {run_dir}")
    except Exception as e:
        print(f"[-] Ошибка при отправке: {e}")
        sys.exit(1)

    # 3. Мониторинг статуса
    print("\n[*] Ожидание завершения (проверка каждые 30 сек)...")
    print("   (Можно нажать Ctrl+C — задача продолжит выполняться на кластере)")
    print("-" * 60)
    
    spinner = ["/", "-", "\\", "|"]
    spin_idx = 0
    failed = False

    try:
        while True:
            try:
                status = get_job_status(UCC_PATH, job_id_file)
                elapsed = time.time() - start

                # Красивый вывод статуса
                if "exit code: 0" in status:
                    print(f"\r{spinner[spin_idx % len(spinner)]} [ OK ] ЗАВЕРШЁН УСПЕШНО! ({elapsed:.0f} сек)")
                    break
                elif "exit code:" in status:
                    print(f"\r{spinner[spin_idx % len(spinner)]} [ERR] ОШИБКА: {status}")
                    failed = True
                    break
                else:
                    # Задача в очереди или выполняется
                    print(f"\r{spinner[spin_idx % len(spinner)]} [RUN] {status.strip()} | {elapsed:.0f} сек", end="")
                    spin_idx += 1
                    
            except Exception as e:
                print(f"\r{spinner[spin_idx % len(spinner)]} [WAR] Ошибка проверки статуса: {e}", end="")
                spin_idx += 1
            
            time.sleep(30)
    except KeyboardInterrupt:
        print(f"\n\n[||] Прервано пользователем. Задача продолжает выполняться на кластере.")
        print(f"    ID задачи: {job_id_file}")
        print(f"    Чтобы проверить статус позже, запусти:")
        print(f"   python check_results.py \"{job_id_file}\"")
        sys.exit(0)

    # 4. Скачивание результатов
    print("\n[+] Скачивание результатов...")
    try:
        docking.get_docking_results(UCC_PATH, job_id_file, run_dir)
        print(f"[+] Результаты сохранены в: {run_dir}")
        print(f"   - dockscore.csv (энергии связывания)")
        print(f"   - *_docked.sdf.gz (позы лигандов)")
    except Exception as e:
        print(f"[-] Ошибка при скачивании: {e}")

    # При ненулевом exit code сам код ничего не объясняет — причина всегда
    # в stderr задачи, поэтому печатаем его хвост сразу.
    if failed:
        print_last_stderr(run_dir)

    print(f"\n[DONE] Общее время: {time.time() - start:.0f} сек")
    print("=" * 60)

if __name__ == "__main__":
    run_docking_with_progress()