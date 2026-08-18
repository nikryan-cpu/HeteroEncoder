# -*- coding: utf-8 -*-
"""
Докинг одновременно на нескольких сайтах (PARALLEL_SITES в settings.py) —
LOCAL_SDF делится между сайтами пропорционально их cpu_count, каждый сайт
получает свою задачу с указанным числом CPU.

SKIF_GRID_CIS и SKIF_GEO — разные машины с разными файловыми системами,
поэтому задачи полностью независимы: свой work_dir на каждом сервере,
свой job-файл, свои dockscore.csv/*_docked.sdf.gz, скачиваются в свою
подпапку results/<дата_время>/<SITE_NAME>/ — без этого разделения оба
сайта писали бы результат в одинаково названный dockscore.csv локально
(имя на сервере одно и то же для любого сайта) и затирали бы друг друга.

    python MyDocking_parallel.py
"""
import time
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
chemplus_path = os.path.join(script_dir, "chemplus-main", "src")
if os.path.exists(chemplus_path):
    sys.path.insert(0, chemplus_path)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

from chemplus.unicore import docking
from chemplus.unicore import get_job_status

from settings import (UCC_PATH, SERVER_EXECUTABLE, SERVER_WORK_DIR,
                      SERVER_PYTHONPATH, LOCAL_WORK_DIR, PROBE_DIR,
                      KEEP_RUNS, LOCAL_RECEPTOR, LOCAL_PDB_RECEPTOR, LOCAL_CONFIG,
                      LOCAL_SDF, PARALLEL_SITES, resources_for)

from docking_common import test_ucc, fix_mgltools, new_run_dir, prune_old_runs, print_last_stderr
from tools.split_sdf import split_sdf_by_weights


def merge_dockscores(jobs, run_dir):
    """Склеивает dockscore.csv успешно завершившихся сайтов в один файл."""
    import pandas as pd
    frames = []
    for job in jobs:
        path = os.path.join(job["local_dir"], "dockscore.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["site"] = job["site_name"]
            frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "E_vina, kcal/mol" in combined.columns:
        combined = combined.sort_values(by="E_vina, kcal/mol", ascending=True)
    out_path = os.path.join(run_dir, "dockscore_combined.csv")
    combined.to_csv(out_path, index=False)
    return out_path


def run_parallel_docking():
    print("=" * 60)
    print(">>> ПАРАЛЛЕЛЬНЫЙ ДОКИНГ НА НЕСКОЛЬКИХ САЙТАХ")
    print("=" * 60)
    for s in PARALLEL_SITES:
        nodes, per_node = resources_for(s["cpu_count"], s["site_name"])
        print(f"   {s['site_name']}: {s['cpu_count']} CPU ({nodes} узла(ов) * {per_node})")
    print(f"   Рецептор: {LOCAL_RECEPTOR}")
    print(f"   Лиганды:  {LOCAL_SDF}")
    print(f"   Конфиг:   {LOCAL_CONFIG}")

    os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
    prune_old_runs(LOCAL_WORK_DIR, KEEP_RUNS)
    run_dir = new_run_dir(LOCAL_WORK_DIR)
    print(f"   Результаты: {run_dir}")
    print("-" * 60)

    if not test_ucc(UCC_PATH):
        print("\n[!] Исправь проблему с UCC перед запуском!")
        sys.exit(1)

    for s in PARALLEL_SITES:
        probe_dir = PROBE_DIR + "_" + s["site_name"]
        if not fix_mgltools(UCC_PATH, s["site_name"], probe_dir, SERVER_EXECUTABLE):
            print(f"\n[!] Не удалось найти MGLTools на {s['site_name']}!")
            sys.exit(1)

    # Делим лиганды пропорционально cpu_count каждого сайта.
    print("\n[+] Деление SDF между сайтами...")
    sdf_base = os.path.splitext(os.path.basename(LOCAL_SDF))[0]
    split_paths = [os.path.join(run_dir, f"{sdf_base}_{s['site_name']}.sdf") for s in PARALLEL_SITES]
    weights = [s["cpu_count"] for s in PARALLEL_SITES]
    counts = split_sdf_by_weights(LOCAL_SDF, split_paths, weights)
    for s, path, count in zip(PARALLEL_SITES, split_paths, counts):
        print(f"   {s['site_name']}: {count} молекул -> {os.path.basename(path)}")

    # Отправка задач — каждая в свою подпапку run_dir/<SITE_NAME>/, чтобы
    # локально скачанные dockscore.csv/*_docked.sdf.gz не перезаписывали
    # друг друга (на сервере имя dockscore.csv не зависит от sdf_name).
    print("\n[+] Отправка задач...")
    jobs = []
    start = time.time()
    for s, split_path in zip(PARALLEL_SITES, split_paths):
        local_dir = os.path.join(run_dir, s["site_name"])
        os.makedirs(local_dir, exist_ok=True)
        try:
            job_id_file = docking.unicore_dock(
                ucc_path=UCC_PATH,
                site_name=s["site_name"],
                server_executable=SERVER_EXECUTABLE,
                server_work_dir=SERVER_WORK_DIR,
                cpu_count=s["cpu_count"],
                local_work_dir=local_dir,
                local_receptor=LOCAL_RECEPTOR,
                local_pdb_receptor=LOCAL_PDB_RECEPTOR,
                local_config=LOCAL_CONFIG,
                local_sdf=split_path,
                sync=False,
                rewrite=False,
                pythonpath=SERVER_PYTHONPATH,
            )
            print(f"[+] {s['site_name']}: задача принята, ID {os.path.basename(job_id_file)}")
            jobs.append({"site_name": s["site_name"], "job_id_file": job_id_file,
                        "local_dir": local_dir, "status": "running", "failed": False})
        except Exception as e:
            print(f"[-] {s['site_name']}: ошибка при отправке: {e}")
            jobs.append({"site_name": s["site_name"], "job_id_file": None,
                        "local_dir": local_dir, "status": "submit_failed", "failed": True})

    if all(j["status"] == "submit_failed" for j in jobs):
        print("\n[!] Ни одна задача не отправилась.")
        sys.exit(1)

    # Мониторинг всех задач сразу.
    print("\n[*] Ожидание завершения всех сайтов (проверка каждые 30 сек)...")
    print("   (Можно нажать Ctrl+C — задачи продолжат выполняться на кластерах)")
    print("-" * 60)

    try:
        while True:
            pending = [j for j in jobs if j["status"] == "running"]
            if not pending:
                break
            for j in pending:
                try:
                    status = get_job_status(UCC_PATH, j["job_id_file"])
                    if "exit code: 0" in status:
                        j["status"] = "done"
                    elif "exit code:" in status:
                        j["status"] = "error"
                        j["failed"] = True
                except Exception as e:
                    status = f"ошибка проверки: {e}"
                j["last_status"] = status.strip() if isinstance(status, str) else str(status)

            elapsed = time.time() - start
            line = " | ".join(f"{j['site_name']}: {j['status']}" for j in jobs)
            print(f"\r[{elapsed:.0f}s] {line}", end="")

            if all(j["status"] != "running" for j in jobs):
                break
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\n[||] Прервано пользователем. Задачи продолжают выполняться на кластерах.")
        for j in jobs:
            if j["job_id_file"]:
                print(f"    {j['site_name']}: python check_results.py \"{j['job_id_file']}\"")
        sys.exit(0)

    print()
    print("-" * 60)

    # Скачивание результатов по каждому сайту независимо.
    for j in jobs:
        if j["status"] == "submit_failed":
            continue
        print(f"\n[+] {j['site_name']}: скачивание результатов...")
        try:
            docking.get_docking_results(UCC_PATH, j["job_id_file"], j["local_dir"])
            print(f"    Сохранено в: {j['local_dir']}")
        except Exception as e:
            print(f"    Ошибка при скачивании: {e}")
        if j["failed"]:
            print_last_stderr(j["local_dir"])

    merged = merge_dockscores(jobs, run_dir)
    if merged:
        print(f"\n[+] Объединённый dockscore: {merged}")
    else:
        print("\n[!] Ни один dockscore.csv не скачался — объединять нечего.")

    print(f"\n[DONE] Общее время: {time.time() - start:.0f} сек")
    print("=" * 60)


if __name__ == "__main__":
    run_parallel_docking()
