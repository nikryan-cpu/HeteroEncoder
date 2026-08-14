# -*- coding: utf-8 -*-
import time
import sys
import os
import glob
import re
import shutil
from datetime import datetime

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

# ============================================================
# ПРОВЕРКА UCC ПЕРЕД ЗАПУСКОМ
# ============================================================
def test_ucc():
    """Проверяет, что UCC работает и не просит пароль."""
    import subprocess
    print("[*] Проверка UCC...")
    try:
        result = subprocess.run(
            [UCC_PATH, "list-sites", "--all"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"[-] UCC вернул ошибку (код {result.returncode}):")
            print(result.stderr)
            if "Password" in result.stderr or "keystore" in result.stderr.lower():
                print("\n[!] ПРОБЛЕМА С KEYSTORE/PASSWORD:")
                print("   UNICORE использует SSL-ключи для аутентификации.")
                print("   Обычно админы кластера выдают файл keystore (например, user.jks)")
                print("   и пароль к нему. Надо настроить один из вариантов:")
                print("   1. Переменная окружения: set UCC_KEYSTORE_PASSWORD=твой_пароль")
                print("   2. Файл конфига: C:/UCC/ucc-1.3.1/conf/ucc.properties")
                print("      keystore=C:/Users/ТЫ/.ucc/keystore.jks")
                print("      keystore.password=твой_пароль")
                print("   3. Флаг при запуске: ucc.bat -k keystore.jks -p пароль list-sites --all")
            return False
        print("[+] UCC работает, сайты доступны")
        return True
    except FileNotFoundError:
        print(f"[-] Файл не найден: {UCC_PATH}")
        print("   Проверь путь к ucc.bat")
        return False
    except subprocess.TimeoutExpired:
        print("[-] UCC завис (timeout 30 сек)")
        return False
    except Exception as e:
        print(f"[-] Ошибка при проверке UCC: {e}")
        return False

# ============================================================
# ПРОВЕРКА MGLTools НА КЛАСТЕРЕ
# ============================================================
def fix_mgltools():
    """
    Находит pythonsh (MGLTools) на кластере и создаёт симлинк
    ~/bio/mgltools/bin/pythonsh в домашней папке пользователя.

    chemplus.vina.docking ищет MGLTools именно по пути
    os.path.expanduser('~') + "/bio/mgltools/bin/pythonsh",
    поэтому без этого симлинка пайплайн падает с ошибкой
    "MGLTools python not found".

    Код выполняется через python -c (без файлов и без bash),
    аргументы оборачиваются в одинарные кавычки как в unicore_dock —
    этот способ уже проверен на кластере.
    """
    from chemplus import unicore

    code = '''import os,glob,sys;h=os.path.expanduser("~");l=h+"/bio/mgltools/bin/pythonsh";c=["/share/bio/data/mgltools/bin/pythonsh","/share/bio/mgltools/bin/pythonsh","/share/bio/data/mgltools_x86_64Linux2_1.5.6/bin/pythonsh","/share/bio/mgltools_x86_64Linux2_1.5.6/bin/pythonsh","/share/mgltools_x86_64Linux2_1.5.6/bin/pythonsh","/share/mgltools/bin/pythonsh"];f=[x for x in c if os.path.exists(x)];print("HOME="+h);print("SEARCH="+l+" -> "+str(glob.glob(l)));print("CANDIDATES_FOUND="+str(f));[(os.unlink(l) if os.path.lexists(l) else None,os.makedirs(os.path.dirname(l),exist_ok=True),os.symlink(f[0],l)) for x in [0] if f];print("MGLTOOLS_RESULT="+("OK" if f else "NOT_FOUND"));print("VERIFY="+str(glob.glob(l)));print("TREE="+str([(r,sorted(os.listdir(r))) for r in ["/share","/share/bio","/share/bio/data"] if os.path.isdir(r)]))'''

    print("[*] Проверка MGLTools на кластере...")
    try:
        # Вывод проверки — в отдельную папку: UCC называет файлы по UUID задачи,
        # и в общей results/ они копились бы с каждым запуском. Заодно glob ниже
        # гарантированно находит вывод именно этой задачи, а не чужой.
        if os.path.isdir(PROBE_DIR):
            shutil.rmtree(PROBE_DIR)
        os.makedirs(PROBE_DIR, exist_ok=True)

        u_file = os.path.join(PROBE_DIR, "fix_mgltools.u")
        unicore.create_job_file(u_file, SERVER_EXECUTABLE,
                                arguments_list=["-c", "'" + code + "'"],
                                memory=16, nodes=1, cpus_per_node=8, run_time=1)
        unicore.run_command([UCC_PATH, "run", "-s", SITE_NAME, "-o", PROBE_DIR, u_file])
        time.sleep(3)

        stdout_files = sorted(glob.glob(os.path.join(PROBE_DIR, "*.stdout")), key=os.path.getmtime)
        stderr_files = sorted(glob.glob(os.path.join(PROBE_DIR, "*.stderr")), key=os.path.getmtime)
        out = ""
        if stdout_files:
            with open(stdout_files[-1], encoding="utf-8", errors="replace") as f:
                out = f.read()
        if "MGLTOOLS_RESULT" not in out and stderr_files:
            with open(stderr_files[-1], encoding="utf-8", errors="replace") as f:
                err = f.read()
            if err.strip():
                out += "\n--- stderr ---\n" + err
        for line in out.strip().splitlines():
            print("   | " + line)
        if "MGLTOOLS_RESULT=OK" in out:
            print("[+] MGLTools найден, симлинк создан")
            return True
        print("[-] MGLTools НЕ найден — пришли мне вывод выше")
        return False
    except Exception as e:
        print(f"[-] Ошибка при запуске проверки MGLTools: {e}")
        return False

# ============================================================
# ПАПКИ ЗАПУСКОВ
# ============================================================
# UCC называет вывод по UUID задачи, который выдаёт сервер, поэтому файлы
# нельзя заставить перезаписываться. Вместо этого каждый запуск получает
# свою папку results/<дата>_<время>/ — и вывод, и результаты лежат вместе.
RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def new_run_dir():
    """Создаёт results/<ГГГГ-ММ-ДД_ЧЧ-ММ-СС>/ под текущий запуск."""
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(LOCAL_WORK_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def list_run_dirs():
    """Папки запусков, от свежих к старым."""
    if not os.path.isdir(LOCAL_WORK_DIR):
        return []
    dirs = [os.path.join(LOCAL_WORK_DIR, n) for n in os.listdir(LOCAL_WORK_DIR)
            if RUN_DIR_RE.match(n) and os.path.isdir(os.path.join(LOCAL_WORK_DIR, n))]
    return sorted(dirs, reverse=True)


def prune_old_runs(keep=KEEP_RUNS):
    """Удаляет папки запусков старше последних `keep`."""
    old = list_run_dirs()[keep:]
    for path in old:
        shutil.rmtree(path, ignore_errors=True)
    if old:
        print(f"[*] Удалено старых запусков: {len(old)} (оставлено последних {keep})")


# ============================================================
# ДИАГНОСТИКА ПАДЕНИЙ
# ============================================================
def print_last_stderr(run_dir, tail_lines=40):
    """Печатает хвост свежего .stderr — там настоящая причина падения задачи."""
    files = sorted(glob.glob(os.path.join(run_dir, "*.stderr")), key=os.path.getmtime)
    if not files:
        print("[!] .stderr не найден — задача могла не дойти до запуска")
        return
    with open(files[-1], encoding="utf-8", errors="replace") as f:
        lines = f.read().strip().splitlines()
    if not lines:
        print(f"[!] {os.path.basename(files[-1])} пуст")
        return
    print(f"\n--- {os.path.basename(files[-1])} (последние {tail_lines} строк) ---")
    for line in lines[-tail_lines:]:
        print("   | " + line)
    print("-" * 60)

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
    prune_old_runs()  # до создания новой папки, чтобы она точно уцелела
    run_dir = new_run_dir()
    print(f"   Результаты: {run_dir}")
    print("-" * 60)

    # 1. Проверка UCC
    if not test_ucc():
        print("\n[!] Исправь проблему с UCC перед запуском!")
        sys.exit(1)

    # 1.5. Проверка/починка MGLTools (pythonsh) на кластере
    if not fix_mgltools():
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