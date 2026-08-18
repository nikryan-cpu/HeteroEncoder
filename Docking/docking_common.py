# -*- coding: utf-8 -*-
"""
Общие шаги запуска докинга, используемые и MyDocking.py (один сайт), и
MyDocking_parallel.py (несколько сайтов одновременно). Вынесено сюда,
чтобы правки (например, в поиске MGLTools) не приходилось дублировать
в обоих скриптах.
"""

import glob
import os
import re
import shutil
import subprocess
import time
from datetime import datetime


def test_ucc(ucc_path):
    """Проверяет, что UCC работает и не просит пароль."""
    print("[*] Проверка UCC...")
    try:
        result = subprocess.run(
            [ucc_path, "list-sites", "--all"],
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
        print(f"[-] Файл не найден: {ucc_path}")
        print("   Проверь путь к ucc.bat")
        return False
    except subprocess.TimeoutExpired:
        print("[-] UCC завис (timeout 30 сек)")
        return False
    except Exception as e:
        print(f"[-] Ошибка при проверке UCC: {e}")
        return False


def fix_mgltools(ucc_path, site_name, probe_dir, server_executable):
    """
    Находит pythonsh (MGLTools) на кластере `site_name` и создаёт симлинк
    ~/bio/mgltools/bin/pythonsh в домашней папке пользователя.

    chemplus.vina.docking ищет MGLTools именно по пути
    os.path.expanduser('~') + "/bio/mgltools/bin/pythonsh",
    поэтому без этого симлинка пайплайн падает с ошибкой
    "MGLTools python not found". Симлинк локальный для файловой системы
    сайта, поэтому при параллельном докинге на нескольких сайтах вызывать
    отдельно для каждого (probe_dir у каждого сайта должен быть свой —
    иначе параллельные проверки затрут вывод друг друга).

    Код выполняется через python -c (без файлов и без bash),
    аргументы оборачиваются в одинарные кавычки как в unicore_dock —
    этот способ уже проверен на кластере.
    """
    from chemplus import unicore

    code = '''import os,glob,sys;h=os.path.expanduser("~");l=h+"/bio/mgltools/bin/pythonsh";c=["/share/bio/data/mgltools/bin/pythonsh","/share/bio/mgltools/bin/pythonsh","/share/bio/data/mgltools_x86_64Linux2_1.5.6/bin/pythonsh","/share/bio/mgltools_x86_64Linux2_1.5.6/bin/pythonsh","/share/mgltools_x86_64Linux2_1.5.6/bin/pythonsh","/share/mgltools/bin/pythonsh"];f=[x for x in c if os.path.exists(x)];print("HOME="+h);print("SEARCH="+l+" -> "+str(glob.glob(l)));print("CANDIDATES_FOUND="+str(f));[(os.unlink(l) if os.path.lexists(l) else None,os.makedirs(os.path.dirname(l),exist_ok=True),os.symlink(f[0],l)) for x in [0] if f];print("MGLTOOLS_RESULT="+("OK" if f else "NOT_FOUND"));print("VERIFY="+str(glob.glob(l)));print("TREE="+str([(r,sorted(os.listdir(r))) for r in ["/share","/share/bio","/share/bio/data"] if os.path.isdir(r)]))'''

    print(f"[*] Проверка MGLTools на кластере ({site_name})...")
    try:
        # Вывод проверки — в отдельную папку: UCC называет файлы по UUID задачи,
        # и в общей results/ они копились бы с каждым запуском. Заодно glob ниже
        # гарантированно находит вывод именно этой задачи, а не чужой.
        if os.path.isdir(probe_dir):
            shutil.rmtree(probe_dir)
        os.makedirs(probe_dir, exist_ok=True)

        u_file = os.path.join(probe_dir, "fix_mgltools.u")
        unicore.create_job_file(u_file, server_executable,
                                arguments_list=["-c", "'" + code + "'"],
                                memory=16, nodes=1, cpus_per_node=8, run_time=1)
        unicore.run_command([ucc_path, "run", "-s", site_name, "-o", probe_dir, u_file])
        time.sleep(3)

        stdout_files = sorted(glob.glob(os.path.join(probe_dir, "*.stdout")), key=os.path.getmtime)
        stderr_files = sorted(glob.glob(os.path.join(probe_dir, "*.stderr")), key=os.path.getmtime)
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
            print(f"[+] MGLTools найден на {site_name}, симлинк создан")
            return True
        print(f"[-] MGLTools НЕ найден на {site_name} — пришли мне вывод выше")
        return False
    except Exception as e:
        print(f"[-] Ошибка при запуске проверки MGLTools на {site_name}: {e}")
        return False


# UCC называет вывод по UUID задачи, который выдаёт сервер, поэтому файлы
# нельзя заставить перезаписываться. Вместо этого каждый запуск получает
# свою папку results/<дата>_<время>/ — и вывод, и результаты лежат вместе.
RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def new_run_dir(local_work_dir):
    """Создаёт results/<ГГГГ-ММ-ДД_ЧЧ-ММ-СС>/ под текущий запуск."""
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(local_work_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def list_run_dirs(local_work_dir):
    """Папки запусков, от свежих к старым."""
    if not os.path.isdir(local_work_dir):
        return []
    dirs = [os.path.join(local_work_dir, n) for n in os.listdir(local_work_dir)
            if RUN_DIR_RE.match(n) and os.path.isdir(os.path.join(local_work_dir, n))]
    return sorted(dirs, reverse=True)


def prune_old_runs(local_work_dir, keep):
    """Удаляет папки запусков старше последних `keep`."""
    old = list_run_dirs(local_work_dir)[keep:]
    for path in old:
        shutil.rmtree(path, ignore_errors=True)
    if old:
        print(f"[*] Удалено старых запусков: {len(old)} (оставлено последних {keep})")


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
