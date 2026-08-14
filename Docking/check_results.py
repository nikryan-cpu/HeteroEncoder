#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для проверки статуса и скачивания результатов докинга
ПОСЛЕ ВЫКЛЮЧЕНИЯ КОМПЬЮТЕРА.

Использование:
    python check_results.py "путь/к/файлу_задачи.job"
    python check_results.py  # интерактивный выбор из results/*.job
"""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemplus.unicore import docking, get_job_status

# Настройки общие с MyDocking.py — правь в settings.py.
from settings import UCC_PATH, LOCAL_WORK_DIR

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')


def find_job_files():
    """Находит все .job файлы задач в папках запусков results/<дата>_<время>/."""
    files = glob.glob(os.path.join(LOCAL_WORK_DIR, "**", "*.job"), recursive=True)
    return sorted(files, key=os.path.getmtime, reverse=True)


def check_and_download(job_id_file):
    """Проверяет статус и скачивает результаты если готово."""
    # Результаты кладём рядом с .job — в папку того же запуска.
    run_dir = os.path.dirname(os.path.abspath(job_id_file))
    print(f"🔍 Проверка задачи: {os.path.basename(job_id_file)}")
    print(f"📂 Папка запуска: {run_dir}")
    print("-" * 60)

    try:
        status = get_job_status(UCC_PATH, job_id_file)
        print(f"📊 Статус: {status.strip()}")

        if "exit code: 0" in status:
            print("\n✅ Задача завершена успешно! Скачиваю результаты...")
            docking.get_docking_results(UCC_PATH, job_id_file, run_dir)
            print(f"\n📁 Результаты сохранены в: {run_dir}")
            print(f"   - dockscore.csv (энергии связывания)")
            print(f"   - *_docked.sdf.gz (позы лигандов)")
            return True

        elif "exit code:" in status:
            print(f"\n❌ Задача завершилась с ошибкой: {status}")
            # Всё равно пробуем скачать логи
            try:
                docking.get_docking_results(UCC_PATH, job_id_file, run_dir)
                print("📥 Логи ошибки скачаны (проверь stdout/stderr в папке запуска)")
            except:
                pass
            return False

        else:
            print("\n⏳ Задача ещё выполняется или в очереди.")
            print("   Запусти скрипт позже для проверки.")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при проверке: {e}")
        return False


def main():
    # Если передан аргумент — используем его
    if len(sys.argv) > 1:
        job_id_file = sys.argv[1]
        if not os.path.exists(job_id_file):
            print(f"❌ Файл не найден: {job_id_file}")
            sys.exit(1)
        check_and_download(job_id_file)
        return
    
    # Иначе — интерактивный выбор
    print("=" * 60)
    print("🔍 ПРОВЕРКА СТАТУСА ДОКИНГА НА SKIF")
    print("=" * 60)
    
    job_files = find_job_files()
    
    if not job_files:
        print(f"❌ В {LOCAL_WORK_DIR} не найдено .job файлов задач")
        print("   Убедись, что задача была отправлена через MyDocking.py")
        sys.exit(1)
    
    print(f"Найдено задач: {len(job_files)}")
    print("-" * 60)
    
    # Имя папки запуска читается лучше, чем UUID задачи.
    for i, f in enumerate(job_files, 1):
        run_name = os.path.basename(os.path.dirname(os.path.abspath(f)))
        print(f"  {i}. {run_name}   ({os.path.basename(f)})")
    
    print("-" * 60)
    try:
        choice = input("Выбери номер задачи (Enter = последняя): ").strip()
        if not choice:
            idx = 0
        else:
            idx = int(choice) - 1
        
        if 0 <= idx < len(job_files):
            check_and_download(job_files[idx])
        else:
            print("❌ Неверный номер")
    except (ValueError, KeyboardInterrupt):
        print("\n❌ Отмена")


if __name__ == "__main__":
    main()