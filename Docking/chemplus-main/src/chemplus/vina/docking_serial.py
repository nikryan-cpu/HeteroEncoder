"""
Модуль для последовательного молекулярного докинга.

Выполняет докинг молекул одну за другой (последовательно) с использованием
ВСЕХ доступных CPU ядер для каждой молекулы. Подходит для:
- Одного-двух молекул с высокой точностью
- Когда нет доступа к MPI
- Локального использования

Функции:
- dock_lig(): Выполнить докинг одной молекулы
- run_docking(): Выполнить докинг всех молекул в директории

Workflow:
1. Проверить наличие всех необходимых файлов
2. Для каждой молекулы PDBQT в директории:
   - Запустить AutoDock Vina с полным количеством CPU
   - Записать результат в лог
3. Сохранить результаты в выходную директорию

Параметры логирования:
- Каждый результат записывается с timestamp
- Для успешных: время выполнения и random seed
- Для ошибок: код ошибки и сообщение об ошибке
- Пропуск уже додокированных (если rewrite=False)

Пример использования:
    from chemplus.vina import docking_serial

    docking_serial.run_docking(
        vina="/path/to/vina.exe",
        receptor_file="receptor.pdbqt",
        config_file="config.txt",
        ligands_dir="ligands_pdbqt/",
        out_dir="docked_results/",
        log_file="docking.log",
        cpu_count=4,
        rewrite=True
    )
"""

import time
import glob
import subprocess
import os
from datetime import datetime

def dock_lig(lig_name, cmd_args):
    """
    Выполнить докинг одной молекулы используя AutoDock Vina.

    Запускает процесс AutoDock Vina для дока одного лиганда и
    извлекает информацию о random seed из вывода.

    Параметры:
        lig_name (str): Название лиганда (для логирования)
        cmd_args (list): Аргументы для subprocess.run() включая:
                        ["vina", "--receptor", "rec.pdbqt",
                         "--config", "conf.txt", "--ligand", "lig.pdbqt",
                         "--out", "out.pdbqt", "--cpu", "4"]

    Возвращает:
        str: Строка с результатом в формате:
            "SUCCESS - ligand_name, spent time - X.Xs, SEED - Y" (успех)
            "FAILED - ligand_name, ERROR - code message" (ошибка)

    Примечание: Может занять долгое время в зависимости от размера лиганда
    """
    start_time = time.time()
    result = subprocess.run(cmd_args, capture_output=True, shell=False)
    spent_time = time.time() - start_time

    if result.returncode:
        return "FAILED - {0}, ERROR - {1} {2}".format(lig_name, result.returncode, result.stderr.decode("utf-8").strip())
    else:
        stdout = result.stdout.decode("utf-8")
        seed_begin_index = stdout.index("random seed: ") + 13
        seed_end_index = seed_begin_index + stdout[seed_begin_index:].index(")")
        seed = stdout[seed_begin_index:seed_end_index]
        return "SUCCESS - {0}, spent time - {1}s, SEED - {2}".format(lig_name, round(spent_time, 1), seed)

def run_docking(vina, receptor_file, config_file, ligands_dir, out_dir, log_file, cpu_count, rewrite=True):
    """
    Запустить последовательный докинг всех молекул в директории.

    Для каждой PDBQT молекулы в папке:
    1. Проверить, документирована ли уже (если rewrite=False)
    2. Запустить AutoDock Vina с заданным количеством CPU
    3. Сохранить результат в выходную папку
    4. Записать результат в лог-файл

    Проверяемые файлы и условия перед запуском:
    - Существует ли vina (AutoDock Vina исполняемый файл)
    - Существует ли receptor_file (PDBQT файл рецептора)
    - Существует ли config_file (конфиг Vina)
    - Существует ли ligands_dir (директория с молекулами)

    Параметры:
        vina (str): Путь к AutoDock Vina исполняемому файлу
                   Windows: "C:/Program Files/Vina/vina.exe"
                   Linux: "/usr/bin/vina"

        receptor_file (str): Путь к PDBQT файлу рецептора
                            Пример: "receptor.pdbqt"

        config_file (str): Путь к конфиг-файлу AutoDock Vina
                          Содержит: receptor, center_x/y/z, size_x/y/z
                          Пример: "config.txt"

        ligands_dir (str): Директория с PDBQT молекулами
                          Должны быть в формате: molecule_name.pdbqt
                          Пример: "ligands_pdbqt/"

        out_dir (str): Директория для сохранения результатов
                      Будет создана если не существует
                      Результаты: molecule_name_docked.pdbqt
                      Пример: "docked_results/"

        log_file (str): Путь к файлу лога
                       Содержит дату/время, пути файлов, статус каждой молекулы
                       Пример: "docking.log"

        cpu_count (int): Максимальное количество CPU ядер для использования
                        Фактически используется min(cpu_count, os.cpu_count())
                        Пример: 4 (использует 4 ядра для каждой молекулы)

        rewrite (bool): Перезаписывать ли результаты если они уже существуют
                       True - перезаписать (по умолчанию)
                       False - пропустить уже обработанные молекулы

    Логирование:
        - Начало докинга с timestamp
        - Использованные параметры
        - Для каждой молекулы:
          * "SUCCESS - name, spent time - Xs, SEED - seed_value"
          * "ALREADY DONE - name" (если rewrite=False)
          * "FAILED - name, ERROR - code message"
        - Окончание докинга

    Вызывает исключения если:
        - Файл Vina не найден
        - Файл рецептора не найден
        - Конфиг-файл не найден
        - Директория с молекулами не найдена

    Пример:
        run_docking(
            vina="vina.exe",
            receptor_file="target.pdbqt",
            config_file="docking.conf",
            ligands_dir="molecules/",
            out_dir="results/",
            log_file="docking_log.txt",
            cpu_count=4,
            rewrite=False  # Не перезаписывать
        )
    """
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log = open(log_file, 'a')
    log.write("\n---------------------------------------------DOCKING------------------------------------------------\n")
    log.write("Data: {0}\nVina: {1}\nReceptor: {2}\nConfig file: {3}\nLigands dir: {4}\nOut dir: {5}\nRewrite: {6}\n\n"
                   .format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), vina, receptor_file, config_file, 
                           ligands_dir, out_dir, rewrite))
    log.close()

    if not os.path.exists(vina):
        log = open(log_file, 'a')
        log.write('Vina not found\n')
        log.close()
        raise Exception("Vina not found")

    if not os.path.exists(receptor_file):
        log = open(log_file, 'a')
        log.write('Receptor not found\n')
        log.close()
        raise Exception("Receptor not found")

    if not os.path.exists(config_file):
        log = open(log_file, 'a')
        log.write('Config file not found\n')
        log.close()
        raise Exception("Config file not found")

    if not os.path.exists(ligands_dir):
        log = open(log_file, 'a')
        log.write('Ligands dir not found\n')
        log.close()
        raise Exception("Ligands dir not found")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cpu_count = min(os.cpu_count(), cpu_count)
    for lig_path in glob.iglob(ligands_dir + '/*.pdbqt'):
        lig_name = os.path.basename(lig_path).split('.')[0]
        docked_lig_path = out_dir + '/' + lig_name + '.pdbqt'
        cmd_args = [vina, "--receptor", receptor_file, "--config", config_file, 
                    "--ligand", lig_path, "--out", docked_lig_path, "--cpu", str(cpu_count)]
        if os.path.isfile(docked_lig_path) and not rewrite:
            result_message = "ALREADY DONE - {0}".format(lig_name)
        else:
            result_message = dock_lig(lig_name, cmd_args)
        
        log = open(log_file, 'a')
        log.write(result_message + '\n')
        log.close()

    log = open(log_file, 'a')
    log.write('\n--------------------------------------------FINISHED------------------------------------------------\n')
    log.close()
    print("Docking finished successfully!")
