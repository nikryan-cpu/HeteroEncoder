"""
Основной модуль для молекулярного докинга с AutoDock Vina.

Предоставляет высокоуровневый интерфейс для выполнения полного цикла докинга:
1. Конвертация молекул SDF → PDB → PDBQT
2. Проверка PDBQT файлов (ротируемые связи, количество атомов)
3. Запуск докинга (serial или MPI)
4. Извлечение лучших поз (top model)
5. Конвертация результатов PDBQT → SDF
6. Подсчет энергий связывания (Vina, RF-score, NNScore)

Основные функции:
- sdf_dock_solved_path(): Полный цикл докинга с автоматическими параметрами
- sdf_dock(): Полный цикл с явными параметрами
- run_docking(): Запуск MPI докинга (низкоуровневый)

Автоматическое нахождение зависимостей:
- AutoDock Vina (vina.exe / vina)
- RF-score модель машинного обучения
- NNScore2.py скрипт
- MGLTools Python интерпретатор для преобразования PDB→PDBQT

Требуемые внешние инструменты:
- AutoDock Vina 1.1.2+
- MGLTools
- Python 3.9+
- RDKit (для работы с молекулами)
- joblib (для параллелизации)

Пример использования (автоматический поиск инструментов):
    from chemplus.vina import docking

    docking.sdf_dock_solved_path(
        work_dir="my_docking",
        receptor_file="target.pdbqt",
        config_file="vina_config.txt",
        sdf_file="compounds.sdf",
        cpu_count=8,
        serial=False,
        rewrite=True
    )

Результаты в work_dir:
    - compounds_pdb/: Конвертированные в PDB молекулы
    - compounds_pdbqt/: PDBQT молекулы с проверенными связями
    - compounds_docked/: Результаты докинга (PDBQT)
    - compounds_top_model/: Лучшие позы (PDBQT)
    - compounds_sdf_format/: Лучшие позы в SDF
    - compounds_pdb_format/: Лучшие позы в PDB
    - compounds_docked.sdf.gz: Сжатый SDF со всеми результатами
    - dockscore.csv: Таблица энергий связывания
"""

import os
import sys
import glob
import subprocess
from chemplus import sdf
from chemplus.vina import pdbqt
from chemplus.vina import dockscore
from chemplus.vina import pdbqt_to_sdf
from chemplus.vina import docking_serial

def run_docking(vina, receptor_file, config_file, ligands_dir, out_dir, log_file, cpu_count, rewrite=False):
    """
    Запустить параллельный МPI докинг нескольких молекул.

    Это низкоуровневая функция, обычно используется внутри sdf_dock().
    Для пользователей рекомендуется использовать sdf_dock_solved_path().

    Создает MPI процессы и запускает chemplus.vina.docking_mpi модуль.

    Параметры:
        vina (str): Полный путь к AutoDock Vina исполняемому файлу
        receptor_file (str): Путь к PDBQT файлу рецептора
        config_file (str): Путь к конфиг-файлу Vina
        ligands_dir (str): Директория с PDBQT молекулами
        out_dir (str): Директория для результатов
        log_file (str): Файл лога
        cpu_count (int): Количество MPI процессов (примерно = количество ядер)
        rewrite (bool): Перезаписывать результаты

    Использует conda окружение "chem" и mpiexec для запуска MPI программы.

    Вызывает исключение если MPI докинг завершился с ошибкой.
    """

    cmd_args = ["/share/bio/data/miniconda3/bin/conda", "run", "-n", "chem",
                "mpiexec", "-np", str(cpu_count), sys.executable, "-m", "chemplus.vina.docking_mpi",
                "--vina", vina, "--receptor_file", receptor_file, "--config_file", config_file, 
                "--ligands_dir", ligands_dir, "--out_dir", out_dir, "--log_file", log_file]
    if rewrite:
        cmd_args += ["--rewrite"]
    
    result = subprocess.run(cmd_args, capture_output=True, shell=False)
    
    if result.returncode:
        raise Exception(result.stderr.decode("utf-8").strip())


def sdf_dock(work_dir, vina, nn2_script, rf4, rf4_model, mgl_python, receptor_file, config_file, sdf_file, cpu_count, serial=False, rewrite=True):
    """
    Выполнить полный цикл молекулярного докинга с явными параметрами.

    ВНИМАНИЕ: Используйте sdf_dock_solved_path() вместо этой функции!
    Эта функция требует явного указания всех путей к инструментам.

    Полный workflow:
    1. SDF → PDB конвертация
    2. PDB → PDBQT конвертация (добавление атомных типов)
    3. Проверка PDBQT файлов (ротируемые связи, фильтрация по количеству связей)
    4. Запуск докинга (serial или MPI)
    5. Извлечение лучшей позы из результатов
    6. PDBQT → SDF конвертация результатов
    7. Подсчет энергий связывания (Vina, RF-score, NNScore2)

    Параметры:
        work_dir (str): Рабочая директория для сохранения всех промежуточных файлов
                       Будет создана если не существует
                       Пример: "docking_results"

        vina (str): Полный путь к AutoDock Vina исполняемому файлу
                   Пример: "C:/Program Files/Vina/vina.exe"

        nn2_script (str): Путь к NNScore2.py скрипту
                         Пример: "chemplus/vina/sf/nnscore2.0/NNScore2.py"

        rf4 (str): Путь к RF-score исполняемому файлу
                  Пример: "chemplus/vina/sf/rfscore4/rf-score.exe"

        rf4_model (str): Путь к обученной модели RF-score
                        Пример: "chemplus/vina/sf/rfscore4/pdbbind-2014-refined.rf"

        mgl_python (str): Путь к Python интерпретатору из MGLTools
                         Используется для PDB→PDBQT конвертации
                         Пример: "C:/MGLTools/python.exe"

        receptor_file (str): Путь к PDBQT файлу рецептора
                            Пример: "receptor.pdbqt"

        config_file (str): Путь к конфиг-файлу AutoDock Vina
                          Должен содержать center_x/y/z и size_x/y/z
                          Пример: "vina_config.txt"

        sdf_file (str): Путь к SDF файлу с молекулами для докинга
                       Пример: "compounds.sdf"

        cpu_count (int): Максимальное количество CPU ядер для использования
                        Пример: 8

        serial (bool): Режим докинга
                      True - последовательный (одна молекула за раз, но с полным cpu_count)
                      False - параллельный MPI (несколько молекул одновременно)
                      По умолчанию False

        rewrite (bool): Перезаписывать ли результаты если они уже существуют
                       По умолчанию True

    Производит:
        - work_dir/sdf_name_pdb/: PDB файлы молекул
        - work_dir/sdf_name_pdbqt/: PDBQT файлы молекул
        - work_dir/sdf_name_docked/: Результаты докинга (PDBQT)
        - work_dir/sdf_name_top_model/: Лучшие позы (PDBQT)
        - work_dir/sdf_name_sdf_format/: Лучшие позы (SDF)
        - work_dir/sdf_name_pdb_format/: Лучшие позы (PDB)
        - work_dir/sdf_name_docked.sdf.gz: Результаты (сжатый SDF)
        - work_dir/sdf_name_log.txt: Лог файл
        - work_dir/dockscore.csv: Таблица энергий

    Вызывает исключение если:
        - SDF файл не найден
        - Инструменты не найдены
        - Конвертация не удалась
        - Докинг завершился с ошибкой
    """
    if not os.path.exists(sdf_file):
        raise Exception("SDF file doesn't exists: " + sdf_file)
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    sdf_name = os.path.basename(sdf_file).split(".")[0]

    print("Converting SDF to PDB files:")
    pdb_dir = work_dir + "/" + sdf_name + "_pdb"
    mol_num, converted_mol_num, bad_mol_list = sdf.sdf_to_pdb(sdf_file, pdb_dir, overwrite=rewrite)
    print("SDF contains %s molecules.\nConverted %s molecules." % (mol_num, converted_mol_num))
    if len(bad_mol_list):
        print("There are the following %s bad molecules:\n%s" % (len(bad_mol_list), "\n".join(bad_mol_list)))
    
    print("--------------------------------")
    print("Converting PDB's to PDBQT files:")
    pdbqt_dir = work_dir + "/" + sdf_name + "_pdbqt"
    pdb_num, converted_pdb_num, bad_pdb_list = pdbqt.pdb_dir_to_pdbqt(pdb_dir, pdbqt_dir, mgl_python, cpu_count=cpu_count, overwrite=rewrite)
    print("PDB directory contains %s molecules.\nConverted %s PDB's." % (pdb_num, converted_pdb_num))
    if len(bad_pdb_list):
          print("There are the following %s bad pdb's:\n%s" % (len(bad_pdb_list), "\n".join(bad_pdb_list)))
    
    print("--------------------------------")
    print("Checking PDBQT's rotatable bonds:")
    illegal_rb_pdbqt_list, missed_rb_pdbqt_list = pdbqt.check_pdbqt_dir_rot_bonds(pdbqt_dir, pdb_dir, fix_bad_pdbqt=True)
    if not len(illegal_rb_pdbqt_list) and not len(missed_rb_pdbqt_list):
        print("OK")
    if len(illegal_rb_pdbqt_list):
        print("Fixed %s PDBQT's with illegal rotatable bonds:\n%s" % (len(illegal_rb_pdbqt_list), "\n".join(illegal_rb_pdbqt_list)))
    if len(missed_rb_pdbqt_list):
        print("Found %s PDBQT's with missed rotatable bonds:\n%s" % (len(missed_rb_pdbqt_list), "\n".join(missed_rb_pdbqt_list)))

    print("--------------------------------")
    print("Searching for PDBQT's with more than 20 rotatable bonds:")
    overnrb_pdbqt_list = pdbqt.filter_pdbqt(pdbqt_dir, delete_overnrb_pdbqt=True)
    if len(overnrb_pdbqt_list):
        print("Deleted %s PDBQT's with more than 20 rotatable bonds:\n%s" 
              % (len(overnrb_pdbqt_list), "\n".join(overnrb_pdbqt_list)))
    else:
        print("All is well")

    print("--------------------------------")
    print("Docking...")
    out_dir = work_dir + "/" + sdf_name + "_docked"
    log_file = work_dir + "/" + sdf_name + "_log.txt"
    if not serial:
        run_docking(vina=vina, receptor_file=receptor_file, config_file=config_file, ligands_dir=pdbqt_dir, out_dir=out_dir, 
                    log_file=log_file, cpu_count=cpu_count, rewrite=rewrite)
    else:
        docking_serial.run_docking(vina=vina, receptor_file=receptor_file, config_file=config_file, ligands_dir=pdbqt_dir, 
                                   out_dir=out_dir, log_file=log_file, cpu_count=cpu_count, rewrite=rewrite)
    print("Docking finished.")

    print("--------------------------------")
    print("Extracting top model from docked PDBQT's:")
    top_model_dir = work_dir + "/" + sdf_name + "_top_model"
    top_model_failures = pdbqt.get_top_model(out_dir, top_model_dir)
    if len(top_model_failures):
        print("Occurred %s failures when extracting top model from PDBQTs:\n%s" 
              % (len(top_model_failures), "\n".join(top_model_failures)))

    print("--------------------------------")
    print("Writing top model PDBQT's to SDF:")
    sdf_format_dir = work_dir + "/" + sdf_name + "_sdf_format"
    if not os.path.exists(sdf_format_dir):
        os.makedirs(sdf_format_dir)
    pdb_format_dir = work_dir + "/" + sdf_name + "_pdb_format"
    if not os.path.exists(pdb_format_dir):
        os.makedirs(pdb_format_dir)
    sdf_docked_file = work_dir + "/" + sdf_name + "_docked.sdf.gz"
    pdbqt_to_sdf.pdbqt_to_sdf(top_model_dir, pdb_dir, sdf_docked_file, sdf_format_dir, pdb_format_dir)
  
    print("--------------------------------")
    print("Getting SF results for top model PDBQT's:")
    dockscore.run_dockscore(nn2_script=nn2_script, rf4=rf4, rf4_model=rf4_model, receptor_file=receptor_file, work_dir=work_dir, ligs_dir=top_model_dir, cpu_count=cpu_count, rewrite=True)

sf_dir = os.path.dirname(os.path.realpath(__file__)) + "/sf"

def sdf_dock_solved_path(work_dir, receptor_file, config_file, sdf_file, cpu_count, serial=False, rewrite=True):
    """
    ⭐ ГЛАВНАЯ ФУНКЦИЯ ДЛЯ ЛОКАЛЬНОГО ДОКИНГА ⭐

    Выполнить полный цикл молекулярного докинга с АВТОМАТИЧЕСКИМ поиском
    всех необходимых инструментов. Это рекомендуемый способ для большинства пользователей.

    Автоматически находит:
    - AutoDock Vina (vina.exe на Windows, vina на Linux)
    - RF-score программу и модель
    - NNScore2.py скрипт
    - MGLTools Python интерпретатор

    Полный workflow:
    1. SDF → PDB (конвертация молекул)
    2. PDB → PDBQT (добавление атомных типов)
    3. Проверка PDBQT (ротируемые связи, фильтрация)
    4. Докинг (serial или MPI запуск)
    5. Извлечение лучших поз
    6. PDBQT → SDF (конвертация результатов)
    7. Подсчет энергий (Vina score, RF-score, NNScore2)

    Параметры:
        work_dir (str): Рабочая директория для всех файлов
                       Пример: "docking_results" или "C:/docking"

        receptor_file (str): Путь к PDBQT файлу рецептора
                            Должна уже быть в PDBQT формате (готова к докингу)
                            Пример: "target_receptor.pdbqt"

        config_file (str): Путь к конфиг-файлу AutoDock Vina
                          Обязательно содержит:
                              receptor = /path/to/receptor.pdbqt
                              center_x = 10.5
                              center_y = 20.3
                              center_z = 15.8
                              size_x = 20
                              size_y = 20
                              size_z = 20
                          Пример: "vina.conf" или "config.txt"

        sdf_file (str): Путь к SDF файлу с молекулами для докинга
                       Может содержать одну или множество молекул
                       Пример: "drug_candidates.sdf"

        cpu_count (int): Максимальное количество CPU ядер
                        Фактическое использование: min(cpu_count, os.cpu_count())
                        Пример: 4, 8, 16

        serial (bool): Режим выполнения
                      True - последовательный (одна молекула за раз)
                            - медленнее, но использует все CPU для одной молекулы
                            - для 1-2 молекул с высокой точностью
                      False - параллельный MPI (несколько молекул одновременно)
                             - быстрее для множества молекул
                             - рекомендуется (по умолчанию)

        rewrite (bool): Перезаписывать результаты
                       True - всегда пересчитывать (по умолчанию)
                       False - пропустить уже обработанные

    Результаты (в work_dir):

        Директории:
        - sdfname_pdb/: Исходные молекулы в PDB формате
        - sdfname_pdbqt/: Молекулы в PDBQT формате с проверенными связями
        - sdfname_docked/: Результаты докинга из AutoDock Vina
        - sdfname_top_model/: Лучшие позы (PDBQT)
        - sdfname_sdf_format/: Результаты в SDF формате
        - sdfname_pdb_format/: Результаты в PDB формате

        Файлы:
        - sdfname_docked.sdf.gz: Все результаты в одном сжатом файле
        - sdfname_log.txt: Лог выполнения докинга
        - dockscore.csv: Таблица энергий связывания (главный результат!)

    ГЛАВНЫЙ РЕЗУЛЬТАТ - dockscore.csv содержит:
        Name: Название молекулы
        E_vina, kcal/mol: Энергия связывания по Vina
        pKd_vina: Предсказанный Kd по Vina
        (плюс RF-score и NNScore если доступны)

    Пример использования:
        from chemplus.vina import docking

        # Простой случай
        docking.sdf_dock_solved_path(
            work_dir="my_docking",
            receptor_file="protein.pdbqt",
            config_file="config.txt",
            sdf_file="molecules.sdf",
            cpu_count=4
        )

        # С явными параметрами
        docking.sdf_dock_solved_path(
            work_dir="C:/Projects/HTS/docking",
            receptor_file="target.pdbqt",
            config_file="vina_settings.conf",
            sdf_file="lib1000.sdf",
            cpu_count=8,
            serial=False,  # MPI для 1000+ молекул
            rewrite=False  # Не пересчитывать
        )

    Требуемые установки:
        - AutoDock Vina 1.1.2+
        - MGLTools (для PDB→PDBQT)
        - RDKit (для работы с молекулами)
        - joblib (для параллелизации)
        - mpi4py (если используется MPI)

    Windows: MGLTools обычно находится в
        C:/Program Files/MGLTools*/

    Linux: Обычно установлен через pip или package manager
        sudo apt-get install mgltools

    Вызывает исключение если:
        - Файлы не найдены
        - Инструменты не найдены
        - Конвертация молекул не удалась
        - Докинг завершился с ошибкой

    Время выполнения зависит от:
        - Количества молекул (линейное)
        - Размера молекул (экспоненциальное)
        - Количества CPU (обратное)
        Пример: 1000 молекул на 8 CPU = 1-2 часа
    """
    if sys.platform.startswith('linux'):
        vina = sf_dir + "/autodock_vina/vina"
        rf4 = sf_dir + '/rfscore4/rf-score'
        mgl_search_location = os.path.expanduser('~') + "/bio/mgltools/bin/pythonsh"
    elif sys.platform.startswith('win32'):
        vina = sf_dir + "/autodock_vina/vina.exe"
        rf4 = sf_dir + '/rfscore4/rf-score.exe'
        mgl_search_location = "C:/Program Files*/MGLTools*/python.exe"
    else:
        raise Exception("Unknown OS")
    
    try:
        mgl_python = glob.glob(mgl_search_location)[0]
    except:
        raise Exception("MGLTools python not found")
    
    rf4_model = sf_dir + '/rfscore4/pdbbind-2014-refined.rf'
    nn2_script = sf_dir + '/nnscore2.0/NNScore2.py'
    
    # print(mgl_python)
    # print(rf4)
    # print(rf4_model)
    # print(vina)
    # print(nn2_script)
    sdf_dock(work_dir, vina, nn2_script, rf4, rf4_model, mgl_python, receptor_file, config_file, sdf_file, cpu_count, serial, rewrite)