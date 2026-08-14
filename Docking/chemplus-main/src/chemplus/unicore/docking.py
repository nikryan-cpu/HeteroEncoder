"""
Модуль для запуска молекулярного докинга на GRID кластере SKIF.

Предоставляет высокоуровневые функции для:
1. Запуска докинга на удаленном суперкомпьютере
2. Отслеживания статуса выполнения
3. Получения и обработки результатов

Основной workflow:
1. unicore_dock() - загрузить файлы, создать задачу, запустить на GRID
2. get_docking_results() - получить результаты по завершении (если sync=False)

Требуемые файлы:
- Файл молекул в SDF формате
- Белок-мишень в PDBQT и PDB форматах
- Конфиг-файл для AutoDock Vina

Результаты:
- dockscore.csv - таблица с энергиями связывания
- molname_docked.sdf.gz - докированные молекулы в сжатом формате

Пример использования:
    from chemplus.unicore import docking

    job_id_file = docking.unicore_dock(
        ucc_path="C:/ucc.bat",
        site_name="SKIF_GRID_CIS",
        server_executable="/path/to/python",
        server_work_dir="~/projects/docking",
        cpu_count=80,
        local_work_dir="results",
        local_receptor="receptor.pdbqt",
        local_pdb_receptor="receptor.pdb",
        local_config="config.txt",
        local_sdf="ligands.sdf",
        sync=False  # Асинхронно
    )
    # Позже получить результаты:
    docking.get_docking_results(ucc_path, job_id_file, "results")
"""

import os
import json
import time
from datetime import datetime
from chemplus import unicore

def get_docking_results(ucc_path, job_id_file, output_dir="."):
    """
    Получить результаты докинга с GRID кластера.

    Скачивает файлы результатов (dockscore.csv и sdf) после завершения задачи.
    Должна использоваться после unicore_dock(..., sync=False).

    Параметры:
        ucc_path (str): Путь к UNICORE Command Client (ucc.bat)
        job_id_file (str): Файл ID задачи (возвращается unicore_dock)
        output_dir (str): Локальная директория для сохранения результатов

    Возвращает:
        str: Статус задачи (если "exit code: 0" - успешно)

    Загружаемые файлы:
        - dockscore.csv: таблица энергий связывания
        - *_docked.sdf.gz: молекулы в позе с лучшей энергией
        - stdout/stderr логи выполнения задачи

    Пример:
        status = get_docking_results(ucc_path, job_id_file, "./results/")
        if "exit code: 0" in status:
            print("Докинг завершен успешно!")
    """
    status = unicore.get_job_status(ucc_path, job_id_file)
    if "exit code" not in status:
        return "Job status: " + status 
    
    with open(job_id_file, "r") as json_file:
        data = json.load(json_file)
    
    for i, arg in enumerate(data["Arguments"]):
        if arg == "--work_dir":
            work_dir = data["Arguments"][i+1]
        if arg == "--sdf":
            sdf = data["Arguments"][i+1]
    
    if work_dir.startswith("'"):
        work_dir = work_dir[1:]
    if work_dir.endswith("'"):
        work_dir = work_dir[:-1]
    if sdf.startswith("'"):
        sdf = sdf[1:]
    if sdf.endswith("'"):
        sdf = sdf[:-1]
    
    dockscore_file = work_dir + "/dockscore.csv"
    dockscore_file = unicore.get_correct_location(data["Site"], dockscore_file)
    sdf_docked_file = work_dir + "/" + os.path.basename(sdf).split(".")[0] + "_docked.sdf.gz"
    sdf_docked_file = unicore.get_correct_location(data["Site"], sdf_docked_file)
    
    unicore.get_file(ucc_path, dockscore_file, data["Output"])
    unicore.get_file(ucc_path, sdf_docked_file, data["Output"])
    unicore.get_job_out(ucc_path, job_id_file, output_dir)
    return status

def unicore_dock(ucc_path, site_name, server_executable, server_work_dir, cpu_count,
                 local_work_dir, local_receptor, local_pdb_receptor, local_config, local_sdf, sync=True, serial=False, rewrite=True,
                 pythonpath=None):
    """
    Запустить молекулярный докинг на GRID кластере SKIF.

    Основная функция для запуска полного цикла докинга на суперкомпьютере.
    Автоматически:
    1. Создает рабочую директорию на сервере
    2. Загружает все необходимые файлы
    3. Рассчитывает оптимальное распределение вычислительных ресурсов
    4. Создает конфиг задачи для кластера
    5. Запускает задачу
    6. Получает результаты (если sync=True)

    Параметры:
        ucc_path (str): Полный путь к UNICORE Command Client (уcc.bat)
                       Пример: "C:/Users/USER/software/ucc-1.3.1/bin/ucc.bat"

        site_name (str): Название GRID сайта
                        "SKIF_GRID_CIS" - кластер SKIF Grid CIS (40 CPU/узел)
                        "SKIF_GEO" - кластер SKIF GEO (36 CPU/узел)

        server_executable (str): Абсолютный путь к Python интерпретатору на сервере
                                Пример: "/share/bio/data/miniconda3/envs/chem/bin/python"

        server_work_dir (str): Рабочая директория на сервере (будет создана автоматически)
                              Пример: "~/projects/my_docking_project"

        cpu_count (int): Требуемое количество CPU ядер
                        Автоматически вычисляется оптимальное распределение
                        Примеры: 40, 80, 120

        local_work_dir (str): Локальная директория для сохранения файла ID задачи
                             Пример: "C:/docking_results"

        local_receptor (str): Путь к локальному файлу рецептора в PDBQT формате
                             Пример: "receptor.pdbqt"

        local_pdb_receptor (str): Путь к локальному файлу рецептора в PDB формате
                                 Пример: "receptor.pdb"

        local_config (str): Путь к конфиг-файлу AutoDock Vina
                           Пример: "config.txt"
                           Файл должен содержать:
                               receptor = /path/to/receptor.pdbqt
                               center_x = 10.5
                               center_y = 20.3
                               center_z = 15.8
                               size_x = 20
                               size_y = 20
                               size_z = 20

        local_sdf (str): Путь к файлу молекул в SDF формате
                        Пример: "ligands.sdf"

        sync (bool): Режим выполнения
                    True - ждать завершения (синхронно)
                    False - вернуть сразу (асинхронно, нужно проверить статус позже)
                    По умолчанию True

        serial (bool): Режим параллелизма
                      True - последовательный (обычно не используется)
                      False - параллельный с MPI (рекомендуется)
                      По умолчанию False

        rewrite (bool): Перезаписывать результаты если они уже существуют
                       По умолчанию True

        pythonpath (str): Абсолютный путь на сервере, который встанет в PYTHONPATH
                         задачи. Нужен, чтобы своя копия chemplus перекрыла общую
                         в /share/bio/data/chemplus. Раскрытие "~" не работает —
                         путь должен быть абсолютным.
                         Пример: "/home/USERNAME/projects/docking/chemplus_local"
                         По умолчанию None (переменная не задаётся)

    Возвращает:
        Если sync=True:
            None (результаты скачаны автоматически)

        Если sync=False:
            str: Путь к файлу ID задачи (например, "receptor_config_ligands_2024-01-15_10-30-45_docking.u")
                 Используется для проверки статуса:
                    status = unicore.get_job_status(ucc_path, job_id_file)

    Автоматическое распределение ресурсов:
        SKIF_GRID_CIS (40 CPU/узел):
            cpu_count=80  → 2 узла × 40 CPU = 93 GB памяти на узел
            cpu_count=40  → 1 узел × 40 CPU = 93 GB памяти на узел
            cpu_count=120 → 3 узла × 40 CPU = 93 GB памяти на узел

    Пример использования:
        # Синхронный запуск (ждать результатов)
        docking.unicore_dock(
            ucc_path="C:/ucc.bat",
            site_name="SKIF_GRID_CIS",
            server_executable="/share/bio/data/miniconda3/envs/chem/bin/python",
            server_work_dir="~/projects/drug_discovery",
            cpu_count=80,
            local_work_dir="./results",
            local_receptor="receptor.pdbqt",
            local_pdb_receptor="receptor.pdb",
            local_config="config.txt",
            local_sdf="compounds.sdf",
            sync=True
        )

        # Асинхронный запуск (не ждать)
        job_id = docking.unicore_dock(
            # ... же параметры ...
            sync=False
        )
        # Позже проверить статус:
        from chemplus import unicore
        status = unicore.get_job_status(ucc_path, job_id)
        if "exit code: 0" in status:
            docking.get_docking_results(ucc_path, job_id, "./results")

    Требуемые установки на сервере:
        - ChemPlus пакет с модулями chemplus.vina и chemplus.tasks
        - AutoDock Vina
        - MGLTools для преобразования PDB → PDBQT
        - Python 3.9+

    Вызывает исключение если:
        - Файлы не найдены
        - Недостаточно ресурсов на кластере
        - Ошибка подключения к GRID
    """
    server_work_dir_orig = server_work_dir
    server_work_dir = unicore.get_correct_location(site_name, server_work_dir)
    unicore.create_dir(ucc_path, server_work_dir)
    unicore.put_file(ucc_path, local_sdf, server_work_dir)
    unicore.put_file(ucc_path, local_receptor, server_work_dir)
    unicore.put_file(ucc_path, local_pdb_receptor, server_work_dir)
    unicore.put_file(ucc_path, local_config, server_work_dir)
    time.sleep(5)
    
    receptor_name = os.path.basename(local_receptor).split(".")[0]
    config_name = os.path.basename(local_config).split(".")[0]
    sdf_name = os.path.basename(local_sdf).split(".")[0]
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    job_file_name = receptor_name + "_" + config_name + "_" + sdf_name + "_" + current_time + "_docking.u"
    job_file_path = local_work_dir + os.sep + job_file_name

    if site_name == "SKIF_GEO":
        nodes = cpu_count // 36 + (cpu_count % 36 > 0)
        cpus_per_node = cpu_count // nodes
        memory = 185 * (cpus_per_node / 36)
    elif site_name == "SKIF_GRID_CIS":
        nodes = cpu_count // 40 + (cpu_count % 40 > 0)
        cpus_per_node = cpu_count // nodes
        memory = 93 * (cpus_per_node / 40)
    
    # Derive work_dir for job arguments from original server path
    if server_work_dir_orig.startswith("~"):
        work_dir = server_work_dir_orig
    elif server_work_dir_orig.startswith("/home/"):
        work_dir = "~" + server_work_dir_orig[server_work_dir_orig.index("/", 6):]
    else:
        work_dir = "~" + server_work_dir_orig
    server_receptor = "'" + work_dir + "/" + os.path.basename(local_receptor) + "'"
    server_config = "'" + work_dir + "/" + os.path.basename(local_config) + "'"
    server_sdf = "'" + work_dir + "/" + os.path.basename(local_sdf) + "'"
    
    work_dir = "'" + work_dir + "'"
    args = ["-m", "chemplus.tasks.docking", "--work_dir", work_dir, "--receptor", server_receptor, 
            "--config", server_config, "--sdf", server_sdf, "--cpu_count", cpu_count]
    if serial:
        args += ["--serial"]
    if rewrite:
        args += ["--rewrite"]
    env_vars = ["PYTHONPATH=" + pythonpath] if pythonpath else None
    unicore.create_job_file(job_file_path, server_executable, arguments_list=args, memory=memory,
                        nodes=nodes, cpus_per_node=cpus_per_node, run_time=800, environment_vars=env_vars)
    
    if sync:
        unicore.run_sync(ucc_path, site_name, job_file_path, local_work_dir)
        get_docking_results(ucc_path, local_work_dir, local_receptor, local_config, local_sdf, server_work_dir)
    else:
        return unicore.run_async(ucc_path, site_name, job_file_path, local_work_dir)
    