"""
UNICORE интеграция - модуль для работы с GRID кластером SKIF

Этот модуль предоставляет функции для взаимодействия с суперкомпьютером через
UNICORE Command Client (UCC). Позволяет:
- Загружать и скачивать файлы на/с сервера
- Создавать и запускать задачи на кластере
- Проверять статус выполнения задач
- Управлять ресурсами (CPU, память, время выполнения)

Основной workflow:
1. Создать конфигурационный файл задачи (create_job_file)
2. Проверить ресурсы сервера (check_site_resources_for_job)
3. Загрузить необходимые файлы (put_file)
4. Запустить задачу синхронно или асинхронно (run_sync/run_async)
5. Получить результаты (get_file, get_job_out)

Пример использования:
    ucc_path = "C:/path/to/ucc.bat"
    site_name = "SKIF_GRID_CIS"
    create_job_file("job.u", "/path/to/python",
                    arguments_list=["-m", "mymodule"],
                    memory=4, nodes=2, cpus_per_node=20, run_time=1)
    run_sync(ucc_path, site_name, "job.u", "output_dir")
"""

import os
import re
import json
import subprocess
import sys
from xml.etree import cElementTree as ET

def get_site_resources(ucc_path, site_name):
    """
    Получить информацию о ресурсах доступных на GRID сайте.

    Запрашивает у кластера информацию о максимальных и минимальных значениях:
    - Памяти на узел
    - Времени выполнения
    - Количества CPU на узел
    - Количества узлов
    - Общего количества CPU

    Параметры:
        ucc_path (str): Полный путь к UNICORE Command Client (ucc.bat)
        site_name (str): Название сайта, например "SKIF_GRID_CIS"

    Возвращает:
        dict: Словарь с ключами:
            - 'memory': кортеж (min, max) в байтах
            - 'runtime': кортеж (min, max) в секундах
            - 'node_cpus': кортеж (min, max) CPU на узел
            - 'nodes': кортеж (min, max) количество узлов
            - 'cpus': кортеж (min, max) общее количество CPU

    Вызывает исключение если сайт не найден или доступ невозможен.
    """
    cmd_args = [ucc_path, "list-sites", "--all"]
    res = run_command(cmd_args)
    
    site_index = res.index(f"{site_name} ")
    xml_begin = "<rp:GetResourcePropertyDocumentResponse"
    xml_begin_index = res.index(xml_begin, site_index)
    xml_end = "</rp:GetResourcePropertyDocumentResponse>"
    xml_end_index = res.index(xml_end, site_index) + len(xml_end)
    
    root = ET.fromstring(res[xml_begin_index:xml_end_index])
    
    memory_min = int(root.find(".//{*}TargetSystemProperties//{*}IndividualPhysicalMemory//{*}Range//{*}LowerBound").text)
    memory_max = int(root.find(".//{*}TargetSystemProperties//{*}IndividualPhysicalMemory//{*}Range//{*}UpperBound").text)

    runtime_min = int(root.find(".//{*}TargetSystemProperties//{*}IndividualCPUTime//{*}Range//{*}LowerBound").text)
    runtime_max = int(root.find(".//{*}TargetSystemProperties//{*}IndividualCPUTime//{*}Range//{*}UpperBound").text)

    node_cpus_min = int(float(root.find(".//{*}TargetSystemProperties//{*}IndividualCPUCount//{*}Range//{*}LowerBound").text))
    node_cpus_max = int(float(root.find(".//{*}TargetSystemProperties//{*}IndividualCPUCount//{*}Range//{*}UpperBound").text))

    nodes_min = int(float(root.find(".//{*}TargetSystemProperties//{*}TotalResourceCount//{*}Range//{*}LowerBound").text))
    nodes_max = int(float(root.find(".//{*}TargetSystemProperties//{*}TotalResourceCount//{*}Range//{*}UpperBound").text))

    cpus_min = int(float(root.find(".//{*}TargetSystemProperties//{*}TotalCPUCount//{*}Range//{*}LowerBound").text))
    cpus_max = int(float(root.find(".//{*}TargetSystemProperties//{*}TotalCPUCount//{*}Range//{*}UpperBound").text))
    
    return {"memory" : (memory_min, memory_max), "runtime" : (runtime_min, runtime_max), "node_cpus" : (node_cpus_min, node_cpus_max), "nodes" : (nodes_min, nodes_max), "cpus" :(cpus_min, cpus_max)}

def create_job_file(job_file_path, server_executable, arguments_list=[], memory=1, nodes=1, cpus_per_node=1, run_time=1, 
                     file_to_execute=None, environment_vars=None):
    """
    Создать JSON-файл конфигурации задачи для запуска на GRID кластере.

    Генерирует .u файл с параметрами задачи, которая будет выполняться на суперкомпьютере.

    Параметры:
        job_file_path (str): Путь где сохранить файл задачи (например, "job.u")
        server_executable (str): Путь к исполняемому файлу на сервере
                                (например, "/share/bio/miniconda3/envs/chem/bin/python")
        arguments_list (list): Список аргументов командной строки
                              (по умолчанию пусто)
        memory (int/float): Объем памяти в гигабайтах (по умолчанию 1 GB)
        nodes (int): Количество узлов кластера (по умолчанию 1)
        cpus_per_node (int): Количество CPU на каждый узел (по умолчанию 1)
        run_time (int/float): Максимальное время выполнения в часах (по умолчанию 1 час)
        file_to_execute (str): Локальный файл скрипта для загрузки на сервер (опционально)
        environment_vars (list): Список переменных окружения (опционально)

    Пример:
        create_job_file("docking_job.u",
                       "/share/bio/data/miniconda3/envs/chem/bin/python",
                       arguments_list=["-m", "chemplus.tasks.docking", "--work_dir", "~/dock"],
                       memory=10,
                       nodes=2,
                       cpus_per_node=40,
                       run_time=10)

    Генерирует JSON с ключами: Executable, Arguments, Resources, Imports (если есть), Environment
    """

    if not isinstance(server_executable, str):
        raise Exception("Server executable must be of type STR")
    
    if not isinstance(arguments_list, list) and not isinstance(arguments_list, tuple):
        raise Exception("Arguments list must be a tuple or a list")
    
    data = {"Executable" : server_executable, "Arguments" : arguments_list, "Resources" : dict()}
    
    if file_to_execute is not None:
        if not os.path.isfile(file_to_execute):
            raise Exception("Non-valid file location: " + file_to_execute)
        file_basename = os.path.basename(file_to_execute)
        data["Imports"] = [{"From" : file_to_execute, "To": file_basename}]
        data["Arguments"] = [file_basename] + data["Arguments"]
    
    if not isinstance(memory, int) and not isinstance(memory, float):
        raise Exception("Memory value must be of type INT/FLOAT")
    data["Resources"]["Memory"] = int(memory * 1024**3)
    
    if not isinstance(nodes, int):
        raise Exception("Nodes number must be of type INT")
    data["Resources"]["Nodes"] = nodes
    
    if not isinstance(cpus_per_node, int):
        raise Exception("CPUsPerNode number must be of type INT")
    data["Resources"]["CPUsPerNode"] = cpus_per_node
    
    if not isinstance(run_time, int) and not isinstance(run_time, float):
        raise Exception("Runtime value must be of type INT/FLOAT")
    data["Resources"]["Runtime"] = int(run_time * 3600)
    
    if environment_vars is not None:
        if not isinstance(environment_vars, list) and not isinstance(environment_vars, tuple):
            raise Exception("Environment vars must be of type LIST/TUPLE")
        data["Environment"] = environment_vars
    
    with open(job_file_path, "w") as json_file:
        json.dump(data, json_file, indent="  ")

def check_site_resources_for_job(ucc_path, site_name, job_file_path):
    """
    Проверить соответствие параметров задачи ресурсам GRID сайта.

    Сравнивает требуемые ресурсы в файле задачи с доступными ресурсами сайта.
    Вызывает исключение если ресурсы выходят за пределы допустимых значений.

    Проверяемые параметры:
    - Память на узел
    - Количество узлов
    - CPU на узел
    - Время выполнения

    Параметры:
        ucc_path (str): Путь к UCC (UNICORE Command Client)
        site_name (str): Название GRID сайта
        job_file_path (str): Путь к файлу задачи (job.u)

    Вызывает исключение если параметры не соответствуют.
    """
    resources = get_site_resources(ucc_path, site_name)
    
    with open(job_file_path, "r") as json_file:
        data = json.load(json_file)
    
    if not resources["memory"][0] <= data["Resources"]["Memory"] <= resources["memory"][1]:
        raise Exception(site_name + ": memory out of bounds")
        
    if not resources["nodes"][0] <= data["Resources"]["Nodes"] <= resources["nodes"][1]:
        raise Exception(site_name + ": nodes out of bounds")
    
    if not resources["node_cpus"][0] <= data["Resources"]["CPUsPerNode"] <= resources["node_cpus"][1]:
        raise Exception(site_name + ": cpus per node out of bounds")

    if not resources["runtime"][0] <= data["Resources"]["Runtime"] <= resources["runtime"][1]:
        raise Exception(site_name + ": run time out of bounds")    
    
def run_command(cmd_args, timeout=None):
    """
    Выполнить UCC команду и вернуть результат.

    Запускает команду UCC (UNICORE Command Client) и возвращает стандартный вывод.
    Вызывает исключение если команда завершилась с ошибкой.

    Параметры:
        cmd_args (list): Список аргументов команды для subprocess
        timeout (float|None): Максимальное время ожидания в секундах. По
                             умолчанию None — без ограничения (как раньше);
                             используй явный таймаут для вызовов, которые не
                             должны блокировать скрипт навсегда (см. вызовы
                             ucc.bat из get_docking_progress).

    Возвращает:
        str: Стандартный вывод команды

    Вызывает subprocess.TimeoutExpired, если команда не уложилась в timeout.

    Пример:
        result = run_command([ucc_path, "list-sites", "--all"])
    """

    result = subprocess.run(cmd_args, capture_output=True, timeout=timeout)
    # Используем системную кодировку для Windows (cp1251/cp866) вместо utf-8
    encoding = sys.stdout.encoding or 'utf-8'
    stderr = result.stderr.decode(encoding, errors='replace').strip()
    if stderr:
        raise Exception(stderr)
    else:
        return result.stdout.decode(encoding, errors='replace').strip()

def get_correct_location(site_name, path):
    """
    Преобразовать локальный путь в адрес UNICORE формата для сервера.

    Конвертирует различные форматы путей в стандартный UNICORE формат:
    u6://SITE_NAME/Home/username/path/to/file

    Принимает форматы:
    - ~/path (домашняя директория)
    - /home/username/path (абсолютный Linux путь)
    - Уже готовый UNICORE адрес

    Параметры:
        site_name (str): Название GRID сайта
        path (str): Путь в любом формате

    Возвращает:
        str: Путь в формате u6://SITE_NAME/Home/username/...

    Вызывает исключение если путь имеет некорректный формат.
    """
    if path.startswith("u6://" + site_name + "/Home") or path.startswith("unicore6://" + site_name + "/Home"):
        return path
    
    prefix = "u6://" + site_name + "/Home"
    if path.startswith("~"):
        return prefix + path[1:]
    else:
        str_match = re.match(r"/home/[^/]+", path)
        if str_match:
            return prefix + path[str_match.start():]
        else:
            raise Exception("Incorrect server path")

def ls_dir(ucc_path, server_dir_path, timeout=None):
    """
    Вывести содержимое директории на сервере.

    Эквивалент команды 'ls -l' на удаленном сервере.

    Параметры:
        ucc_path (str): Путь к UCC
        server_dir_path (str): Путь директории на сервере в формате UNICORE
        timeout (float|None): См. run_command()

    Возвращает:
        str: Отформатированный список файлов и директорий
    """
    cmd_args = [ucc_path, "ls", "-l", server_dir_path]

    return run_command(cmd_args, timeout=timeout)

def create_dir(ucc_path, server_dir):
    """
    Создать директорию на сервере.

    Параметры:
        ucc_path (str): Путь к UCC
        server_dir (str): Путь новой директории в формате UNICORE

    Пример:
        create_dir(ucc_path, "u6://SKIF_GRID_CIS/Home/myuser/projects/docking")
    """
    cmd_args = [ucc_path, "mkdir", server_dir]
    
    return run_command(cmd_args)

def get_file(ucc_path, server_file_path, local_destination_dir=".", timeout=None):
    """
    Скачать один файл с сервера на локальный компьютер.

    Параметры:
        ucc_path (str): Путь к UCC
        server_file_path (str): Полный путь к файлу на сервере в формате UNICORE
        local_destination_dir (str): Локальная директория для сохранения файла
                                    (по умолчанию текущая директория)
        timeout (float|None): См. run_command()

    Пример:
        get_file(ucc_path, "u6://SKIF_GRID_CIS/Home/user/projects/results.txt", "./local_results/")
    """
    destination_file_path = local_destination_dir + "/" + os.path.basename(server_file_path)
    cmd_args = [ucc_path, "get-file", "-s", server_file_path, "-t", destination_file_path]

    return run_command(cmd_args, timeout=timeout)

def get_dir_files(ucc_path, server_dir_path, local_destination_dir="."):
    """
    Скачать все файлы из директории на сервере.

    Параметры:
        ucc_path (str): Путь к UCC
        server_dir_path (str): Путь директории на сервере в формате UNICORE
        local_destination_dir (str): Локальная директория для сохранения файлов

    Примечание: Загружаются только файлы, не рекурсивно в поддиректории.
    """
    ls_result = ls_dir(ucc_path, server_dir_path)
    home_prefix = server_dir_path.split("Home")[0] + "Home"
    
    for file_info in ls_result.split("\n"):
        if file_info[0] == "-":
            server_file_path = home_prefix + file_info[37:]
            get_file(ucc_path, server_file_path, local_destination_dir)

def put_file(ucc_path, local_file_path, server_destination_dir):
    """
    Загрузить файл с локального компьютера на сервер.

    Параметры:
        ucc_path (str): Путь к UCC
        local_file_path (str): Полный путь к локальному файлу
        server_destination_dir (str): Директория на сервере в формате UNICORE
                                     где сохранить файл

    Пример:
        put_file(ucc_path, "C:/ligands.sdf", "u6://SKIF_GRID_CIS/Home/user/projects/dock/")
    """
    destination_file_path = server_destination_dir + "/" + os.path.basename(local_file_path)
    cmd_args = [ucc_path, "put-file", "-s", local_file_path, "-t", destination_file_path]
    
    return run_command(cmd_args)

def run_sync(ucc_path, site_name, job_file, output_dir):
    """
    Запустить задачу синхронно и ждать её завершения.

    Отправляет задачу на выполнение на GRID кластер и ждет когда она
    полностью завершится. Проверяет ресурсы перед запуском.

    Параметры:
        ucc_path (str): Путь к UCC
        site_name (str): Название GRID сайта (например, "SKIF_GRID_CIS")
        job_file (str): Путь к файлу задачи (job.u)
        output_dir (str): Локальная директория для сохранения выходных файлов

    Возвращает:
        str: Результат выполнения команды (информация о задаче)

    Пример:
        result = run_sync(ucc_path, "SKIF_GRID_CIS", "docking.u", "./results/")
    """

    check_site_resources_for_job(ucc_path, site_name, job_file)

    cmd_args = [ucc_path, "run", "-s", site_name, "-o", output_dir, job_file]
    
    return run_command(cmd_args)

def run_async(ucc_path, site_name, job_file, output_dir):
    """
    Запустить задачу асинхронно (не ждать завершения).

    Отправляет задачу на выполнение и сразу возвращает ID файл.
    Позволяет запустить задачу и проверить результаты позже.

    Параметры:
        ucc_path (str): Путь к UCC
        site_name (str): Название GRID сайта
        job_file (str): Путь к файлу задачи
        output_dir (str): Локальная директория для вывода

    Возвращает:
        str: Путь к файлу ID задачи (например, "0b123456-7890-abcd-ef01-234567890abc.u")
             Используется для проверки статуса и получения результатов

    Пример:
        job_id_file = run_async(ucc_path, "SKIF_GRID_CIS", "docking.u", "./results/")
        # Позже проверить статус:
        status = get_job_status(ucc_path, job_id_file)
    """

    check_site_resources_for_job(ucc_path, site_name, job_file)

    cmd_args = [ucc_path, "run", "-s", site_name, "-o", output_dir, "-a", job_file]
    
    return run_command(cmd_args).split("\n")[0].strip()
    
def get_job_status(ucc_path, job_id_file):
    """
    Получить статус выполнения задачи.

    Параметры:
        ucc_path (str): Путь к UCC
        job_id_file (str): Путь к файлу ID задачи (возвращается run_async)

    Возвращает:
        str: Информация о статусе задачи. Содержит:
             - "Job status: ..." если задача еще выполняется
             - "exit code: 0" если успешно завершена
             - "exit code: X" (X > 0) если произошла ошибка

    Пример:
        status = get_job_status(ucc_path, "task_id.u")
        if "exit code: 0" in status:
            print("Задача завершена успешно!")
    """
    cmd_args = [ucc_path, "get-status", job_id_file]
    
    return run_command(cmd_args)

def get_job_out(ucc_path, job_id_file, output_dir="."):
    """
    Получить выходные файлы задачи.

    Скачивает все выходные файлы (stdout, stderr и т.д.)
    на локальный компьютер.

    Параметры:
        ucc_path (str): Путь к UCC
        job_id_file (str): Путь к файлу ID задачи
        output_dir (str): Локальная директория для сохранения выходных файлов

    Пример:
        get_job_out(ucc_path, "task_id.u", "./results/")
    """
    cmd_args = [ucc_path, "get-output", job_id_file, "-o", output_dir]
    
    return run_command(cmd_args)

def list_jobs(ucc_path):
    """
    Вывести список всех задач пользователя на GRID кластере.

    Параметры:
        ucc_path (str): Путь к UCC

    Возвращает:
        str: Отформатированный список задач с информацией о статусе
    """
    cmd_args = [ucc_path, "list-jobs"]
    
    return run_command(cmd_args)

def abort_job(ucc_path, job_id_file):
    """
    Остановить (отменить) выполняющуюся задачу.

    Параметры:
        ucc_path (str): Путь к UCC
        job_id_file (str): Путь к файлу ID задачи

    Пример:
        abort_job(ucc_path, "task_id.u")
    """
    cmd_args = [ucc_path, "abort-job", job_id_file]
    
    return run_command(cmd_args)