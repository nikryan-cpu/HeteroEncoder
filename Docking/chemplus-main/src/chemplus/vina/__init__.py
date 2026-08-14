"""
Модуль VINA - молекулярный докинг с использованием AutoDock Vina.

Этот модуль предоставляет функции для выполнения молекулярного докинга:
- Локально на одном компьютере (serial)
- Параллельно с использованием MPI (для кластеров)
- На GRID кластере через UNICORE

Основные подмодули:
- docking.py: Основной интерфейс для локального докинга
- docking_serial.py: Последовательный докинг (одна молекула за раз)
- docking_mpi.py: Параллельный докинг с MPI (несколько молекул одновременно)
- pdbqt.py: Работа с PDBQT форматом (преобразование PDB, проверка связей)
- dockscore.py: Подсчет результатов докинга (Vina score, RF-score, NNScore)
- pdbqt_to_sdf.py: Конвертация результатов из PDBQT в SDF
- rot_bond_helper.py: Обработка ротируемых связей молекул

Основной workflow:
1. Подготовить молекулы (SDF → PDB → PDBQT)
2. Подготовить рецептор (PDB → PDBQT)
3. Запустить докинг
4. Обработать результаты (PDBQT → SDF, подсчет энергий)

Пример локального докинга:
    from chemplus.vina import docking

    docking.sdf_dock_solved_path(
        work_dir="docking_results",
        receptor_file="receptor.pdbqt",
        config_file="config.txt",
        sdf_file="ligands.sdf",
        cpu_count=4,
        serial=True,
        rewrite=True
    )

Пример для GRID:
    from chemplus.unicore import docking as unicore_docking

    unicore_docking.unicore_dock(
        ucc_path="C:/ucc.bat",
        site_name="SKIF_GRID_CIS",
        server_executable="/path/to/python",
        server_work_dir="~/projects/docking",
        cpu_count=80,
        local_work_dir="results",
        local_receptor="receptor.pdbqt",
        local_pdb_receptor="receptor.pdb",
        local_config="config.txt",
        local_sdf="ligands.sdf"
    )
"""

