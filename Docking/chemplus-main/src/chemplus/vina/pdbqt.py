"""
Модуль для работы с PDBQT форматом - специальный формат для AutoDock Vina.

PDBQT (Protein Data Bank, Partial Charges, Atom Types) - расширение формата PDB.
Содержит:
- Структуру молекулы (как PDB)
- Атомные типы AutoDock (GAFF типы)
- Частичные заряды
- Информацию о ротируемых связях

Функции этого модуля:
1. Конвертация PDB → PDBQT (через prepare_receptor4.py для белков)
2. Проверка и исправление ротируемых связей
3. Фильтрация молекул с избыточным количеством связей
4. Извлечение лучшей позы из результатов докинга

Требует: MGLTools с Python интерпретатором

Типичный workflow:
    1. pdb_to_pdbqt(pdb_file, pdbqt_file, mgl_python)
    2. check_pdbqt_dir_rot_bonds(pdbqt_dir, pdb_dir, fix_bad_pdbqt=True)
    3. filter_pdbqt(pdbqt_dir, delete_overnrb_pdbqt=True)
    4. После докинга: get_top_model(docked_dir, top_model_dir)

Ограничения AutoDock Vina:
- Максимум 20 ротируемых связей в молекуле
- Максимум около 1000 атомов
"""

from chemplus.vina import rot_bond_helper
from chemplus import draw
import os
import glob
import subprocess
from joblib import Parallel, delayed
from tqdm import tqdm
from sys import platform
from rdkit import Chem


def _prepare_receptor4_path(mgl_python):
    """
    prepare_receptor4.py входит в штатную поставку MGLTools (AutoDockTools) —
    берём его оттуда, а не из приватной копии в chemnotes-main. Раньше путь
    был константой, посчитанной смещением от __file__ этого модуля в
    расчёте на локальную вложенность Docking/chemplus-main/src/chemplus/vina/;
    на кластере chemplus_local/ (см. tools/sync_cluster.py) той же глубины,
    но chemnotes-main туда не заливается вообще — офсет попадал в
    ~/projects/chemnotes-main, а он либо битый симлинк (SKIF_GRID_CIS,
    указывает на исчезнувший /share/bio/data/chemnotes-main), либо
    отсутствует (SKIF_GEO). Итог — конвертация PDB->PDBQT падала на каждой
    молекуле с "No such file or directory". Ищем скрипт рядом с mgl_python
    (.../bin/pythonsh -> .../MGLToolsPckgs/AutoDockTools/Utilities24/) —
    так путь верен на любом сайте, где вообще нашёлся MGLTools.
    """
    mgltools_root = os.path.dirname(os.path.dirname(os.path.realpath(mgl_python)))
    return os.path.join(mgltools_root, "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_receptor4.py")


def pdb_to_pdbqt(pdb_file, pdbqt_file, mgl_python, overwrite=True, silent=False, timeout=300):
    """
    Конвертировать один PDB файл в PDBQT формат для AutoDock Vina.

    Использует MGLTools скрипт prepare_receptor4.py для добавления:
    - Атомных типов
    - Частичных зарядов
    - Информации о ротируемых связях

    Параметры:
        pdb_file (str): Путь к исходному PDB файлу
        pdbqt_file (str): Путь для сохранения PDBQT файла
        mgl_python (str): Путь к Python интерпретатору из MGLTools
                         Пример: "C:/MGLTools/python.exe"
        overwrite (bool): Перезаписывать если PDBQT уже существует (по умолчанию True)
        silent (bool): Не выводить сообщения об ошибках (по умолчанию False)

    Возвращает:
        1 если успешно
        str (имя файла) если произошла ошибка

    Вызывает исключение если:
        - mgl_python не найден
        - Конвертация заняла более `timeout` секунд (timeout)
    """
    if not os.path.exists(mgl_python):
        raise Exception("MGL python is not exists")

    if not overwrite and os.path.exists(pdbqt_file):
         return

    pdb_name = os.path.basename(pdb_file).split('.')[0]

    pr4_py = _prepare_receptor4_path(mgl_python)
    if not os.path.exists(pr4_py):
        raise Exception("prepare_receptor4.py not found next to mgl_python: " + pr4_py)

    cmd_args = [mgl_python, pr4_py, '-r', pdb_file, '-o', pdbqt_file, '-A', 'checkhydrogens']

    try:
        result = subprocess.run(cmd_args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if not silent:
            print('\nTimeoutExpired:', pdb_name)
        return pdb_name

    if result.returncode:
        if not silent:
            print('\nError:', pdb_name, "-", result.stderr.decode("utf-8").strip())
        return pdb_name

    return 1

def pdb_dir_to_pdbqt(pdb_dir, pdbqt_dir, mgl_python, cpu_count=2, overwrite=True, silent=False):
    """
    Конвертировать все PDB файлы в директории в PDBQT формат.
    
    Параллельно обрабатывает все .pdb файлы в директории.
    
    Параметры:
        pdb_dir (str): Директория с PDB файлами
        pdbqt_dir (str): Директория для сохранения PDBQT файлов (будет создана)
        mgl_python (str): Путь к Python интерпретатору MGLTools
        cpu_count (int): Количество параллельных процессов
        overwrite (bool): Перезаписывать существующие (по умолчанию True)
        silent (bool): Не выводить сообщения об ошибках (по умолчанию False)
    
    Возвращает:
        (pdb_num, converted_pdb_num, bad_pdb_list)
        
        pdb_num (int): Всего файлов найдено
        converted_pdb_num (int): Успешно конвертировано
        bad_pdb_list (list): Список файлов, которые не удалось конвертировать
    
    Пример:
        total, converted, bad = pdb_dir_to_pdbqt(
            "molecules/pdb/",
            "molecules/pdbqt/",
            "C:/MGLTools/python.exe",
            cpu_count=4
        )
        print(f"Конвертировано {converted} из {total} молекул")
    """
    if not os.path.exists(pdb_dir):
        raise Exception("PDB folder is not exists")

    if not os.path.exists(mgl_python):
        raise Exception("MGL python is not exists")

    if not os.path.exists(pdbqt_dir):
        os.makedirs(pdbqt_dir)

    cpu_count = min(os.cpu_count(), cpu_count)
    exit_codes = Parallel(n_jobs=cpu_count, verbose=0)(delayed(pdb_to_pdbqt)(pdb_file, pdbqt_dir + os.sep + os.path.basename(pdb_file).split('.')[0] + ".pdbqt", mgl_python, overwrite, silent) for pdb_file in tqdm(glob.glob(pdb_dir + os.sep + '*.pdb')))

    pdb_num = len(exit_codes)
    bad_pdb_list = []
    converted_pdb_num = 0

    for code in exit_codes:
        if code == 1:
            converted_pdb_num += 1
        elif type(code) is str:
            bad_pdb_list.append(code)

    return pdb_num, converted_pdb_num, bad_pdb_list

def get_rotbond_atoms_pdb(pdb_file):    
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    
    rot_bonds = rot_bond_helper.get_rot_bonds(mol)
    rot_bonds_atoms = []
    
    for bond in rot_bonds:
        first_atom = mol.GetBondWithIdx(bond).GetBeginAtom().GetPDBResidueInfo().GetSerialNumber()
        second_atom = mol.GetBondWithIdx(bond).GetEndAtom().GetPDBResidueInfo().GetSerialNumber()
        rot_bonds_atoms.append((first_atom, second_atom))
        
    return rot_bonds_atoms

def get_rotbond_atoms_pdbqt(pdbqt_file):
    lines = open(pdbqt_file).readlines()
    rotbond_lines = lines[2:lines.index("ROOT\n")]
    
    rot_bonds_atoms = []

    for line in rotbond_lines:
        if line[13] != "A":
            continue
        atoms_info = line[32:].split()
    
        first_atom = int(atoms_info[0].split("_")[1])
        second_atom = int(atoms_info[2].split("_")[1])
        rot_bonds_atoms.append((first_atom, second_atom))
        
    return rot_bonds_atoms

def rot_bonds_check(pdbqt_list, original_list, silent=True):
    errors = ([], [])
    
    for atom_pair in pdbqt_list:
        #non-rotatable bond is rotatable        
        if atom_pair not in original_list and (atom_pair[1], atom_pair[0]) not in original_list:
            error=True
            if not silent:
                print("ERROR: non-rotatable bond between atoms", atom_pair, "is rotatable")
            errors[0].append(atom_pair)
    
    for atom_pair in original_list:
        #rotatable bond is not rotatable
        if atom_pair not in pdbqt_list and (atom_pair[1], atom_pair[0]) not in pdbqt_list:
            warning=True
            if not silent:
                print("WARNING: rotatable bond between atoms", atom_pair, "is non-rotatable")
            errors[1].append(atom_pair)
    
    return errors

def fix_pdbqt(pdbqt_file, bad_bonds_list):
    lines = open(pdbqt_file).readlines()
    rb_section_end_index = lines.index("ROOT\n")
    for i in range(len(lines)):
        if "active torsions:" in lines[i]:
            rb_section_begin_index = i + 2
            break
    rotbond_lines = lines[rb_section_begin_index:rb_section_end_index]
    
    counter = 0

    for i in range(len(rotbond_lines)):
        line = rotbond_lines[i]
        if line[13] != "A":
            continue
            
        atoms_info = line[32:].split()
        first_atom = int(atoms_info[0].split("_")[1])
        second_atom = int(atoms_info[2].split("_")[1])
        
        if (first_atom, second_atom) in bad_bonds_list or (second_atom, first_atom) in bad_bonds_list:
            rotbond_lines[i] = "REMARK       I" + line[14:]
        else:
            counter += 1
            rotbond_num = str(counter)
            rotbond_lines[i] = "REMARK" + (5 - len(rotbond_num)) * " " + rotbond_num + line[11:]
    
    lines[rb_section_begin_index:rb_section_end_index] = rotbond_lines
    
    old_rotbond_num = lines[rb_section_begin_index-2].split()[1]
    torsdof_line_index = lines.index("TORSDOF " + old_rotbond_num + "\n")
    lines[torsdof_line_index] = "TORSDOF " + str(counter) + "\n"
    lines[rb_section_begin_index-2] = "REMARK  " + str(counter) + " active torsions:\n"
    
    open(pdbqt_file, "w").writelines(lines)
    
    return

def check_pdbqt_rot_bonds(pdbqt_file, pdb_file, silent=True, fix_bad_pdbqt=False):
    mol_name = os.path.basename(pdbqt_file).split(".")[0]
        
    rotbonds_pdbqt = get_rotbond_atoms_pdbqt(pdbqt_file)
    
    try:
        rotbonds_pdb = get_rotbond_atoms_pdb(pdb_file)
    except AttributeError:
        raise
    
    bad_rb, missed_rb = rot_bonds_check(rotbonds_pdbqt, rotbonds_pdb, silent)
    if bad_rb:
        if not silent:
            print("Mol:", mol_name)
            print("***")
    elif missed_rb:
        if not silent:
            print("Mol:", mol_name)
            print("***")
    
    if bad_rb and fix_bad_pdbqt:
        fix_pdbqt(pdbqt_file, bad_rb)
    
    return bad_rb, missed_rb

def check_pdbqt_rot_bonds_image(mol_name, pdbqt_dir, pdb_dir, silent=True, fix_bad_pdbqt=False, png_file=None):
    pdbqt_file = pdbqt_dir + os.sep + mol_name + ".pdbqt"
    if not os.path.exists(pdbqt_file):
        return "PDBQT doesn't exists"
    
    pdb_file = pdb_dir + os.sep + mol_name + ".pdb"
    
    try:
        bad_rb, missed_rb = check_pdbqt_rot_bonds(pdbqt_file, pdb_file, silent, fix_bad_pdbqt)
    except (AttributeError, ValueError):
        return "PDB is broken"
    
    if bad_rb or missed_rb:
        return draw.draw_bad_rot_bonds(pdb_file, bad_rb, missed_rb, png_file)

def check_pdbqt_dir_rot_bonds(pdbqt_dir, pdb_dir, silent=True, fix_bad_pdbqt=False, check_images_dir=None):
    
    if not os.path.exists(pdbqt_dir):
        raise Exception("PDBQT folder is not exists")
        
    if not os.path.exists(pdb_dir):
        raise Exception("PDB folder is not exists")
    
    if check_images_dir and not os.path.exists(check_images_dir):
        os.makedirs(check_images_dir)
    
    bad_pdbqt_list = []
    missed_rb_pdbqt_list = []
    
    for pdbqt_file in tqdm(glob.glob(pdbqt_dir + os.sep + "*.pdbqt")):
        mol_name = os.path.basename(pdbqt_file).split(".")[0]

        pdb_file = pdb_dir + os.sep + mol_name + ".pdb"

        try:
            bad_rb, missed_rb = check_pdbqt_rot_bonds(pdbqt_file, pdb_file, silent, fix_bad_pdbqt)
        except (AttributeError, ValueError):
            # AttributeError - PDB не распарсился (Chem.MolFromPDBFile вернул None);
            # ValueError - PDBQT без секции ROOT (get_rotbond_atoms_pdbqt) - оба
            # случая означают брак конвертации одной молекулы из тысяч, не повод
            # ронять весь докинг-прогон целиком. Файл остаётся в pdbqt_dir как
            # есть (не "исправлен" - fix_pdbqt тут не вызывался) и просто дойдёт
            # до Vina как обычно, где уже есть штатная обработка неудач по
            # каждой молекуле (see docking_mpi.py: FAILED в логе, без падения всего прогона).
            if not silent:
                print("PDB/PDBQT %s is broken" % mol_name)
            continue
        
        if bad_rb:
            bad_pdbqt_list.append(mol_name)
        elif missed_rb:
            missed_rb_pdbqt_list.append(mol_name)
            
        if (bad_rb or missed_rb) and check_images_dir:
            image_path = check_images_dir + os.sep + mol_name + ".png"
            draw.draw_bad_rot_bonds(pdb_file, bad_rb, missed_rb, image_path)
    
    return bad_pdbqt_list, missed_rb_pdbqt_list

def filter_pdbqt(pdbqt_dir, silent=True, delete_overnrb_pdbqt=True):
    
    if not os.path.exists(pdbqt_dir):
        raise Exception("PDBQT folder is not exists")
    
    overnrb_pdbqt_list = []
    
    for pdbqt_file in tqdm(glob.glob(pdbqt_dir + os.sep + "*.pdbqt")):
        mol_name = os.path.basename(pdbqt_file).split(".")[0]
        
        pdbqt_stream = open(pdbqt_file)
        while True:
            line = pdbqt_stream.readline()
            if "active torsions:" in line:
                pdbqt_stream.close()
                break
        rotbonds_num = int(line.split()[1])
        
        if rotbonds_num > 20:
            overnrb_pdbqt_list.append(mol_name)
            if not silent:
                print(mol_name, "contains", rotbonds_num, "rotatable bonds")
            if delete_overnrb_pdbqt:
                os.remove(pdbqt_file)
        
    return overnrb_pdbqt_list

def get_top_model(pdbqt_dir, top_model_dir):
    """" Function for getting and saving best MODEL after docking. """
    failures = []

    if not os.path.exists(top_model_dir):
        os.makedirs(top_model_dir)

    for filepath in tqdm(glob.glob(pdbqt_dir + os.sep + '*.pdbqt')):
        filename = os.path.basename(filepath)

        pdbqt_content = open(filepath).read()
        try:
            top_model = pdbqt_content.split('\nENDMDL')[0].split('MODEL 1\n')[1]
            top_model = top_model.rstrip('\x00')
            top_model_file = open(os.path.join(top_model_dir, filename), 'w+')
            top_model_file.write(top_model)
            top_model_file.close()
        except Exception as e:
            failures.append(filename)
    
    return failures
