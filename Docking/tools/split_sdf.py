"""
Делит SDF-файл на несколько частей пропорционально весам (используется
MyDocking_parallel.py для разбиения лигандов между несколькими сайтами
по заданному числу CPU на каждом).

Читает и пишет построчно, без парсинга через RDKit — молекулы просто
переносятся байт-в-байт по границам записей ($$$$), поэтому подходит для
файлов в сотни МБ - единицы ГБ без роста потребления памяти.

    python tools/split_sdf.py ligands.sdf out_a.sdf:600 out_b.sdf:1200
"""

import os
import sys


def split_sdf_by_weights(input_path, output_paths, weights):
    """
    Разбивает input_path на len(output_paths) SDF-файлов взвешенным
    round-robin: на каждую запись выбирается тот выходной файл, который
    сейчас больше всего "недополучил" относительно своего веса
    (assigned[j] / weights[j] минимален). Один проход по входному файлу,
    точное количество записей заранее знать не нужно — по мере чтения
    доли сходятся к заданным весам.

    Параметры:
        input_path (str): исходный SDF (может быть очень большим)
        output_paths (list[str]): пути выходных SDF, по одному на часть
        weights (list[float]): веса частей, тем же порядком что output_paths
                              (например, число CPU на каждом сайте)

    Возвращает:
        list[int]: число записей, попавших в каждый выходной файл
    """
    if len(output_paths) != len(weights):
        raise ValueError("output_paths и weights должны быть одной длины")
    if any(w <= 0 for w in weights):
        raise ValueError("Все веса должны быть положительными")

    outputs = [open(p, "w", encoding="utf-8", newline="") for p in output_paths]
    assigned = [0] * len(outputs)
    try:
        with open(input_path, encoding="utf-8", errors="replace") as inf:
            buf = []
            for line in inf:
                buf.append(line)
                if line.rstrip("\r\n") == "$$$$":
                    j = min(range(len(outputs)), key=lambda k: assigned[k] / weights[k])
                    outputs[j].writelines(buf)
                    assigned[j] += 1
                    buf = []
            if buf:
                # Файл без завершающего $$$$ на последней записи — не должно
                # происходить для валидного SDF, но не молча терять хвост.
                j = min(range(len(outputs)), key=lambda k: assigned[k] / weights[k])
                outputs[j].writelines(buf)
                assigned[j] += 1
    finally:
        for f in outputs:
            f.close()

    return assigned


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python split_sdf.py вход.sdf выход1.sdf:вес1 выход2.sdf:вес2 ...")
        sys.exit(1)

    input_path = sys.argv[1]
    output_paths = []
    weights = []
    for spec in sys.argv[2:]:
        path, _, weight = spec.rpartition(":")
        if not path:
            print(f"Некорректный аргумент (нужно путь:вес): {spec}")
            sys.exit(1)
        output_paths.append(path)
        weights.append(float(weight))

    print(f"Делим {input_path} на {len(output_paths)} частей, веса: {weights}")
    counts = split_sdf_by_weights(input_path, output_paths, weights)
    for path, count in zip(output_paths, counts):
        print(f"  {path}: {count} молекул")
