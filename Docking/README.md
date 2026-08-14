# Молекулярный докинг на кластере SKIF

Докинг сгенерированных молекул к VEGFR2 (PDB **3WZD**) через AutoDock Vina
на суперкомпьютере SKIF. Управление — по UNICORE, локально ставится только
клиент UCC; сам Vina и MGLTools живут на кластере.

Вход — SDF с 3D-конформерами из `../pre-trained/`, выход — `results/dockscore.csv`
с энергиями связывания и `*_docked.sdf.gz` с позами лигандов.

---

## Структура

```text
Docking/
├── settings.py            # ВСЕ настройки: кластер, CPU, пути. Правь только здесь
├── MyDocking.py           # главный запуск: отправка задачи + мониторинг
├── check_results.py       # проверка статуса и докачка (после перезагрузки ПК)
├── tools/
│   └── sync_cluster.py    # пересборка chemplus_local + заливка на кластер
├── chemnotes-main/        # подготовленный белок и config.txt (внешний репозиторий)
├── chemplus-main/         # библиотека chemplus, ИСТОЧНИК ИСТИНЫ для правок
├── chemplus_local/        # ГЕНЕРИРУЕТСЯ sync_cluster.py, руками не править
└── results/               # вывод задач (в .gitignore)
```

### Зачем нужен `chemplus_local`

На кластере есть общая копия chemplus в `/share/bio/data/chemplus`, но править
её нельзя. Поэтому урезанная копия нужных модулей заливается в
`~/projects/docking/chemplus_local`, а задача запускается с
`PYTHONPATH=/home/USERNAME/projects/docking/chemplus_local` — он перекрывает общую.

Отсюда правило: **любая правка идёт в `chemplus-main/src/chemplus/`**, затем
`tools/sync_cluster.py` пересобирает `chemplus_local` и заливает его на кластер.
Прямая правка `chemplus_local` будет затёрта при следующей сборке.

`chemplus-main/src` при этом стоит как editable-пакет (`pip install -e`), так что
`import chemplus` работает из любой директории.

---

## Рабочий цикл

```powershell
conda activate HeteroEncoder
cd Docking

python tools/sync_cluster.py   # только если правил chemplus-main
python MyDocking.py            # отправка задачи + ожидание
```

`sync_cluster.py` делает оба шага сразу — собирает и заливает. Раньше это были
два отдельных скрипта, и пропуск второго молча оставлял на кластере старый код:
задача падала, а traceback указывал на строку, которая локально уже исправлена.
Флаг `--build` собирает без заливки, если нужно просто проверить сборку.

`MyDocking.py` делает по шагам:

1. Проверяет, что UCC отвечает и сайт доступен.
2. Ищет на кластере `pythonsh` (MGLTools) и ставит симлинк `~/bio/mgltools/bin/pythonsh` —
   chemplus ищет его строго по этому пути.
3. Заливает рецептор, конфиг и SDF в `~/projects/docking`, формирует `.u` (JSON задачи)
   и отправляет асинхронно.
4. Каждые 30 секунд опрашивает статус. Ctrl+C безопасен — задача продолжит считаться.
5. По завершении скачивает результаты; при ненулевом exit code печатает хвост `.stderr`.

Если ПК выключили или прервали мониторинг:

```powershell
python check_results.py                                  # выбрать запуск из списка
python check_results.py results/<дата_время>/<id>.job    # конкретный
```

---

## Конфигурация

Все параметры собраны в `settings.py` — больше нигде их менять не надо. Этого
файла нет в репозитории (личный логин и локальный путь к UCC), поэтому перед
первым запуском скопируй [settings.example.py](settings.example.py) в
`settings.py` и впиши свои значения:

| Параметр | Значение | Комментарий |
|---|---|---|
| `UCC_PATH` | `...\ucc-1.3.1\bin\ucc.bat` | локальный клиент UNICORE |
| `SITE_NAME` | `SKIF_GRID_CIS` | либо `SKIF_GEO` |
| `CLUSTER_USER` | `USERNAME` | логин, из него строится `PYTHONPATH` |
| `SERVER_EXECUTABLE` | `/share/bio/data/miniconda3/envs/chem/bin/python` | Python на кластере |
| `SERVER_WORK_DIR` | `~/projects/docking` | рабочая папка на кластере |
| `CPU_COUNT` | `200` | см. таблицу ниже |
| `LOCAL_SDF` | `../outputs/protonated_..._3D_*.sdf` | лиганды из ML-пайплайна |

Адрес заливки и `PYTHONPATH` задачи выводятся из `SITE_NAME` и `CLUSTER_USER`
автоматически, руками их не задают.

Узлы считаются автоматически из `CPU_COUNT`:

| Сайт | CPU/узел | 200 CPU |
|---|---|---|
| `SKIF_GRID_CIS` | 40 | 5 узлов × 40 |
| `SKIF_GEO` | 36 | 6 узлов × 33 |

### Переход на другой кластер

`SKIF_GRID_CIS` и `SKIF_GEO` — **разные машины с разными файловыми системами**
(`gpucis1.basnet.by` и `geo1.basnet.by`). Код, залитый на один сайт, на втором
не появится.

1. Поменять `SITE_NAME` в `settings.py`.
2. Прогнать `python tools/sync_cluster.py` — иначе `chemplus_local` на новом
   сайте просто нет и задача упадёт на импорте.
3. Проверить, что `SERVER_EXECUTABLE` существует на новом сайте: у него своё
   `/share`, и conda-окружение `chem` может лежать иначе.

Путь к MGLTools перепроверять не нужно — `fix_mgltools()` в `MyDocking.py`
перебирает кандидатов при каждом запуске и скажет, если не нашёл.

Область докинга задана в
[chemnotes-main/1. Protein processing/prepared_protein/config.txt](chemnotes-main/1.%20Protein%20processing/prepared_protein/config.txt) —
центр по положению лиганда LEV из кристаллической структуры,
бокс 11.7 × 12.9 × 21.8 Å, `exhaustiveness = 25`.

---

## Результаты

Каждый запуск получает свою папку `results/<ГГГГ-ММ-ДД_ЧЧ-ММ-СС>/`, куда
складывается и вывод UCC, и результаты докинга:

```text
results/
├── 2026-08-13_11-20-16/
│   ├── dockscore.csv                  # энергии связывания — главный результат
│   ├── *_docked.sdf.gz                # позы лигандов
│   ├── <uuid>.stderr                  # traceback задачи — сюда смотреть при падении
│   ├── <uuid>.job                     # ID задачи для check_results.py
│   └── <uuid>.properties, .stdout     # служебный вывод UCC
├── 2026-08-13_11-16-00/
└── _probe/                            # проверка MGLTools, стирается каждый запуск
```

Имена `<uuid>` задаёт сервер при постановке задачи в очередь — переопределить
их с клиента нельзя, поэтому разделение идёт на уровне папок.

Старые папки удаляются автоматически при следующем запуске: остаются последние
`KEEP_RUNS` (по умолчанию 5, меняется в `settings.py`).

---

## Разбор проблем

**`exit code: 1`, ничего больше.** Код возврата ничего не объясняет — причина всегда
в `results/<дата_время>/<uuid>.stderr`. `MyDocking.py` печатает его хвост сам;
для старой задачи открыть файл вручную.

**`IndentationError` / `ImportError` в traceback с путём `chemplus_local/...`.** На кластере
лежит устаревшая или битая копия — правка есть локально, но не доехала.
Лечится одной командой: `python tools/sync_cluster.py`.

**`SocketTimeoutException: Connect timed out`.** UNICORE-сервер недоступен, код ни при чём.
Повторить через несколько минут.

**`Incorrect server path`.** `SERVER_WORK_DIR` не разбирается функцией
`unicore.get_correct_location` — она принимает только `~/...` или `/home/<user>/...`.

**`No matching target system available`.** Сайт не отвечает либо запрошено больше
ресурсов, чем он даёт. Проверить: `ucc.bat list-sites --all`.

**`MGLTools не найден`.** На кластере не нашёлся `pythonsh`. Кандидаты перебираются
в `fix_mgltools()` внутри `MyDocking.py`; рабочий путь —
`/share/bio/mgltools/bin/pythonsh`.

---

## Требования

Локально: Python 3.12 (окружение `HeteroEncoder`), UCC 1.3.1 с настроенным
keystore, editable-установка `chemplus` из `chemplus-main`.

На кластере: conda-окружение `chem`, AutoDock Vina, MGLTools, MPI.
