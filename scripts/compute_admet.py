#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Предсказывает ADMET-эндпоинты (токсичность, метаболизм, всасывание) через admet-ai.

RDKit сам по себе токсичность НЕ предсказывает — он только ищет проблемные
подструктуры по спискам (это делает compute_properties.py). Настоящие эндпоинты
вроде hERG, Ames и DILI считаются ML-моделями, обученными на данных TDC.

Установка (около 1 ГБ моделей, ставится один раз, работает офлайн):

    pip install admet-ai

Запуск:

    python scripts/compute_admet.py outputs/novel_molecules.csv
    python scripts/compute_admet.py outputs/gen_props.csv --out outputs/gen_admet.csv

Считает те же эндпоинты, что лежат в figures/ADMET (hERG, Ames, DILI, CYP3A4,
Caco-2, BBB, HIA), плюс ещё около сорока.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Эндпоинты, по которым у проекта уже есть графики распределений.
KEY_ENDPOINTS = [
    "hERG", "AMES", "DILI", "CYP3A4_Veith", "CYP2D6_Veith",
    "Caco2_Wang", "BBB_Martins", "HIA_Hou", "Bioavailability_Ma",
    "Solubility_AqSolDB", "Clearance_Hepatocyte_AZ", "LD50_Zhu",
]


def load_model():
    try:
        from admet_ai import ADMETModel
    except ImportError:
        sys.exit(
            "[-] Пакет admet-ai не установлен.\n"
            "    pip install admet-ai\n"
            "    (~1 ГБ моделей, скачиваются один раз, дальше работает офлайн)"
        )
    print("[*] Загрузка моделей ADMET-AI...")
    return ADMETModel()


def main():
    p = argparse.ArgumentParser(description="ADMET-предсказания для сгенерированных молекул")
    p.add_argument("input", help="CSV со SMILES")
    p.add_argument("--smiles-col", default=None, help="колонка со SMILES (по умолчанию ищется сама)")
    p.add_argument("--out", default=None, help="куда писать (по умолчанию <input>_admet.csv)")
    p.add_argument("--batch", type=int, default=2000, help="размер батча")
    p.add_argument("--key-only", action="store_true",
                   help="оставить только ключевые эндпоинты вместо всех сорока")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    col = args.smiles_col
    if col is None:
        cand = [c for c in df.columns if c.strip().lower() in ("smiles", "canonical_smiles")]
        if not cand:
            sys.exit(f"[-] Не нашёл колонку со SMILES. Есть: {list(df.columns)}")
        col = cand[0]

    smiles = df[col].astype(str).tolist()
    print(f"[*] Вход: {args.input}  ({len(smiles)} молекул, колонка '{col}')")

    model = load_model()

    chunks = []
    for i in range(0, len(smiles), args.batch):
        batch = smiles[i:i + args.batch]
        print(f"    батч {i // args.batch + 1}/{(len(smiles) - 1) // args.batch + 1}  ({len(batch)} мол)")
        chunks.append(model.predict(smiles=batch))
    preds = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]

    # admet-ai кладёт SMILES в индекс — возвращаем колонкой
    if preds.index.name or preds.index.dtype == object:
        preds = preds.reset_index().rename(columns={"index": "SMILES"})
    if "SMILES" not in preds.columns:
        preds.insert(0, "SMILES", smiles[:len(preds)])

    if args.key_only:
        keep = ["SMILES"] + [c for c in preds.columns if c in KEY_ENDPOINTS]
        missing = [e for e in KEY_ENDPOINTS if e not in preds.columns]
        if missing:
            print(f"[!] Нет в выдаче модели: {', '.join(missing)}")
        preds = preds[keep]

    out = args.out or os.path.splitext(args.input)[0] + "_admet.csv"
    preds.to_csv(out, index=False)
    print(f"[+] Записано: {out}   ({preds.shape[0]} строк, {preds.shape[1]} колонок)")

    print("\n--- ключевые эндпоинты, медианы ---")
    for e in KEY_ENDPOINTS:
        if e in preds.columns:
            print(f"  {e:<26} {preds[e].median():.3f}")
    print("\n  Классификационные эндпоинты (hERG, AMES, DILI, CYP) — вероятность 0..1,")
    print("  выше = хуже. Регрессионные (Caco2, LD50, Solubility) — в своих единицах.")


if __name__ == "__main__":
    main()
