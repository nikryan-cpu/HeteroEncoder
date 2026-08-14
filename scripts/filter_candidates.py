#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Сводит вместе дескрипторы, ADMET и энергии докинга, применяет фильтры
и выдаёт отранжированный шорт-лист кандидатов.

    # только физхим
    python scripts/filter_candidates.py outputs/gen_props.csv

    # всё вместе
    python scripts/filter_candidates.py outputs/gen_props.csv \
        --admet outputs/gen_admet.csv \
        --dockscore Docking/results/2026-08-13_11-20-16/dockscore.csv \
        --top 100

Пороги задаются флагами, дефолты — общепринятые для hit-to-lead. Каждый
фильтр можно отключить, поставив заведомо широкий порог.

Скрипт НЕ решает за тебя: он печатает, сколько молекул отсеивает каждый
критерий по отдельности, чтобы было видно, какой из них реально режет выборку.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_filters(p):
    g = p.add_argument_group("пороги физхим")
    g.add_argument("--max-mw", type=float, default=500.0)
    g.add_argument("--max-logp", type=float, default=5.0)
    g.add_argument("--min-logp", type=float, default=-1.0)
    g.add_argument("--max-tpsa", type=float, default=140.0)
    g.add_argument("--max-rotb", type=int, default=10)
    g.add_argument("--min-qed", type=float, default=0.0)
    g.add_argument("--max-sa", type=float, default=6.0, help="синтетическая доступность, 1 просто — 10 сложно")
    g.add_argument("--allow-alerts", action="store_true", help="не отсеивать по структурным алертам")
    g.add_argument("--max-tanimoto-train", type=float, default=1.0,
                   help="отсечь слишком похожие на обучающую выборку")

    a = p.add_argument_group("пороги ADMET (нужен --admet)")
    a.add_argument("--max-herg", type=float, default=0.5, help="вероятность блокады hERG")
    a.add_argument("--max-ames", type=float, default=0.5, help="вероятность мутагенности")
    a.add_argument("--max-dili", type=float, default=0.5, help="вероятность гепатотоксичности")

    d = p.add_argument_group("докинг (нужен --dockscore)")
    d.add_argument("--max-energy", type=float, default=0.0,
                   help="верхняя граница E_vina, ккал/моль (например -8)")


def run_filter(df, max_mw=500.0, min_logp=-1.0, max_logp=5.0, max_tpsa=140.0,
              max_rotb=10, min_qed=0.0, max_sa=6.0, allow_alerts=False,
              max_tanimoto_train=1.0, max_herg=0.5, max_ames=0.5, max_dili=0.5,
              max_energy=0.0, top=0, log=print):
    """
    Ядро фильтрации — используется и CLI (main), и GUI (gui/runner.py), чтобы
    не дублировать логику. df уже должен содержать нужные колонки (см. merge
    ADMET/dockscore в main() ниже — GUI делает то же самое перед вызовом).

    Возвращает (passed_df, checks: dict[str, pd.Series[bool]]).
    """
    checks = {}
    if "MW" in df:      checks["MW"] = df["MW"] <= max_mw
    if "LogP" in df:    checks["LogP"] = df["LogP"].between(min_logp, max_logp)
    if "TPSA" in df:    checks["TPSA"] = df["TPSA"] <= max_tpsa
    if "RotB" in df:    checks["RotB"] = df["RotB"] <= max_rotb
    if "QED" in df:     checks["QED"] = df["QED"] >= min_qed
    if "SA_Score" in df and df["SA_Score"].notna().any():
        checks["SA_Score"] = df["SA_Score"].fillna(0) <= max_sa
    if "Alerts_total" in df and not allow_alerts:
        checks["без алертов"] = df["Alerts_total"] == 0
    if "MaxTanimoto_vs_TrainSet" in df and max_tanimoto_train < 1.0:
        checks["новизна"] = df["MaxTanimoto_vs_TrainSet"] <= max_tanimoto_train
    for col, thr, name in [("hERG", max_herg, "hERG"),
                           ("AMES", max_ames, "AMES"),
                           ("DILI", max_dili, "DILI")]:
        if col in df:
            checks[name] = df[col] <= thr
    if "E_vina, kcal/mol" in df and max_energy < 0:
        checks["E_vina"] = df["E_vina, kcal/mol"] <= max_energy

    if not checks:
        raise ValueError("Ни один критерий не применим — проверь колонки во входном файле")

    log("\n--- вклад каждого критерия по отдельности ---")
    for name, mask in checks.items():
        n = int(mask.sum())
        log(f"  {name:<14} проходит {n:>6} из {len(df)}  ({100*n/len(df):5.1f}%)")

    combined = pd.Series(True, index=df.index)
    for mask in checks.values():
        combined &= mask.fillna(False)
    passed = df[combined].copy()
    log(f"\n[+] Проходят ВСЕ критерии: {len(passed)} из {len(df)}  ({100*len(passed)/len(df):.1f}%)")

    if passed.empty:
        log("[!] Не осталось ни одной молекулы. Ослабь пороги — таблица выше показывает,")
        log("    какой критерий режет сильнее всего.")
        return passed, checks

    # Ранжирование: докинг важнее всего, дальше QED
    if "E_vina, kcal/mol" in passed and passed["E_vina, kcal/mol"].notna().any():
        passed = passed.sort_values("E_vina, kcal/mol")
        log("[*] Отсортировано по E_vina (лучшие сверху)")
    elif "QED" in passed:
        passed = passed.sort_values("QED", ascending=False)
        log("[*] Отсортировано по QED (докинга нет)")

    if top:
        passed = passed.head(top)

    return passed, checks


def main():
    p = argparse.ArgumentParser(description="Фильтрация и ранжирование кандидатов")
    p.add_argument("props", help="CSV от compute_properties.py")
    p.add_argument("--admet", default=None, help="CSV от compute_admet.py")
    p.add_argument("--dockscore", default=None, help="dockscore.csv из папки запуска докинга")
    p.add_argument("--out", default=None, help="куда писать шорт-лист")
    p.add_argument("--top", type=int, default=0, help="оставить только N лучших (0 = все прошедшие)")
    add_filters(p)
    args = p.parse_args()

    df = pd.read_csv(args.props)
    print(f"[*] Дескрипторы: {len(df)} молекул")

    if args.admet:
        ad = pd.read_csv(args.admet)
        before = df.shape[1]
        df = df.merge(ad, on="SMILES", how="left", suffixes=("", "_admet"))
        print(f"[+] ADMET приклеен: +{df.shape[1] - before} колонок")

    if args.dockscore:
        if os.path.getsize(args.dockscore) == 0:
            print(f"[!] {args.dockscore} пустой — докинг ещё не досчитал")
        else:
            ds = pd.read_csv(args.dockscore)
            key = "Mol ID" if "Mol ID" in df.columns else None
            if key:
                df = df.merge(ds, left_on=key, right_on="Name", how="left")
                print(f"[+] Докинг приклеен: {df['E_vina, kcal/mol'].notna().sum()} совпадений")
            else:
                print("[!] Нет колонки 'Mol ID' — склеить с dockscore не по чему")

    try:
        passed, _checks = run_filter(
            df, max_mw=args.max_mw, min_logp=args.min_logp, max_logp=args.max_logp,
            max_tpsa=args.max_tpsa, max_rotb=args.max_rotb, min_qed=args.min_qed,
            max_sa=args.max_sa, allow_alerts=args.allow_alerts,
            max_tanimoto_train=args.max_tanimoto_train, max_herg=args.max_herg,
            max_ames=args.max_ames, max_dili=args.max_dili, max_energy=args.max_energy,
            top=args.top,
        )
    except ValueError as e:
        sys.exit(f"[-] {e}")

    if passed.empty:
        return

    out = args.out or os.path.splitext(args.props)[0] + "_shortlist.csv"
    passed.to_csv(out, index=False)
    print(f"[+] Записано: {out}   ({len(passed)} молекул)")


if __name__ == "__main__":
    main()
