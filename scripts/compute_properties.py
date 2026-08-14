#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Считает физхим-дескрипторы, правила лекарствоподобности и структурные алерты
для сгенерированных молекул. Всё локально, только RDKit — новых зависимостей нет.

    python scripts/compute_properties.py outputs/novel_molecules.csv
    python scripts/compute_properties.py outputs/gen.csv --novelty --jobs 8
    python scripts/compute_properties.py outputs/gen.csv --dockscore Docking/results/2026-08-13_11-20-16/dockscore.csv

По умолчанию пишет <входной_файл>_props.csv рядом с исходным.

ВАЖНО: структурные алерты (PAINS, BRENK) — это НЕ предсказание токсичности,
а поиск проблемных подструктур по спискам. Настоящие эндпоинты (hERG, Ames,
DILI) считает scripts/compute_admet.py.
"""

import argparse
import os
import sys

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem import FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

RDLogger.DisableLog('rdApp.*')

# Каркас проекта — хинолин-4-карбоксамид, тот же, что в src/
SCAFFOLD = "O=C(N)c1ccnc2ccccc12"
LENVATINIB = "COc1cc2nccc(Oc3ccc(NC(=O)NC4CC4)c(Cl)c3)c2cc1C(N)=O"

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _build_alert_catalogs():
    """Каталоги структурных алертов RDKit."""
    catalogs = {}
    for name in ["PAINS_A", "PAINS_B", "PAINS_C", "BRENK", "NIH"]:
        try:
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, name))
            catalogs[name] = FilterCatalog.FilterCatalog(params)
        except AttributeError:
            pass
    return catalogs


ALERTS = _build_alert_catalogs()


def _load_sa_scorer():
    """
    SA score живёт в RDKit Contrib, которого в pip-сборке может не быть.
    Возвращает функцию или None.
    """
    import rdkit
    contrib = os.path.join(os.path.dirname(rdkit.__file__), "Contrib", "SA_Score")
    if not os.path.isdir(contrib):
        return None
    sys.path.append(contrib)
    try:
        import sascorer
        return sascorer.calculateScore
    except Exception:
        return None


SA_SCORER = _load_sa_scorer()


def describe(smiles):
    """Все дескрипторы для одной молекулы. Возвращает dict или None."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)

    d = {
        "SMILES": smiles,
        # --- физхим ---
        "MW": round(mw, 2),
        "LogP": round(logp, 3),
        "TPSA": round(tpsa, 2),
        "HBD": hbd,
        "HBA": hba,
        "RotB": rotb,
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "FractionCSP3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "Stereocenters": rdMolDescriptors.CalcNumAtomStereoCenters(mol),
        # --- лекарствоподобность ---
        "QED": round(QED.qed(mol), 4),
        "BertzCT": round(Descriptors.BertzCT(mol), 1),
    }

    # Правила. Считаем как «сколько нарушений», 0 = проходит.
    d["Lipinski_viol"] = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    d["Veber_pass"] = bool(rotb <= 10 and tpsa <= 140)
    d["Egan_pass"] = bool(tpsa <= 131.6 and -1 <= logp <= 5.88)
    d["Ghose_pass"] = bool(160 <= mw <= 480 and -0.4 <= logp <= 5.6
                           and 20 <= mol.GetNumAtoms() <= 70)
    # Lead-like: более строгие рамки для отправной точки оптимизации
    d["LeadLike_pass"] = bool(250 <= mw <= 350 and logp <= 3.5 and rotb <= 7)

    # --- структурные алерты ---
    for name, cat in ALERTS.items():
        entry = cat.GetFirstMatch(mol)
        d[f"Alert_{name}"] = entry.GetDescription() if entry else ""
    d["Alerts_total"] = sum(1 for name in ALERTS if d[f"Alert_{name}"])

    # --- синтетическая доступность ---
    d["SA_Score"] = round(SA_SCORER(mol), 3) if SA_SCORER else None

    # --- каркас ---
    scaf = Chem.MolFromSmarts(SCAFFOLD)
    d["Has_Scaffold"] = bool(scaf and mol.HasSubstructMatch(scaf))
    try:
        d["MurckoScaffold"] = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        d["MurckoScaffold"] = ""

    # --- сходство с лензватинибом ---
    ref = Chem.MolFromSmiles(LENVATINIB)
    if ref is not None:
        d["Tanimoto_Lenvatinib"] = round(DataStructs.TanimotoSimilarity(
            _MORGAN.GetFingerprint(mol), _MORGAN.GetFingerprint(ref)), 4)

    return d


def max_tanimoto_vs_reference(smiles_list, ref_smiles, jobs=1):
    """
    Максимальное сходство каждой молекулы с эталонным набором.
    Низкое значение = молекула не похожа на обучающую выборку (новизна).
    """
    ref_fps = []
    for s in tqdm(ref_smiles, desc="отпечатки эталона", unit="мол"):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            ref_fps.append(_MORGAN.GetFingerprint(m))

    out = []
    for s in tqdm(smiles_list, desc="сходство", unit="мол"):
        m = Chem.MolFromSmiles(str(s))
        if m is None or not ref_fps:
            out.append(None)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(_MORGAN.GetFingerprint(m), ref_fps)
        out.append(round(max(sims), 4))
    return out


def main():
    p = argparse.ArgumentParser(description="Дескрипторы и фильтры для сгенерированных молекул")
    p.add_argument("input", help="CSV со сгенерированными молекулами")
    p.add_argument("--smiles-col", default=None, help="имя колонки со SMILES (по умолчанию ищется само)")
    p.add_argument("--out", default=None, help="куда писать (по умолчанию <input>_props.csv)")
    p.add_argument("--novelty", action="store_true",
                   help="посчитать max Tanimoto к обучающей выборке (медленно)")
    p.add_argument("--ref-sample", type=int, default=10000,
                   help="сколько молекул датасета брать для новизны (0 = все)")
    p.add_argument("--dockscore", default=None,
                   help="приклеить dockscore.csv из папки запуска докинга")
    p.add_argument("--jobs", type=int, default=1, help="параллельных процессов")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    col = args.smiles_col
    if col is None:
        cand = [c for c in df.columns if c.strip().lower() in ("smiles", "canonical_smiles")]
        if not cand:
            sys.exit(f"[-] Не нашёл колонку со SMILES. Есть: {list(df.columns)}")
        col = cand[0]
    print(f"[*] Вход: {args.input}  ({len(df)} строк, колонка '{col}')")

    if SA_SCORER is None:
        print("[!] SA_Score недоступен: нет RDKit Contrib. Колонка будет пустой.")
        print("    Лечится установкой rdkit из conda-forge.")
    print(f"[*] Каталоги алертов: {', '.join(ALERTS) or 'НЕТ'}")

    if args.jobs > 1:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=args.jobs)(
            delayed(describe)(s) for s in tqdm(df[col], desc="дескрипторы", unit="мол"))
    else:
        rows = [describe(s) for s in tqdm(df[col], desc="дескрипторы", unit="мол")]

    bad = sum(1 for r in rows if r is None)
    res = pd.DataFrame([r for r in rows if r is not None])
    print(f"[+] Посчитано: {len(res)}   не распарсилось: {bad}")

    if args.novelty:
        ref = pd.read_csv(config.DATASET)
        ref_col = "CANONICAL_SMILES" if "CANONICAL_SMILES" in ref.columns else "SMILES"
        if args.ref_sample and len(ref) > args.ref_sample:
            ref = ref.sample(args.ref_sample, random_state=0)
            print(f"[*] Новизна считается по подвыборке датасета: {len(ref)} молекул")
        res["MaxTanimoto_vs_TrainSet"] = max_tanimoto_vs_reference(
            res["SMILES"], ref[ref_col], jobs=args.jobs)

    if args.dockscore:
        if not os.path.exists(args.dockscore):
            print(f"[!] {args.dockscore} не найден — пропускаю склейку")
        elif os.path.getsize(args.dockscore) == 0:
            print(f"[!] {args.dockscore} пустой (докинг ещё не досчитал) — пропускаю склейку")
        else:
            ds = pd.read_csv(args.dockscore)
            key = "Mol ID" if "Mol ID" in df.columns else None
            if key is None:
                print("[!] В исходном CSV нет колонки 'Mol ID' — по чему склеивать с dockscore, неясно.")
                print("    Сначала прогони scripts/add_mol_id.py, затем генерируй SDF с теми же ID.")
            else:
                res[key] = df[key].values[:len(res)]
                res = res.merge(ds, left_on=key, right_on="Name", how="left")
                print(f"[+] Приклеен dockscore: {ds.shape[1]} колонок, "
                      f"совпало {res['E_vina, kcal/mol'].notna().sum()} молекул")

    out = args.out or os.path.splitext(args.input)[0] + "_props.csv"
    res.to_csv(out, index=False)
    print(f"[+] Записано: {out}   ({res.shape[0]} строк, {res.shape[1]} колонок)")

    # Короткая сводка, чтобы сразу видеть, что получилось
    print("\n--- сводка ---")
    print(f"  Lipinski без нарушений : {(res['Lipinski_viol'] == 0).sum():>6}  ({100*(res['Lipinski_viol']==0).mean():.1f}%)")
    print(f"  Veber проходит         : {res['Veber_pass'].sum():>6}  ({100*res['Veber_pass'].mean():.1f}%)")
    print(f"  Без структурных алертов: {(res['Alerts_total'] == 0).sum():>6}  ({100*(res['Alerts_total']==0).mean():.1f}%)")
    print(f"  QED медиана            : {res['QED'].median():.3f}")
    if res["SA_Score"].notna().any():
        print(f"  SA_Score медиана       : {res['SA_Score'].median():.2f}  (1 просто — 10 сложно)")


if __name__ == "__main__":
    main()
