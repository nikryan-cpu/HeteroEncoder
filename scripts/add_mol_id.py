#!/usr/bin/env python3
"""
Simple utility to add a sequential `Mol ID` column to an existing novel molecules CSV.

Usage examples:
  python scripts/add_mol_id.py outputs/novel_molecules_2026-08-06_22-39.csv
  python scripts/add_mol_id.py input.csv --out output.csv --overwrite

The script writes a new CSV by default named `<input>_with_Mol_ID.csv`.
"""
import argparse
from pathlib import Path
import pandas as pd


def add_mol_id(input_path: Path, out_path: Path = None, column: str = 'Mol ID', overwrite: bool = False, padding: int = 6):
    if not input_path.exists():
        raise SystemExit(f"Error: input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if column in df.columns and not overwrite:
        print(f"Column '{column}' already exists in {input_path}. Use --overwrite to replace it.")
        # Write a copy unchanged so user has a predictable output
        if out_path is None:
            out_path = input_path.with_name(input_path.stem + f"_with_{column.replace(' ', '_')}.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote unchanged file to {out_path}")
        return

    # Assign sequential IDs starting from 1
    df[column] = [f"mol_{i:0{padding}d}" for i in range(1, len(df) + 1)]

    if out_path is None:
        out_path = input_path.with_name(input_path.stem + f"_with_{column.replace(' ', '_')}.csv")

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows with column '{column}' to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Add sequential Mol ID column to a CSV of generated molecules')
    parser.add_argument('input', help='Path to input CSV file')
    parser.add_argument('--out', '-o', help='Path to output CSV file (optional)')
    parser.add_argument('--column', '-c', default='Mol ID', help='Column name to add (default: "Mol ID")')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing column if present')
    parser.add_argument('--padding', type=int, default=6, help='Zero-padding width for numeric IDs (default: 6)')

    args = parser.parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out) if args.out else None

    add_mol_id(input_path, out_path=out_path, column=args.column, overwrite=args.overwrite, padding=args.padding)


if __name__ == '__main__':
    main()
