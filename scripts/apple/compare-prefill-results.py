#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


KEY_FIELDS = ["model_label", "quant", "prompt", "depth", "fa", "batch", "ubatch"]


def load_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("summary.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["_summary_path"] = str(path)
                rows.append(row)
    return rows


def compare(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = tuple(row.get(field, "") for field in KEY_FIELDS)
        grouped[key][row.get("variant", "")] = row

    compared: list[dict[str, str]] = []
    for key in sorted(grouped):
        variants = grouped[key]
        baseline = variants.get("baseline")
        if not baseline:
            continue
        try:
            baseline_avg = float(baseline.get("avg_ts", "") or 0.0)
        except ValueError:
            continue
        if baseline_avg <= 0:
            continue

        for variant, row in sorted(variants.items()):
            try:
                avg_ts = float(row.get("avg_ts", "") or 0.0)
            except ValueError:
                continue
            delta_pct = ((avg_ts - baseline_avg) / baseline_avg) * 100.0
            compared.append(
                {
                    **{field: row.get(field, "") for field in KEY_FIELDS},
                    "variant": variant,
                    "avg_ts": row.get("avg_ts", ""),
                    "stddev_ts": row.get("stddev_ts", ""),
                    "backend": row.get("backend", ""),
                    "model_type": row.get("model_type", ""),
                    "summary_path": row.get("_summary_path", ""),
                    "delta_pct_vs_baseline": f"{delta_pct:.2f}",
                }
            )
    return compared


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = KEY_FIELDS + [
        "variant",
        "avg_ts",
        "stddev_ts",
        "delta_pct_vs_baseline",
        "backend",
        "model_type",
        "summary_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    header = (
        "| model | quant | prompt | depth | fa | batch | ubatch | variant | avg_ts | delta % vs baseline |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['quant']} | {row['prompt']} | {row['depth']} | "
            f"{row['fa']} | {row['batch']} | {row['ubatch']} | {row['variant']} | "
            f"{row['avg_ts']} | {row['delta_pct_vs_baseline']} |\n"
        )
    path.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    rows = compare(load_rows(args.input_root))
    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
