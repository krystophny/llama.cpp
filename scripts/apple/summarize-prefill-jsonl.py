#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_name(path: Path) -> dict[str, str]:
    stem = path.stem
    parts = stem.split("__")
    result = {
        "model_label": "",
        "variant": "",
        "quant": "",
        "prompt": "",
        "depth": "",
        "fa": "",
        "batch": "",
        "ubatch": "",
    }
    if len(parts) >= 8:
        result["model_label"] = parts[0]
        result["variant"] = parts[1]
        result["quant"] = parts[2]
        result["prompt"] = parts[3].removeprefix("p")
        result["depth"] = parts[4].removeprefix("d")
        result["fa"] = parts[5].removeprefix("fa")
        result["batch"] = parts[6].removeprefix("b")
        result["ubatch"] = parts[7].removeprefix("ub")
    return result


def load_rows(input_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        name_fields = parse_name(path)
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(
                {
                    **name_fields,
                    "model_type": str(data.get("model_type", "")),
                    "backend": str(data.get("backends", "")),
                    "n_prompt": str(data.get("n_prompt", "")),
                    "n_batch": str(data.get("n_batch", "")),
                    "n_ubatch": str(data.get("n_ubatch", "")),
                    "flash_attn": str(data.get("flash_attn", "")),
                    "avg_ts": str(data.get("avg_ts", "")),
                    "stddev_ts": str(data.get("stddev_ts", "")),
                    "avg_ns": str(data.get("avg_ns", "")),
                    "test_time": str(data.get("test_time", "")),
                    "model_filename": str(data.get("model_filename", "")),
                }
            )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
        "model_label",
        "variant",
        "quant",
        "prompt",
        "depth",
        "fa",
        "batch",
        "ubatch",
        "model_type",
        "backend",
        "n_prompt",
        "n_batch",
        "n_ubatch",
        "flash_attn",
        "avg_ts",
        "stddev_ts",
        "avg_ns",
        "test_time",
        "model_filename",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    header = (
        "| model | variant | quant | prompt | depth | fa | batch | ubatch | avg_ts | stddev_ts | backend | model_type |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['variant']} | {row['quant']} | {row['prompt']} | "
            f"{row['depth']} | {row['fa']} | {row['batch']} | {row['ubatch']} | "
            f"{row['avg_ts']} | {row['stddev_ts']} | {row['backend']} | {row['model_type']} |\n"
        )
    path.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.input_dir)
    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
