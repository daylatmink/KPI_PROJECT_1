#!/usr/bin/env python3
"""
Extract project-specific issue_links.csv from a global links file.
Supports CSV and JSONL inputs.
"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def extract_from_csv(input_path: Path, output_path: Path, project_key: str, chunk_size: int):
    prefix = f"{project_key}-"
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    wrote_header = False
    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        if "from_issue_key" not in chunk.columns or "to_issue_key" not in chunk.columns:
            raise ValueError("CSV must have from_issue_key and to_issue_key columns.")

        mask = chunk["from_issue_key"].astype(str).str.startswith(prefix) | \
               chunk["to_issue_key"].astype(str).str.startswith(prefix)
        filtered = chunk[mask]

        if filtered.empty:
            continue

        mode = "w" if not wrote_header else "a"
        filtered.to_csv(output_path, index=False, mode=mode, header=not wrote_header)
        wrote_header = True

    if not wrote_header:
        output_path.write_text("from_issue_key,to_issue_key,link_type,raw_link_type,direction\n", encoding="utf-8")


def extract_from_jsonl(input_path: Path, output_path: Path, project_key: str):
    prefix = f"{project_key}-"
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["from_issue_key", "to_issue_key", "link_type", "raw_link_type", "direction"]
    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            from_key = str(row.get("from_issue_key", ""))
            to_key = str(row.get("to_issue_key", ""))
            if not (from_key.startswith(prefix) or to_key.startswith(prefix)):
                continue
            writer.writerow(
                {
                    "from_issue_key": row.get("from_issue_key", ""),
                    "to_issue_key": row.get("to_issue_key", ""),
                    "link_type": row.get("link_type", ""),
                    "raw_link_type": row.get("raw_link_type", ""),
                    "direction": row.get("direction", ""),
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Extract project issue links from a global links file.")
    parser.add_argument("--input", required=True, help="Path to global links file (CSV or JSONL).")
    parser.add_argument("--project-key", required=True, help="Project key (e.g., ZOOKEEPER).")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--chunk-size", type=int, default=200000, help="CSV chunk size.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    project_key = args.project_key.strip().upper()
    output_path = Path(args.output) if args.output else Path("projects") / project_key / "issue_links.csv"

    if input_path.suffix.lower() == ".csv":
        extract_from_csv(input_path, output_path, project_key, args.chunk_size)
    elif input_path.suffix.lower() in {".jsonl", ".ndjson"}:
        extract_from_jsonl(input_path, output_path, project_key)
    else:
        raise ValueError("Unsupported input format. Use .csv or .jsonl/.ndjson.")

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
