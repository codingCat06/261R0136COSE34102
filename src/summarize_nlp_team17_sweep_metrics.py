#!/usr/bin/env python3
"""Summarize NLP-TEAM 17 coefficient-sweep metric JSON files.

Reads output/metrics/nlp_team17_sweep/*_metrics.json and
writes a flat group-level table with off/on/delta values for every sweep.
This is CPU-only and does not touch model or representation caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


GROUPS = ["jb_corr", "jb_blocked", "harm_refuse", "or_corr", "benign_ans"]
METRICS = ["refuse", "asr", "pure_or"]


def parse_args():
    here = Path(__file__).resolve()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=here.parents[1])
    p.add_argument("--metrics-dir", type=Path, default=Path("output/metrics/nlp_team17_sweep"))
    p.add_argument("--out-csv", type=Path, default=Path("output/metrics/nlp_team17_sweep_group_rates.csv"))
    p.add_argument("--out-json", type=Path, default=Path("output/metrics/nlp_team17_sweep_group_rates.json"))
    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def coef_from_tag_value(text: str) -> float | None:
    try:
        return float(text.replace("m", "-").replace("p", "."))
    except ValueError:
        return None


def parse_tag(tag: str):
    m = re.fullmatch(r"jb([^_]+)_or([^_]+)_thr([^_]+)", tag)
    if not m:
        return None, None, None
    return tuple(coef_from_tag_value(x) for x in m.groups())


def tag_from_file(path: Path) -> str:
    name = path.name
    suffix = "_metrics.json"
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def row_for_group(tag: str, obj: dict, group: str) -> dict:
    cfg = obj.get("config", {}) or {}
    jb_tag, or_tag, thr_tag = parse_tag(tag)
    jb = cfg.get("JB_SCALE", cfg.get("jb_scale", jb_tag))
    or_s = cfg.get("OR_SCALE", cfg.get("or_scale", or_tag))
    thr = cfg.get("ABSTAIN_THRESHOLD", cfg.get("abstain_threshold", thr_tag))
    off = obj.get("baseline_off", {}).get(group, {}) or {}
    on = obj.get("adapter_on", {}).get(group, {}) or {}
    diff = obj.get("aggregate_diff", {}).get(group, {}) or {}
    row = {
        "sweep": tag,
        "jb_scale": jb,
        "or_scale": or_s,
        "abstain_threshold": thr,
        "group": group,
        "n": off.get("n", on.get("n")),
    }
    for metric in METRICS:
        off_v = off.get(metric)
        on_v = on.get(metric)
        delta = diff.get(f"{metric}_delta_pp")
        if delta is None and off_v is not None and on_v is not None:
            delta = (on_v - off_v) * 100
        row[f"{metric}_off"] = off_v
        row[f"{metric}_on"] = on_v
        row[f"{metric}_delta_pp"] = delta
    return row


def sort_row(row: dict):
    def val(x):
        return float(x) if x is not None else 999.0

    return (val(row["jb_scale"]), val(row["or_scale"]), val(row["abstain_threshold"]), row["group"])


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sweep",
        "jb_scale",
        "or_scale",
        "abstain_threshold",
        "group",
        "n",
        "refuse_off",
        "refuse_on",
        "refuse_delta_pp",
        "asr_off",
        "asr_on",
        "asr_delta_pp",
        "pure_or_off",
        "pure_or_on",
        "pure_or_delta_pp",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    args = parse_args()
    root = args.root.resolve()
    metrics_dir = resolve(root, args.metrics_dir)
    if not metrics_dir.exists():
        raise SystemExit(f"Missing metrics dir: {metrics_dir}")
    files = sorted(metrics_dir.glob("*_metrics.json"))
    if not files:
        raise SystemExit(f"No *_metrics.json files found under {metrics_dir}")

    rows = []
    for path in files:
        tag = tag_from_file(path)
        obj = load_json(path)
        for group in GROUPS:
            rows.append(row_for_group(tag, obj, group))
    rows = sorted(rows, key=sort_row)

    csv_path = write_csv(resolve(root, args.out_csv), rows)
    json_path = resolve(root, args.out_json)
    save_json(
        json_path,
        {
            "schema": "nlp_team17_sweep_group_rates_v1",
            "metrics_dir": str(metrics_dir),
            "groups": GROUPS,
            "metrics": METRICS,
            "rows": rows,
        },
    )

    print(f"Read metric files: {len(files)}")
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    print()
    print(
        f"{'sweep':24s} {'group':12s} {'refuse':>17s} {'refΔ':>8s} "
        f"{'ASR':>17s} {'ASRΔ':>8s} {'pure_OR':>17s} {'ORΔ':>8s}"
    )
    for row in rows:
        fmt_rate = lambda x: "  nan" if x is None else f"{100 * float(x):5.1f}"
        fmt_pp = lambda x: "   nan" if x is None else f"{float(x):+7.2f}"
        print(
            f"{row['sweep']:24s} {row['group']:12s} "
            f"{fmt_rate(row['refuse_off'])}->{fmt_rate(row['refuse_on'])}% {fmt_pp(row['refuse_delta_pp'])} "
            f"{fmt_rate(row['asr_off'])}->{fmt_rate(row['asr_on'])}% {fmt_pp(row['asr_delta_pp'])} "
            f"{fmt_rate(row['pure_or_off'])}->{fmt_rate(row['pure_or_on'])}% {fmt_pp(row['pure_or_delta_pp'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
