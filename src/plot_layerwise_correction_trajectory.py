#!/usr/bin/env python3
"""Layer-wise inference correction trajectory analysis.

CPU-only analysis for the final NLP-TEAM 17 coefficient setting. It reads existing
adapter-off representations, adapter-on representations/capture, training
targets, and the analysis-derived correction subspace. It does not load the
model and does not run generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


GROUPS = ["jb_corr", "jb_blocked", "harm_refuse", "or_corr", "benign_ans"]
ACT_LAYERS_DEFAULT = [9, 12, 15, 18, 21]
LAYER_WEIGHTS_DEFAULT = {9: 0.5, 12: 1.0, 15: 1.5, 18: 2.0, 21: 2.0}
FINAL_TAG = "jb1p5_or2p5_thr0p2"
COLORS = {
    "harm_refuse": "#8B1E1E",
    "jb_blocked": "#C26A00",
    "jb_corr": "#E7A95C",
    "or_corr": "#2A9D8F",
    "benign_ans": "#1D4E89",
    "jb_corr_to_refused": "#E7A95C",
    "jb_corr_still_answered": "#E7A95C",
    "or_corr_to_answered": "#2A9D8F",
    "or_corr_still_refused": "#2A9D8F",
}
LINE_STYLES = {
    "jb_corr_to_refused": ("-", "o"),
    "jb_corr_still_answered": ("--", "s"),
    "or_corr_to_answered": ("-", "o"),
    "or_corr_still_refused": ("--", "s"),
}

torch = None
plt = None


def parse_args():
    here = Path(__file__).resolve()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=here.parents[1])
    p.add_argument("--config-tag", default=FINAL_TAG)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--out-dir", type=Path, default=Path("output/figures"))
    p.add_argument("--eval-off-json", type=Path, default=None)
    p.add_argument("--eval-emb-off", type=Path, default=None)
    p.add_argument("--eval-on-json", type=Path, default=None)
    p.add_argument("--eval-emb-on", type=Path, default=None)
    p.add_argument("--capture-detailed", type=Path, default=None)
    p.add_argument("--directions", type=Path, default=None)
    p.add_argument("--targets", type=Path, default=None)
    p.add_argument("--subspace", type=Path, default=None)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--metrics", type=Path, default=None)
    return p.parse_args()


def import_deps(root: Path):
    global torch, plt
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch as torch_mod

    mpl_dir = root / "output/figures/.matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_mod

    torch = torch_mod
    plt = plt_mod
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def resolve(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else root / path


def first_existing(root: Path, name: str, explicit: Path | None, candidates: list[str], required=True) -> Path | None:
    if explicit is not None:
        path = resolve(root, explicit)
        if path and path.exists():
            return path
        if required:
            raise FileNotFoundError(f"Missing required {name}: {path}")
        return path
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    wanted = {Path(c).name for c in candidates}
    hits = [p for p in root.rglob("*") if p.is_file() and p.name in wanted and ".git" not in p.parts]
    if hits:
        return sorted(hits, key=lambda p: (len(p.parts), str(p)))[0]
    if required:
        raise FileNotFoundError(f"Missing required {name}. Tried:\n" + "\n".join(f"  {c}" for c in candidates))
    return None


def load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def sha16(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def layer_get(mapping: dict, layer: int):
    if layer in mapping:
        return mapping[layer]
    key = str(layer)
    if key in mapping:
        return mapping[key]
    raise KeyError(f"missing layer {layer}")


def user_mean_off(sample: dict, layer: int):
    h = layer_get(sample["span_h"], layer).float()
    return h if h.dim() == 1 else h.mean(0)


def final_off(sample: dict, layer: int):
    return layer_get(sample["last_h"], layer).float()


def user_mean_on(sample: dict, layer: int):
    if "span_h_corr" in sample:
        return layer_get(sample["span_h_corr"], layer).float()
    h = layer_get(sample["span_h"], layer).float()
    return h if h.dim() == 1 else h.mean(0)


def final_on(sample: dict, layer: int):
    if "last_h_corr" in sample:
        return layer_get(sample["last_h_corr"], layer).float()
    return layer_get(sample["last_h"], layer).float()


def q_basis_from_cache(subspace_cache: dict):
    return subspace_cache["Q"] if isinstance(subspace_cache, dict) and "Q" in subspace_cache else subspace_cache


def weighted_layers(config: dict, layers: list[int]):
    raw = config.get("LAYER_WEIGHTS", LAYER_WEIGHTS_DEFAULT) if isinstance(config, dict) else LAYER_WEIGHTS_DEFAULT
    vals = []
    for layer in layers:
        vals.append(float(raw.get(layer, raw.get(str(layer), 1.0))) if isinstance(raw, dict) else 1.0)
    arr = np.asarray(vals, dtype=np.float64)
    if not np.isfinite(arr).all() or arr.sum() <= 0:
        arr = np.ones(len(layers), dtype=np.float64)
    return arr / arr.sum()


def eval_off_records(eval_off: dict, eval_emb_off: list[dict]):
    meta = eval_off.get("eval_samples")
    if not isinstance(meta, list):
        raise RuntimeError("eval_off.json missing eval_samples.")
    if len(meta) != len(eval_emb_off):
        raise RuntimeError(f"eval_off/eval_emb_off length mismatch: {len(meta)} vs {len(eval_emb_off)}")
    out = []
    for i, (m, rec) in enumerate(zip(meta, eval_emb_off)):
        r = dict(rec)
        r["sample_id"] = m.get("id", rec.get("id", i))
        r["idx"] = m.get("idx", rec.get("idx", i))
        r["group"] = m.get("group", rec.get("group"))
        out.append(r)
    return out


def on_records_from_capture(capture_obj):
    if isinstance(capture_obj, dict) and "records" in capture_obj:
        records = capture_obj["records"]
        source = "eval_repr_on_capture_detailed"
    elif isinstance(capture_obj, list):
        records = capture_obj
        source = "eval_repr_on_capture_list"
    else:
        raise RuntimeError("Unsupported capture format.")
    return records, source


def on_records_from_eval_emb(eval_samples: list[dict], eval_emb_on: list[dict]):
    if len(eval_samples) != len(eval_emb_on):
        raise RuntimeError(f"eval_emb_on length mismatch: {len(eval_emb_on)} vs eval samples {len(eval_samples)}")
    records = []
    for i, (off, on) in enumerate(zip(eval_samples, eval_emb_on)):
        r = dict(on)
        r["sample_id"] = off["sample_id"]
        r["idx"] = on.get("idx", off.get("idx", i))
        r["group"] = on.get("group", off.get("group"))
        records.append(r)
    return records, "eval_emb_on_fallback"


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "refuse", "refused", "1"}:
            return True
        if low in {"false", "no", "answer", "answered", "0"}:
            return False
    return None


def wg_refused(item):
    if isinstance(item, dict):
        for key in ("refusal", "refuse", "is_refusal", "refused"):
            if key in item:
                return coerce_bool(item[key])
    return coerce_bool(item)


def attach_eval_on(records: list[dict], eval_on: dict):
    meta = eval_on.get("eval_samples", [])
    wg = eval_on.get("wg", [])
    by_id = {}
    for i, item in enumerate(wg):
        sid = meta[i].get("id", i) if i < len(meta) and isinstance(meta[i], dict) else i
        by_id[sid] = wg_refused(item)
    matched = labeled = 0
    for rec in records:
        sid = rec.get("sample_id", rec.get("id"))
        if sid in by_id:
            matched += 1
        rec["adapter_on_refused"] = by_id.get(sid)
        if rec["adapter_on_refused"] is not None:
            labeled += 1
    if matched != len(records):
        print(f"WARNING: transition-label join matched {matched}/{len(records)} records")
    if labeled == 0:
        print("WARNING: no adapter-on WildGuard refusal labels parsed.")
    return {"matched": matched, "labeled": labeled}


def transition_name(group: str, refused):
    if group == "jb_corr":
        if refused is None:
            return "jb_corr_unknown"
        return "jb_corr_to_refused" if refused is True else "jb_corr_still_answered"
    if group == "or_corr":
        if refused is None:
            return "or_corr_unknown"
        return "or_corr_still_refused" if refused is True else "or_corr_to_answered"
    return group


def q_coords_user(sample: dict, q_basis: dict, layer: int, source: str):
    q = layer_get(q_basis, layer).float()
    h = user_mean_off(sample, layer) if source == "off" else user_mean_on(sample, layer)
    return h @ q.T


def q_std(targets: dict, group: str, layer: int):
    return torch.clamp(torch.as_tensor(layer_get(targets["g5_std"][group], layer), dtype=torch.float32), min=1.0)


def jb_layer_distance(sample: dict, targets: dict, q_basis: dict, layer: int, source: str):
    coord = q_coords_user(sample, q_basis, layer, source)
    tgt = torch.as_tensor(layer_get(targets["g5_mean"]["harm_refuse"], layer), dtype=torch.float32)
    std = q_std(targets, "harm_refuse", layer)
    return float(torch.sqrt(torch.clamp((((coord - tgt) / std) ** 2).mean(), min=0.0)))


def or_layer_distance(sample: dict, dirs: dict, targets: dict, layer: int, source: str):
    vr = layer_get(dirs["v_refuse"], layer).float()
    h = final_off(sample, layer) if source == "off" else final_on(sample, layer)
    target = layer_get(targets["or_vr_answered"], layer)
    return abs((float(h @ vr) - float(target["mean"])) / max(float(target["std"]), 1.0))


def disp_layer_norm(off: dict, on: dict, targets: dict, q_basis: dict, layer: int, group: str):
    off_c = q_coords_user(off, q_basis, layer, "off")
    on_c = q_coords_user(on, q_basis, layer, "on")
    return float((((on_c - off_c) / q_std(targets, group, layer)) ** 2).mean().sqrt())


def bootstrap_mean_ci(values, rng, n_boot):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return math.nan, math.nan, math.nan
    if vals.size == 1:
        x = float(vals[0])
        return x, x, x
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def split_label(name: str):
    display = {
        "jb_corr_to_refused": "jb_corr\nrefused",
        "jb_corr_still_answered": "jb_corr\nanswered",
        "or_corr_to_answered": "or_corr\nanswered",
        "or_corr_still_refused": "or_corr\nrefused",
    }
    if name in display:
        return display[name]
    parts = name.split("_")
    if len(parts) <= 2:
        return name
    return "_".join(parts[:2]) + "\n" + "_".join(parts[2:])


def line_panel(ax, summary_rows, series_names, *, title, ylabel):
    for name in series_names:
        rows = [r for r in summary_rows if r["series"] == name and r["stat"] == "layer_mean"]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: int(r["layer"]))
        x = [int(r["layer"]) for r in rows]
        y = [r["mean"] for r in rows]
        lo = [r["ci_low"] for r in rows]
        hi = [r["ci_high"] for r in rows]
        color = COLORS.get(name, COLORS.get(name.split("_to_")[0], "0.3"))
        linestyle, marker = LINE_STYLES.get(name, ("-", "o"))
        label = f"{split_label(name)}\n(n={rows[0]['n']})"
        ax.plot(x, y, marker=marker, linestyle=linestyle, lw=1.6, color=color, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Intervention layer")
    ax.set_ylabel(ylabel)
    ax.grid(color="0.92", lw=0.6)
    ax.legend(frameon=False, loc="best")


def count_by_transition(records):
    out = {}
    for rec in records:
        tr = transition_name(rec["group"], rec.get("adapter_on_refused"))
        out[tr] = out.get(tr, 0) + 1
    return out


def main():
    args = parse_args()
    root = args.root.resolve()
    import_deps(root)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.config_tag
    candidates = {
        "eval_off_json": ["cache/unified_vseries_3b/eval_off.json", "cache/v07/eval_off.json"],
        "eval_emb_off": ["cache/unified_vseries_3b/eval_emb_off_v85.pt", "cache/v07/eval_emb_off.pt"],
        "eval_on_json": [f"cache/nlp_team17/sweep/{tag}/eval_on.json", f"sweep/{tag}/eval_on.json"],
        "eval_emb_on": [f"cache/nlp_team17/sweep/{tag}/eval_emb_on.pt", f"sweep/{tag}/eval_emb_on.pt"],
        "capture_detailed": [
            f"cache/nlp_team17/sweep/{tag}/eval_repr_on_capture_detailed.pt",
            f"sweep/{tag}/eval_repr_on_capture_detailed.pt",
        ],
        "directions": ["cache/v85_onepass/directions.pt", "cache/nlp_team17/directions.pt"],
        "targets": ["cache/v85_onepass/targets.pt", "cache/nlp_team17/targets.pt"],
        "subspace": ["cache/v85_onepass/subspace_target.pt", "cache/nlp_team17/subspace_target.pt"],
        "adapter": ["output/adapter_v85_onepass.pt", "output/adapter_nlp_team17.pt"],
        "metrics": [f"nlp_team17_sweep/{tag}_metrics.json", f"output/metrics/nlp_team17_sweep/{tag}_metrics.json"],
    }

    paths = {
        "eval_off_json": first_existing(root, "eval_off_json", args.eval_off_json, candidates["eval_off_json"]),
        "eval_emb_off": first_existing(root, "eval_emb_off", args.eval_emb_off, candidates["eval_emb_off"]),
        "eval_on_json": first_existing(root, "eval_on_json", args.eval_on_json, candidates["eval_on_json"]),
        "directions": first_existing(root, "directions", args.directions, candidates["directions"]),
        "targets": first_existing(root, "targets", args.targets, candidates["targets"]),
        "subspace": first_existing(root, "subspace", args.subspace, candidates["subspace"]),
        "adapter": first_existing(root, "adapter", args.adapter, candidates["adapter"], required=False),
        "metrics": first_existing(root, "metrics", args.metrics, candidates["metrics"], required=False),
    }
    cap_path = first_existing(root, "capture_detailed", args.capture_detailed, candidates["capture_detailed"], required=False)
    emb_on_path = first_existing(root, "eval_emb_on", args.eval_emb_on, candidates["eval_emb_on"], required=False)
    if cap_path is None and emb_on_path is None:
        raise FileNotFoundError(
            "Need adapter-on representation artifact. Missing both:\n"
            + "\n".join(f"  {c}" for c in candidates["capture_detailed"] + candidates["eval_emb_on"])
        )

    eval_off = load_json(paths["eval_off_json"])
    eval_emb_off = load_pt(paths["eval_emb_off"])
    dirs = load_pt(paths["directions"])
    targets = load_pt(paths["targets"])
    q_basis = q_basis_from_cache(load_pt(paths["subspace"]))
    eval_on = load_json(paths["eval_on_json"])
    metrics = load_json(paths["metrics"]) if paths["metrics"] else {}

    cfg = {}
    if paths["adapter"] and paths["adapter"].exists():
        ck = load_pt(paths["adapter"])
        cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    layers = [int(x) for x in cfg.get("ACT_LAYERS", ACT_LAYERS_DEFAULT)]
    layers = [l for l in layers if l in ACT_LAYERS_DEFAULT]
    if not layers:
        layers = ACT_LAYERS_DEFAULT
    weights = weighted_layers(cfg, layers)

    off_records = eval_off_records(eval_off, eval_emb_off)
    if cap_path and cap_path.exists():
        on_records, on_source = on_records_from_capture(load_pt(cap_path))
    else:
        on_records, on_source = on_records_from_eval_emb(off_records, load_pt(emb_on_path))
        print("WARNING: using eval_emb_on.pt fallback rather than detailed h_corr capture; suitable for trajectory from cached on representations, but not hook-only diagnostics.")
    join = attach_eval_on(on_records, eval_on)
    if join["labeled"] == 0:
        raise RuntimeError("No adapter-on WildGuard refusal labels were parsed; cannot form transition-conditioned subsets.")

    off_by_id = {r["sample_id"]: r for r in off_records}
    missing_ids = [r.get("sample_id") for r in on_records if r.get("sample_id") not in off_by_id]
    if missing_ids:
        raise RuntimeError(f"adapter-on/off join failed for {len(missing_ids)} samples; first missing id={missing_ids[0]}")

    rows_sample = []
    for rec in on_records:
        group = rec["group"]
        sid = rec["sample_id"]
        off = off_by_id[sid]
        tr = transition_name(group, rec.get("adapter_on_refused"))
        for layer in layers:
            if group == "jb_corr":
                doff = jb_layer_distance(off, targets, q_basis, layer, "off")
                don = jb_layer_distance(rec, targets, q_basis, layer, "on")
                rows_sample.append({"panel": "A", "series": tr, "sample_id": sid, "group": group, "transition": tr, "layer": layer, "metric": "jb_target_distance_change", "value": don - doff})
            elif group == "or_corr":
                doff = or_layer_distance(off, dirs, targets, layer, "off")
                don = or_layer_distance(rec, dirs, targets, layer, "on")
                rows_sample.append({"panel": "B", "series": tr, "sample_id": sid, "group": group, "transition": tr, "layer": layer, "metric": "or_target_distance_change", "value": don - doff})
            elif group in {"jb_blocked", "harm_refuse", "benign_ans"}:
                disp = disp_layer_norm(off, rec, targets, q_basis, layer, group)
                rows_sample.append({"panel": "C", "series": group, "sample_id": sid, "group": group, "transition": group, "layer": layer, "metric": "subspace_displacement_norm", "value": disp})

    rng = np.random.default_rng(args.seed)
    rows_summary = []
    for series in ["jb_corr_to_refused", "jb_corr_still_answered", "or_corr_to_answered", "or_corr_still_refused", "jb_blocked", "harm_refuse", "benign_ans"]:
        sub_series = [r for r in rows_sample if r["series"] == series]
        if not sub_series:
            print(f"WARNING: no samples for {series}")
            continue
        for layer in layers:
            vals = [r["value"] for r in sub_series if r["layer"] == layer]
            mean, lo, hi = bootstrap_mean_ci(vals, rng, args.n_bootstrap)
            rows_summary.append({"stat": "layer_mean", "series": series, "layer": layer, "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)})
        vals_all = []
        by_sample = {}
        for r in sub_series:
            by_sample.setdefault(r["sample_id"], []).append(r["value"])
        for vals in by_sample.values():
            vals_all.append(float(np.mean(vals)))
        mean, lo, hi = bootstrap_mean_ci(vals_all, rng, args.n_bootstrap)
        rows_summary.append({"stat": "layer_average", "series": series, "layer": "mean_ACT_LAYERS", "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals_all)})
        vals_last = [r["value"] for r in sub_series if r["layer"] == layers[-1]]
        mean, lo, hi = bootstrap_mean_ci(vals_last, rng, args.n_bootstrap)
        rows_summary.append({"stat": "final_layer", "series": series, "layer": layers[-1], "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals_last)})

    required_nonempty = ["jb_corr_to_refused", "jb_corr_still_answered", "or_corr_to_answered", "or_corr_still_refused"]
    counts = count_by_transition(on_records)
    for name in required_nonempty:
        if counts.get(name, 0) == 0:
            print(f"WARNING: transition subset has zero samples: {name}")

    csv_path = out_dir / "layerwise_correction_trajectory.csv"
    write_csv(csv_path, rows_sample + rows_summary)

    def avg(series):
        row = next((r for r in rows_summary if r["series"] == series and r["stat"] == "layer_average"), None)
        return math.nan if row is None else row["mean"]

    comparison = {
        "jb_corr_refused_minus_still_answered_layer_average": avg("jb_corr_to_refused") - avg("jb_corr_still_answered"),
        "or_corr_answered_minus_still_refused_layer_average": avg("or_corr_to_answered") - avg("or_corr_still_refused"),
        "preservation_layer_average": {g: avg(g) for g in ["jb_blocked", "harm_refuse", "benign_ans"]},
    }

    expected = {"tag": FINAL_TAG, "jb_scale": 1.5, "or_scale": 2.5, "abstain_threshold": 0.2}
    config_ok = args.config_tag == expected["tag"]
    if metrics:
        cfgm = metrics.get("config", {})
        config_ok = config_ok and abs(float(cfgm.get("JB_SCALE", expected["jb_scale"])) - expected["jb_scale"]) < 1e-8
        config_ok = config_ok and abs(float(cfgm.get("OR_SCALE", expected["or_scale"])) - expected["or_scale"]) < 1e-8
        config_ok = config_ok and abs(float(cfgm.get("ABSTAIN_THRESHOLD", expected["abstain_threshold"])) - expected["abstain_threshold"]) < 1e-8

    metadata = {
        "configuration_identifier": args.config_tag,
        "expected_final_configuration": expected,
        "configuration_consistent_with_final_table": bool(config_ok),
        "checkpoint_path": str(paths["adapter"]) if paths["adapter"] else None,
        "checkpoint_sha256_16": sha16(paths["adapter"]) if paths["adapter"] else None,
        "intervention_layers": layers,
        "layer_weights": weights.tolist(),
        "artifact_paths": {k: str(v) if v else None for k, v in paths.items()} | {"capture_detailed": str(cap_path) if cap_path else None, "eval_emb_on": str(emb_on_path) if emb_on_path else None},
        "on_representation_source": on_source,
        "transition_label_join": join,
        "sample_count_by_transition": counts,
        "comparisons": comparison,
    }
    json_path = out_dir / "layerwise_correction_trajectory.json"
    save_json(json_path, {"metadata": metadata, "summary_rows": rows_summary, "sample_rows": rows_sample})

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6), constrained_layout=True)
    line_panel(axes[0], rows_summary, ["jb_corr_to_refused", "jb_corr_still_answered"], title="Jailbreak correction trajectory", ylabel="Change in target distance")
    line_panel(axes[1], rows_summary, ["or_corr_to_answered", "or_corr_still_refused"], title="Over-refusal correction trajectory", ylabel="Change in target distance")
    line_panel(axes[2], rows_summary, ["jb_blocked", "harm_refuse", "benign_ans"], title="Preservation-group displacement", ylabel="Subspace displacement norm")
    axes[2].axhline(0, color="0.35", lw=0.8)
    fig_path_pdf = out_dir / "fig_layerwise_correction_trajectory.pdf"
    fig_path_png = out_dir / "fig_layerwise_correction_trajectory.png"
    fig.savefig(fig_path_pdf, bbox_inches="tight")
    fig.savefig(fig_path_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    lines = ["# Layer-wise Correction Trajectory Summary\n"]
    if not config_ok:
        lines.append("WARNING: configuration does not match the declared final `Ours` setting; do not use this as a main-paper result without resolving the mismatch.\n")
    else:
        lines.append("Configuration check passed for `jb1p5_or2p5_thr0p2`.\n")
    jb_ref = avg("jb_corr_to_refused")
    jb_still = avg("jb_corr_still_answered")
    or_ans = avg("or_corr_to_answered")
    or_still = avg("or_corr_still_refused")
    lines.append(f"- `jb_corr_to_refused` layer-average target-distance change: {jb_ref:+.4f}")
    lines.append(f"- `jb_corr_still_answered` layer-average target-distance change: {jb_still:+.4f}")
    lines.append("- `jb_corr_to_refused` moves more strongly toward the target than `jb_corr_still_answered`." if jb_ref < jb_still else "- `jb_corr_to_refused` does not move more strongly toward the target than `jb_corr_still_answered`; avoid overstating the representation story.")
    lines.append(f"- `or_corr_to_answered` layer-average target-distance change: {or_ans:+.4f}")
    lines.append(f"- `or_corr_still_refused` layer-average target-distance change: {or_still:+.4f}")
    lines.append("- `or_corr` transition subsets are clearly separated in target-distance trajectory." if abs(or_ans - or_still) > 0.1 else "- `or_corr` transition subsets are not strongly separated in target-distance trajectory.")
    for g, val in comparison["preservation_layer_average"].items():
        lines.append(f"- `{g}` layer-average subspace displacement norm: {val:.4f}")
    jb_blocked_val = comparison["preservation_layer_average"].get("jb_blocked", math.nan)
    harm_val = comparison["preservation_layer_average"].get("harm_refuse", math.nan)
    benign_val = comparison["preservation_layer_average"].get("benign_ans", math.nan)
    if np.isfinite(jb_blocked_val) and np.isfinite(harm_val) and np.isfinite(benign_val) and jb_blocked_val > max(harm_val, benign_val) * 1.5:
        lines.append("- `jb_blocked` shows non-negligible displacement relative to the other preservation groups; discuss this as a caveat.")
    else:
        lines.append("- Preservation-group displacement does not show a large `jb_blocked` outlier relative to the other preservation groups.")
    safe_main = config_ok and (jb_ref < jb_still) and all(np.isfinite(v) for v in comparison["preservation_layer_average"].values())
    lines.append("- Main-paper use recommendation: " + ("safe as a supporting analysis." if safe_main else "use cautiously or keep in appendix."))
    md_path = out_dir / "layerwise_correction_trajectory_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Configuration consistency check:", "PASS" if config_ok else "WARNING/MISMATCH")
    print("Transition subset counts:")
    for k in ["jb_corr_to_refused", "jb_corr_still_answered", "or_corr_to_answered", "or_corr_still_refused"]:
        print(f"  {k}: {counts.get(k, 0)}")
    print("Layer-averaged target-distance change:")
    print(f"  jb_corr_to_refused: {jb_ref:+.4f}")
    print(f"  jb_corr_still_answered: {jb_still:+.4f}")
    print(f"  or_corr_to_answered: {or_ans:+.4f}")
    print(f"  or_corr_still_refused: {or_still:+.4f}")
    print("Preservation layer-averaged displacement:")
    for g, val in comparison["preservation_layer_average"].items():
        print(f"  {g}: {val:.4f}")
    print("Generated files:")
    for p in [fig_path_pdf, fig_path_png, csv_path, json_path, md_path]:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
