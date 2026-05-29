#!/usr/bin/env python3
"""Fast CPU-only centroid displacement plots for NLP-TEAM 17 sweeps.

This script does not load the model, does not generate responses, and does not
run adapter hook capture. It only reads cached off/on representation files and
plots group-centroid movement in the train-fitted behavior-axis coordinate
system.

The behavior-axis definition intentionally matches
build_q5d_motivation_and_inference_figures.py:
  x = layer-averaged standardized v_harm projection of the user-span mean
  y = layer-averaged standardized v_refuse projection of the final prompt token
with center/scale fitted only from cache/v07/emb_train.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np


ACT_LAYERS = [9, 12, 15, 18, 21]
GROUPS = ["jb_corr", "jb_blocked", "harm_refuse", "or_corr", "benign_ans"]
PLOT_GROUPS = ["harm_refuse", "jb_blocked", "jb_corr", "or_corr", "benign_ans"]
COLORS = {
    "harm_refuse": "#8B1E1E",
    "jb_blocked": "#C26A00",
    "jb_corr": "#E7A95C",
    "or_corr": "#2A9D8F",
    "benign_ans": "#1D4E89",
}
LABELS = {group: group for group in GROUPS}
LABEL_OFFSETS = {
    "harm_refuse": (5, 6),
    "jb_blocked": (5, 6),
    "jb_corr": (-7, 5),
    "or_corr": (5, 7),
    "benign_ans": (5, 7),
}

torch = None
plt = None


def parse_args():
    here = Path(__file__).resolve()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=here.parents[1])
    p.add_argument("--sweep-root", type=Path, default=Path("cache/nlp_team17/sweep"))
    p.add_argument("--out-root", type=Path, default=Path("output/figures/nlp_team17_coef_sweep_centroids"))
    p.add_argument("--emb-train", type=Path, default=Path("cache/v07/emb_train.pt"))
    p.add_argument("--eval-emb-off", type=Path, default=Path("cache/unified_vseries_3b/eval_emb_off_v85.pt"))
    p.add_argument("--eval-off-json", type=Path, default=Path("cache/unified_vseries_3b/eval_off.json"))
    p.add_argument("--directions", type=Path, default=Path("cache/v85_onepass/directions.pt"))
    p.add_argument("--layers", type=int, nargs="+", default=ACT_LAYERS)
    p.add_argument("--sweep-tags", nargs="*", default=None, help="Optional sweep directory names to plot. Defaults to all sweeps under --sweep-root.")
    p.add_argument("--max-cloud-per-group", type=int, default=450)
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


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
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 160,
        }
    )


def load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def layer_get(mapping: dict, layer: int):
    if layer in mapping:
        return mapping[layer]
    key = str(layer)
    if key in mapping:
        return mapping[key]
    raise KeyError(f"missing layer {layer}")


def user_mean(sample: dict, layer: int):
    h = layer_get(sample["span_h"], layer).float()
    return h if h.dim() == 1 else h.mean(0)


def final_token(sample: dict, layer: int):
    return layer_get(sample["last_h"], layer).float()


def group_records(eval_json: dict, emb_records: list[dict]) -> dict[str, list[dict]]:
    meta = eval_json.get("eval_samples")
    if not isinstance(meta, list):
        raise RuntimeError("eval_off.json missing eval_samples")
    if len(meta) != len(emb_records):
        raise RuntimeError(f"metadata/embedding length mismatch: {len(meta)} vs {len(emb_records)}")
    out = {g: [] for g in GROUPS}
    for i, (m, rec) in enumerate(zip(meta, emb_records)):
        r = dict(rec)
        r["sample_id"] = m.get("id", rec.get("id", i))
        r["group"] = m.get("group", rec.get("group"))
        if r["group"] in out:
            out[r["group"]].append(r)
    return out


def group_on_records(emb_records: list[dict]) -> dict[str, list[dict]]:
    out = {g: [] for g in GROUPS}
    for i, rec in enumerate(emb_records):
        r = dict(rec)
        r["sample_id"] = rec.get("id", rec.get("sample_id", i))
        group = rec.get("group")
        if group in out:
            out[group].append(r)
    return out


def train_stats(train_by_group: dict[str, list[dict]], dirs: dict, layers: list[int]):
    stats = {"harm": {}, "refuse": {}}
    for layer in layers:
        vh = layer_get(dirs["v_harm"], layer).float()
        vr = layer_get(dirs["v_refuse"], layer).float()
        vals = {"harm": {}, "refuse": {}}
        for group in GROUPS:
            vals["harm"][group] = np.asarray([float(user_mean(s, layer) @ vh) for s in train_by_group[group]])
            vals["refuse"][group] = np.asarray([float(final_token(s, layer) @ vr) for s in train_by_group[group]])
        for metric in ("harm", "refuse"):
            means = [vals[metric][g].mean() for g in GROUPS if vals[metric][g].size]
            stds = [vals[metric][g].std(ddof=1) for g in GROUPS if vals[metric][g].size > 1]
            center = float(np.mean(means)) if means else 0.0
            scale = float(np.mean(stds)) if stds else 1.0
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            stats[metric][layer] = (center, scale)
    return stats


def behavior_scores(by_group: dict[str, list[dict]], dirs: dict, stats: dict, layers: list[int]):
    out = {}
    for group in GROUPS:
        pts = []
        for rec in by_group[group]:
            xs, ys = [], []
            for layer in layers:
                vh = layer_get(dirs["v_harm"], layer).float()
                vr = layer_get(dirs["v_refuse"], layer).float()
                hc, hs = stats["harm"][layer]
                rc, rs = stats["refuse"][layer]
                xs.append((float(user_mean(rec, layer) @ vh) - hc) / hs)
                ys.append((float(final_token(rec, layer) @ vr) - rc) / rs)
            pts.append([float(np.mean(xs)), float(np.mean(ys))])
        out[group] = np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2), dtype=np.float64)
    return out


def axis_limits(score_sets: list[dict[str, np.ndarray]]):
    arrs = [pts for scores in score_sets for pts in scores.values() if pts.size]
    all_pts = np.concatenate(arrs, axis=0)
    lo = np.nanmin(all_pts, axis=0)
    hi = np.nanmax(all_pts, axis=0)
    span = np.maximum(hi - lo, 1.0)
    return (lo[0] - 0.08 * span[0], hi[0] + 0.08 * span[0]), (lo[1] - 0.08 * span[1], hi[1] + 0.08 * span[1])


def centroid_axis_limits(off_scores: dict[str, np.ndarray], on_score_sets: list[dict[str, np.ndarray]]):
    pts = []
    for group in GROUPS:
        if off_scores[group].size:
            pts.append(off_scores[group])
            pts.append(off_scores[group].mean(axis=0, keepdims=True))
        for scores in on_score_sets:
            if scores[group].size:
                pts.append(scores[group].mean(axis=0, keepdims=True))
    all_pts = np.concatenate(pts, axis=0)
    lo = np.nanmin(all_pts, axis=0)
    hi = np.nanmax(all_pts, axis=0)
    span = np.maximum(hi - lo, 1.0)
    return (lo[0] - 0.08 * span[0], hi[0] + 0.08 * span[0]), (lo[1] - 0.08 * span[1], hi[1] + 0.08 * span[1])


def sample_cloud(pts: np.ndarray, max_n: int, rng: np.random.Generator):
    if len(pts) <= max_n:
        return pts
    idx = rng.choice(len(pts), max_n, replace=False)
    return pts[idx]


def format_behavior_axis(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="0.86", lw=0.7)
    ax.axvline(0, color="0.86", lw=0.7)
    ax.grid(color="0.88", lw=0.7)
    ax.set_xlabel("Harmfulness-related projection")
    ax.set_ylabel("Refusal-related projection")


def draw_base_panel(ax, off_scores, xlim, ylim, rng, max_cloud: int):
    for group in PLOT_GROUPS:
        cloud = sample_cloud(off_scores[group], max_cloud, rng)
        if cloud.size:
            ax.scatter(cloud[:, 0], cloud[:, 1], s=13, alpha=0.18, color=COLORS[group], linewidths=0, rasterized=True)
    for group in PLOT_GROUPS:
        if not off_scores[group].size:
            continue
        c = off_scores[group].mean(axis=0)
        ax.scatter(c[0], c[1], s=72, color=COLORS[group], edgecolors="white", linewidths=0.9, zorder=5)
        ax.annotate(
            LABELS[group],
            c,
            xytext=LABEL_OFFSETS[group],
            textcoords="offset points",
            color=COLORS[group],
            fontsize=12,
            ha="right" if group == "jb_corr" else "left",
            va="center",
        )
    format_behavior_axis(ax, xlim, ylim)
    ax.set_title("Base-model behavior geometry", fontweight="bold")


def draw_centroid_plot(ax, tag: str, off_scores, on_scores, xlim, ylim, rng, max_cloud: int, *, show_tag=False):
    for group in PLOT_GROUPS:
        cloud = sample_cloud(off_scores[group], max_cloud, rng)
        if cloud.size:
            ax.scatter(cloud[:, 0], cloud[:, 1], s=11, alpha=0.055, color=COLORS[group], linewidths=0, rasterized=True)
    for group in PLOT_GROUPS:
        if not off_scores[group].size or not on_scores[group].size:
            continue
        off_c = off_scores[group].mean(axis=0)
        on_c = on_scores[group].mean(axis=0)
        emph = group in {"jb_corr", "or_corr"}
        alpha = 1.0 if emph else 0.45
        lw = 2.4 if emph else 1.2
        size = 86 if emph else 70
        ax.scatter(off_c[0], off_c[1], s=size, facecolors="none", edgecolors=COLORS[group], linewidths=1.6, alpha=alpha, zorder=4)
        ax.scatter(on_c[0], on_c[1], s=size, color=COLORS[group], edgecolors="white", linewidths=0.9, alpha=alpha, zorder=5)
        ax.annotate(
            "",
            xy=on_c,
            xytext=off_c,
            arrowprops=dict(arrowstyle="->", lw=lw, color=COLORS[group], alpha=alpha, shrinkA=5, shrinkB=5),
            zorder=4,
        )
        ox, oy = LABEL_OFFSETS[group]
        if group == "or_corr":
            oy = -13
        ha = "right" if group == "jb_corr" else "left"
        ax.annotate(
            LABELS[group],
            on_c,
            xytext=(ox, oy),
            textcoords="offset points",
            color=COLORS[group],
            fontsize=12 if emph else 11,
            weight="bold" if emph else "normal",
            alpha=alpha,
            ha=ha,
            va="center",
        )
    format_behavior_axis(ax, xlim, ylim)
    ax.set_title("Inference Representation Displacement", fontweight="bold")
    if show_tag:
        ax.text(0.02, 0.98, tag, transform=ax.transAxes, ha="left", va="top", fontsize=8, color="0.35")


def save_per_sweep_figure(out_path: Path, tag: str, off_scores, on_scores, xlim, ylim, rng, max_cloud: int, dpi: int):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    draw_base_panel(axes[0], off_scores, xlim, ylim, rng, max_cloud)
    draw_centroid_plot(axes[1], tag, off_scores, on_scores, xlim, ylim, rng, max_cloud)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def centroid_rows(tag: str, off_scores, on_scores):
    rows = []
    for group in GROUPS:
        if not off_scores[group].size or not on_scores[group].size:
            continue
        off_c = off_scores[group].mean(axis=0)
        on_c = on_scores[group].mean(axis=0)
        rows.append(
            {
                "sweep": tag,
                "group": group,
                "n": int(len(on_scores[group])),
                "off_x": float(off_c[0]),
                "off_y": float(off_c[1]),
                "on_x": float(on_c[0]),
                "on_y": float(on_c[1]),
                "delta_harm": float(on_c[0] - off_c[0]),
                "delta_refuse": float(on_c[1] - off_c[1]),
                "delta_norm": float(np.linalg.norm(on_c - off_c)),
            }
        )
    return rows


def sort_key(path: Path):
    m = re.fullmatch(r"jb([^_]+)_or([^_]+)_thr([^_]+)", path.name)
    if not m:
        return (999.0, 999.0, 999.0, path.name)
    conv = lambda s: float(s.replace("m", "-").replace("p", "."))
    return (conv(m.group(1)), conv(m.group(2)), conv(m.group(3)), path.name)


def main():
    args = parse_args()
    root = args.root.resolve()
    import_deps(root)

    sweep_root = resolve(root, args.sweep_root)
    out_root = resolve(root, args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    emb_train = load_pt(resolve(root, args.emb_train))
    eval_off_json = load_json(resolve(root, args.eval_off_json))
    eval_emb_off = load_pt(resolve(root, args.eval_emb_off))
    dirs = load_pt(resolve(root, args.directions))

    train_by_group = {g: list(emb_train.get(g, [])) for g in GROUPS}
    off_by_group = group_records(eval_off_json, eval_emb_off)
    stats = train_stats(train_by_group, dirs, args.layers)
    off_scores = behavior_scores(off_by_group, dirs, stats, args.layers)
    print("Behavior-axis normalization: fitted from training cache only")
    print(f"  train cache: {resolve(root, args.emb_train)}")
    print(f"  layers: {args.layers}")

    sweep_dirs = [p for p in sweep_root.iterdir() if p.is_dir() and (p / "eval_emb_on.pt").exists()]
    if args.sweep_tags:
        wanted = set(args.sweep_tags)
        sweep_dirs = [p for p in sweep_dirs if p.name in wanted]
        missing = sorted(wanted - {p.name for p in sweep_dirs})
        if missing:
            raise SystemExit("Missing requested sweep tags under " + str(sweep_root) + ":\n" + "\n".join(f"  {x}" for x in missing))
    sweep_dirs = sorted(sweep_dirs, key=sort_key)
    if not sweep_dirs:
        raise SystemExit(f"No sweep eval_emb_on.pt found under {sweep_root}")

    rng = np.random.default_rng(args.seed)
    per_sweep = []
    all_rows = []
    for sweep_dir in sweep_dirs:
        tag = sweep_dir.name
        on_by_group = group_on_records(load_pt(sweep_dir / "eval_emb_on.pt"))
        on_scores = behavior_scores(on_by_group, dirs, stats, args.layers)
        per_sweep.append((tag, on_scores))
        all_rows.extend(centroid_rows(tag, off_scores, on_scores))
        print(f"loaded {tag}", flush=True)

    # Match the original inference-displacement figure: limits are governed by
    # the base cloud plus centroid movement, not by rare adapter-on outlier
    # samples. This keeps the cluster geometry visually comparable across
    # coefficient settings.
    xlim, ylim = centroid_axis_limits(off_scores, [scores for _, scores in per_sweep])

    for tag, on_scores in per_sweep:
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / "fig_centroid_displacement.png"
        save_per_sweep_figure(
            fig_path,
            tag,
            off_scores,
            on_scores,
            xlim,
            ylim,
            rng,
            args.max_cloud_per_group,
            args.dpi,
        )
        print(f"SAVED_FIG\t{tag}\t{fig_path}", flush=True)

    n = len(per_sweep)
    cols = min(4, n)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.7 * rows_n), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, (tag, on_scores) in zip(axes, per_sweep):
        draw_centroid_plot(ax, tag, off_scores, on_scores, xlim, ylim, rng, max(80, args.max_cloud_per_group // 3), show_tag=True)
    for ax in axes[n:]:
        ax.axis("off")
    grid_path = out_root / "fig_all_sweep_centroid_displacement_grid.png"
    fig.savefig(grid_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"SAVED_FIG\tall_sweeps\t{grid_path}", flush=True)

    csv_path = out_root / "centroid_displacement_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    json_path = out_root / "centroid_displacement_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "coordinate_system": "train-fitted standardized behavior axis",
                    "x_axis": "layer-averaged standardized v_harm projection of user-span mean",
                    "y_axis": "layer-averaged standardized v_refuse projection of final prompt token",
                    "normalization_source": str(resolve(root, args.emb_train)),
                    "eval_off_source": str(resolve(root, args.eval_emb_off)),
                    "directions_source": str(resolve(root, args.directions)),
                    "layers": args.layers,
                    "groups": GROUPS,
                },
                "rows": all_rows,
                "sweeps": [tag for tag, _ in per_sweep],
            },
            f,
            indent=2,
        )
        f.write("\n")

    print("\nGenerated centroid displacement plots:")
    print(f"  {out_root / 'fig_all_sweep_centroid_displacement_grid.png'}")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    for tag, _ in per_sweep:
        print(f"  {out_root / tag / 'fig_centroid_displacement.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
