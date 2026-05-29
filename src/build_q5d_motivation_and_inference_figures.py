#!/usr/bin/env python3
"""Build Q5D motivation and inference-correction analysis figures for v85.

This is a diagnostic analysis script only. It does not train or modify the
adapter checkpoint. It uses frozen training/evaluation caches, optionally runs
prompt-prefill capture with the trained adapter, and writes figures plus all
numerical summaries under output/figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np


ACT_LAYERS_DEFAULT = [9, 12, 15, 18, 21]
MEASURE_LAYERS_DEFAULT = list(range(9, 28))
HIDDEN_DEFAULT = 3072
FLOW_RANK_DEFAULT = 256
T_DIM_DEFAULT = 32
HISTORY_N_DEFAULT = 2
GATE_RANK_DEFAULT = 128
GATE_T_DIM_DEFAULT = 16
ABSTAIN_THRESHOLD_DEFAULT = 0.20
OR_SCALE_DEFAULT = 2.0
MODEL_ID_DEFAULT = "meta-llama/Llama-3.2-3B-Instruct"
LAYER_WEIGHTS_DEFAULT = {9: 0.5, 12: 1.0, 15: 1.5, 18: 2.0, 21: 2.0}
BOOTSTRAP_N = 2000
RANDOM5D_N = 50
SEED_DEFAULT = 20260529
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 9
TITLE_SIZE = 10
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8
ANNOTATION_SIZE = 8
FIGSIZE_SINGLE = (7.2, 5.0)
FIGSIZE_WIDE = (10.8, 4.8)
FIGSIZE_GRID = (10.8, 7.2)
FIGSIZE_MAIN_3PANEL = (12.0, 3.8)
FIGSIZE_APPENDIX_3PANEL = (12.0, 3.8)
OUTPUT_DPI = 300

GROUPS = ["jb_corr", "jb_blocked", "harm_refuse", "or_corr", "benign_ans"]
PLOT_GROUPS = ["harm_refuse", "jb_blocked", "jb_corr", "or_corr", "benign_ans"]
HARM_POS = {"jb_corr", "jb_blocked", "harm_refuse"}
SAFE_NEG = {"or_corr", "benign_ans"}
REFUSED_POS = {"jb_blocked", "harm_refuse", "or_corr"}
ANSWERED_NEG = {"jb_corr", "benign_ans"}
CORRECTION_GROUPS = ["jb_corr", "or_corr"]
PRESERVATION_GROUPS = ["jb_blocked", "harm_refuse", "benign_ans"]
SEMANTIC_DIRS = ["v_harm", "v_refuse", "v_harm_svd", "v_refuse_svd", "clf_dir"]
SEMANTIC_LABELS = ["v_harm", "v_refuse", "v_harm_svd", "v_refuse_svd", "v_clf"]
TRANSITION_ORDER = [
    "jb_corr_to_refused",
    "jb_corr_still_answered",
    "or_corr_to_answered",
    "or_corr_still_refused",
    "jb_blocked",
    "harm_refuse",
    "benign_ans",
]
TRANSITION_DISPLAY = {
    "jb_corr_to_refused": "jb_corr\nrefused",
    "jb_corr_still_answered": "jb_corr\nanswered",
    "or_corr_to_answered": "or_corr\nanswered",
    "or_corr_still_refused": "or_corr\nrefused",
}

COLORS = {
    "harm_refuse": "#8B1E1E",
    "jb_blocked": "#C26A00",
    "jb_corr": "#E7A95C",
    "or_corr": "#2A9D8F",
    "benign_ans": "#1D4E89",
}

DISPLAY_GROUP = {
    "jb_corr": "jb_corr",
    "jb_blocked": "jb_blocked",
    "harm_refuse": "harm_refuse",
    "or_corr": "or_corr",
    "benign_ans": "benign_ans",
}

CONTRAST_LABELS = {
    "jb_corr_minus_benign_ans_answered": "jb_corr - benign_ans",
    "harm_refuse_minus_or_corr_refused": "harm_refuse - or_corr",
    "jb_blocked_minus_or_corr_aux": "jb_blocked - or_corr",
    "jb_blocked_minus_jb_corr_jailbreak": "jb_blocked - jb_corr",
    "harm_refuse_minus_jb_corr_aux": "harm_refuse - jb_corr",
    "or_corr_minus_benign_ans_safe": "or_corr - benign_ans",
}

CONTRAST_COLORS = {
    "jb_corr_minus_benign_ans_answered": COLORS["jb_corr"],
    "harm_refuse_minus_or_corr_refused": COLORS["harm_refuse"],
    "jb_blocked_minus_or_corr_aux": COLORS["jb_blocked"],
    "jb_blocked_minus_jb_corr_jailbreak": COLORS["jb_blocked"],
    "harm_refuse_minus_jb_corr_aux": COLORS["harm_refuse"],
    "or_corr_minus_benign_ans_safe": COLORS["or_corr"],
}

TASK_LABELS = {
    "jb_corr_vs_jb_blocked": "jb_corr vs jb_blocked",
    "or_corr_vs_benign_ans": "or_corr vs benign_ans",
    "correction_vs_preservation": "jb_corr + or_corr vs jb_blocked + harm_refuse + benign_ans",
}

TASK_COLORS = {
    "jb_corr_vs_jb_blocked": COLORS["jb_corr"],
    "or_corr_vs_benign_ans": COLORS["or_corr"],
    "correction_vs_preservation": "#4B5563",
}

SUBSPACE_LABELS = {
    "Q1D_refuse": "refusal only",
    "Q2D_harm_refuse": "harm + refusal",
    "Q4D_variation": "harm/refusal + variation",
    "Q5D_full": "full Q5D",
    "Random5D": "random 5D",
}

MAIN_SUBSPACE_LABELS = {
    "Q1D_refuse": "Refusal only",
    "Q2D_harm_refuse": "+ Harmfulness",
    "Q4D_variation": "+ Variation",
    "Q5D_full": "+ Sep",
}

SUBSPACE_ORDER = ["Q1D_refuse", "Q2D_harm_refuse", "Q4D_variation", "Q5D_full", "Random5D"]
MAIN_SUBSPACE_ORDER = ["Q1D_refuse", "Q2D_harm_refuse", "Q4D_variation", "Q5D_full"]
TRANSITION_COLORS = {
    "jb_corr_to_refused": COLORS["jb_corr"],
    "jb_corr_still_answered": COLORS["jb_corr"],
    "or_corr_to_answered": COLORS["or_corr"],
    "or_corr_still_refused": COLORS["or_corr"],
    "jb_blocked": COLORS["jb_blocked"],
    "harm_refuse": COLORS["harm_refuse"],
    "benign_ans": COLORS["benign_ans"],
}

ARTIFACT_CANDIDATES = {
    "emb_train": ["cache/v07/emb_train.pt"],
    "eval_emb_off": ["cache/v07/eval_emb_off.pt"],
    "eval_off": ["cache/v07/eval_off.json"],
    "directions": ["cache/v85_onepass/directions.pt"],
    "targets": ["cache/v85_onepass/targets.pt"],
    "subspace": ["cache/v85_onepass/subspace_target.pt"],
    "eval_on": ["cache/v85_onepass/eval_on.json"],
    "eval_emb_on": ["cache/v85_onepass/eval_emb_on.pt"],
    "capture_simple": ["cache/v85_onepass/eval_repr_on_capture.pt"],
    "capture_detailed": ["cache/v85_onepass/eval_repr_on_capture_detailed.pt"],
    "adapter": ["output/adapter_v85_onepass.pt"],
}

ARTIFACT_PROFILE_CANDIDATES = {
    "v85": ARTIFACT_CANDIDATES,
    "nlp_team17": {
        "emb_train": ["cache/v07/emb_train.pt"],
        "eval_emb_off": ["cache/unified_vseries_3b/eval_emb_off_v85.pt", "cache/v07/eval_emb_off.pt"],
        "eval_off": ["cache/unified_vseries_3b/eval_off.json", "cache/v07/eval_off.json"],
        "directions": ["cache/v85_onepass/directions.pt", "cache/nlp_team17/directions.pt"],
        "targets": ["cache/v85_onepass/targets.pt", "cache/nlp_team17/targets.pt"],
        "subspace": ["cache/v85_onepass/subspace_target.pt", "cache/nlp_team17/subspace_target.pt"],
        "eval_on": ["cache/nlp_team17/eval_on.json"],
        "eval_emb_on": ["cache/nlp_team17/eval_emb_on.pt"],
        "capture_simple": ["cache/nlp_team17/eval_repr_on_capture.pt"],
        "capture_detailed": ["cache/nlp_team17/eval_repr_on_capture_detailed.pt"],
        "adapter": ["output/adapter_v85_onepass.pt", "output/adapter_nlp_team17.pt"],
    },
}

torch = None
plt = None
LogisticRegression = None
SAVE_PDF = True


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=here.parents[1], help="v85_release root.")
    parser.add_argument("--artifact-profile", default="v85", choices=sorted(ARTIFACT_PROFILE_CANDIDATES), help="Artifact layout/profile to analyze.")
    parser.add_argument("--jb-scale", type=float, default=None, help="Override jailbreak steering scale for capture/generation.")
    parser.add_argument("--or-scale", type=float, default=None, help="Override over-refusal steering scale for capture/generation.")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--eval-off-path", type=Path, default=None)
    parser.add_argument("--eval-emb-off-path", type=Path, default=None)
    parser.add_argument("--eval-on-path", type=Path, default=None)
    parser.add_argument("--eval-emb-on-path", type=Path, default=None)
    parser.add_argument("--capture-detailed-path", type=Path, default=None)
    parser.add_argument("--directions-path", type=Path, default=None)
    parser.add_argument("--targets-path", type=Path, default=None)
    parser.add_argument("--subspace-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for figure and summary outputs. Defaults to output/figures under --root.")
    parser.add_argument("--png-only", action="store_true", help="Save PNG figures only and skip PDF figure files.")
    parser.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--random5d-n", type=int, default=RANDOM5D_N)
    parser.add_argument("--force-recapture", action="store_true")
    parser.add_argument("--skip-eval-on-generation", action="store_true")
    parser.add_argument("--skip-detailed-inference", action="store_true", help="Skip detailed routing/update diagnostics and build only representation-analysis figures.")
    parser.add_argument("--use-eval-emb-on-cache", action="store_true", help="Use eval_emb_on.pt for CPU-only adapter-on representation analysis; skips hook capture and applied-update diagnostics.")
    return parser.parse_args()


def import_runtime_deps(root: Path) -> None:
    global torch, plt, LogisticRegression
    try:
        import torch as torch_mod
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: torch.") from exc
    try:
        from sklearn.linear_model import LogisticRegression as LR
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: scikit-learn. Install scikit-learn in the runtime.") from exc

    mpl_dir = root / "output/figures/.matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_mod

    torch = torch_mod
    plt = plt_mod
    LogisticRegression = LR


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.sans-serif": [FONT_FAMILY, "Arial", "Liberation Sans", "sans-serif"],
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": OUTPUT_DPI,
        }
    )


def load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_pt(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def layer_get(mapping: dict, layer: int):
    if layer in mapping:
        return mapping[layer]
    key = str(layer)
    if key in mapping:
        return mapping[key]
    raise KeyError(f"missing layer {layer}")


def unit(vec):
    vec = vec.float()
    return vec / (vec.norm() + 1e-8)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_candidates_for_args(args: argparse.Namespace) -> dict[str, list[str]]:
    candidates = deepcopy(ARTIFACT_PROFILE_CANDIDATES[args.artifact_profile])
    overrides = {
        "adapter": args.adapter_path,
        "eval_off": args.eval_off_path,
        "eval_emb_off": args.eval_emb_off_path,
        "eval_on": args.eval_on_path,
        "eval_emb_on": args.eval_emb_on_path,
        "capture_detailed": args.capture_detailed_path,
        "directions": args.directions_path,
        "targets": args.targets_path,
        "subspace": args.subspace_path,
    }
    for name, path in overrides.items():
        if path is not None:
            candidates[name] = [str(path)]
    return candidates


def resolve_candidate(root: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    return path if path.is_absolute() else root / path


def default_artifact_path(root: Path, candidates: dict[str, list[str]], name: str) -> Path:
    return resolve_candidate(root, candidates[name][0])


def find_artifact(root: Path, candidates_by_name: dict[str, list[str]], name: str, required: bool = True) -> Path | None:
    candidates = candidates_by_name[name]
    for rel in candidates:
        path = resolve_candidate(root, rel)
        if path.exists():
            return path
    wanted = {Path(rel).name for rel in candidates}
    hits = [p for p in root.rglob("*") if p.is_file() and p.name in wanted and ".git" not in p.parts]
    if hits:
        return sorted(hits, key=lambda p: (len(p.parts), str(p)))[0]
    if required:
        return None
    return None


def resolve_artifacts(root: Path, args: argparse.Namespace) -> tuple[dict[str, Path | None], dict[str, list[str]]]:
    candidates = artifact_candidates_for_args(args)
    required = ["emb_train", "eval_emb_off", "eval_off", "directions", "targets", "subspace", "adapter"]
    optional = ["eval_on", "eval_emb_on", "capture_simple", "capture_detailed"]
    artifacts = {name: find_artifact(root, candidates, name, required=name in required) for name in required + optional}
    missing = [name for name in required if artifacts[name] is None]
    print(f"Artifact search: profile={args.artifact_profile}")
    for name in required + optional:
        print(f"  {name:16s} {artifacts[name] if artifacts[name] else 'MISSING'}")
    if missing:
        print("\nRequired artifacts are missing:")
        for name in missing:
            print(f"  {name}: candidates={candidates[name]}")
        raise SystemExit(2)
    return artifacts, candidates


def user_mean_off(sample: dict, layer: int):
    h = layer_get(sample["span_h"], layer).float()
    return h if h.dim() == 1 else h.mean(0)


def final_off(sample: dict, layer: int):
    return layer_get(sample["last_h"], layer).float()


def user_mean_on(sample: dict, layer: int):
    return layer_get(sample["span_h_corr"], layer).float()


def final_on(sample: dict, layer: int):
    return layer_get(sample["last_h_corr"], layer).float()


def group_samples(samples: list[dict]) -> dict[str, list[dict]]:
    out = {group: [] for group in GROUPS}
    for sample in samples:
        group = sample.get("group")
        if group in out:
            out[group].append(sample)
    return out


def require_groups(name: str, grouped: dict[str, list]) -> None:
    missing = [group for group in GROUPS if not grouped.get(group)]
    if missing:
        raise RuntimeError(f"{name} is missing groups: {missing}")


def bootstrap_mean_ci(values, rng: np.random.Generator, n_boot: int):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(values.mean())
    if values.size == 1 or n_boot <= 0:
        return mean, mean, mean
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def bootstrap_centroid_ci(points, rng: np.random.Generator, n_boot: int):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        nan = np.asarray([math.nan, math.nan])
        return nan, nan, nan
    mean = points.mean(axis=0)
    if len(points) == 1 or n_boot <= 0:
        return mean, mean, mean
    idx = rng.integers(0, len(points), size=(n_boot, len(points)))
    boot = points[idx].mean(axis=1)
    return mean, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def auroc_score(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64) + 1.0
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def weighted_layers(config: dict, layers: list[int]) -> np.ndarray:
    raw = config.get("LAYER_WEIGHTS", LAYER_WEIGHTS_DEFAULT)
    weights = []
    for layer in layers:
        if isinstance(raw, dict):
            weights.append(float(raw.get(layer, raw.get(str(layer), 1.0))))
        else:
            weights.append(1.0)
    arr = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(arr).all() or arr.sum() <= 0:
        arr = np.ones(len(layers))
    return arr / arr.sum()


def original_direction_matrix(dirs: dict, layer: int):
    vecs = []
    for key in SEMANTIC_DIRS:
        vec = layer_get(dirs[key], layer).float()
        vecs.append(vec / (vec.norm() + 1e-8))
    return torch.stack(vecs)


def orthonormal_basis(vecs):
    mat = torch.stack([unit(v) for v in vecs], dim=1)
    q, _ = torch.linalg.qr(mat)
    return q.T


def subspace_basis(dirs: dict, layer: int, kind: str):
    if kind == "Q1D_refuse":
        vecs = [layer_get(dirs["v_refuse"], layer)]
    elif kind == "Q2D_harm_refuse":
        vecs = [layer_get(dirs["v_harm"], layer), layer_get(dirs["v_refuse"], layer)]
    elif kind == "Q4D_variation":
        vecs = [
            layer_get(dirs["v_harm"], layer),
            layer_get(dirs["v_refuse"], layer),
            layer_get(dirs["v_harm_svd"], layer),
            layer_get(dirs["v_refuse_svd"], layer),
        ]
    elif kind == "Q5D_full":
        vecs = [layer_get(dirs[key], layer) for key in SEMANTIC_DIRS]
    else:
        raise ValueError(kind)
    return orthonormal_basis(vecs)


def q_basis_from_cache(subspace_cache: dict):
    return subspace_cache["Q"] if "Q" in subspace_cache else subspace_cache


def train_projection_stats(train_by_group, dirs, layers):
    stats = {"harm": {}, "refuse": {}}
    for layer in layers:
        vh = layer_get(dirs["v_harm"], layer).float()
        vr = layer_get(dirs["v_refuse"], layer).float()
        vals = {"harm": {}, "refuse": {}}
        for group in GROUPS:
            vals["harm"][group] = np.asarray([float(user_mean_off(s, layer) @ vh) for s in train_by_group[group]])
            vals["refuse"][group] = np.asarray([float(final_off(s, layer) @ vr) for s in train_by_group[group]])
        for metric in ("harm", "refuse"):
            means = [vals[metric][g].mean() for g in GROUPS if vals[metric][g].size]
            stds = [vals[metric][g].std(ddof=1) for g in GROUPS if vals[metric][g].size > 1]
            center = float(np.mean(means)) if means else 0.0
            scale = float(np.mean(stds)) if stds else 1.0
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            stats[metric][layer] = (center, scale)
    return stats


def behavior_projection(samples_by_group, dirs, stats, layers, source: str):
    out = {"harm": {g: {} for g in GROUPS}, "refuse": {g: {} for g in GROUPS}}
    for layer in layers:
        vh = layer_get(dirs["v_harm"], layer).float()
        vr = layer_get(dirs["v_refuse"], layer).float()
        hc, hs = stats["harm"][layer]
        rc, rs = stats["refuse"][layer]
        for group in GROUPS:
            hx, ry = [], []
            for sample in samples_by_group[group]:
                if source == "off":
                    hu, hf = user_mean_off(sample, layer), final_off(sample, layer)
                else:
                    hu, hf = user_mean_on(sample, layer), final_on(sample, layer)
                hx.append((float(hu @ vh) - hc) / hs)
                ry.append((float(hf @ vr) - rc) / rs)
            out["harm"][group][layer] = np.asarray(hx, dtype=np.float64)
            out["refuse"][group][layer] = np.asarray(ry, dtype=np.float64)
    return out


def aggregate_behavior(proj, layers):
    out = {}
    for group in GROUPS:
        x = np.stack([proj["harm"][group][layer] for layer in layers], axis=1).mean(axis=1)
        y = np.stack([proj["refuse"][group][layer] for layer in layers], axis=1).mean(axis=1)
        out[group] = np.stack([x, y], axis=1)
    return out


def signed_mean_diff(left, right):
    return float(np.asarray(left).mean() - np.asarray(right).mean())


def compute_axis_contrasts(proj, layers, n_boot, seed):
    rng = np.random.default_rng(seed)
    specs = [
        ("harm", "jb_corr_minus_benign_ans_answered", "jb_corr", "benign_ans", +1),
        ("harm", "harm_refuse_minus_or_corr_refused", "harm_refuse", "or_corr", +1),
        ("harm", "jb_blocked_minus_or_corr_aux", "jb_blocked", "or_corr", +1),
        ("refuse", "jb_blocked_minus_jb_corr_jailbreak", "jb_blocked", "jb_corr", +1),
        ("refuse", "harm_refuse_minus_jb_corr_aux", "harm_refuse", "jb_corr", +1),
        ("refuse", "or_corr_minus_benign_ans_safe", "or_corr", "benign_ans", +1),
    ]
    rows, warnings = [], []
    for metric, name, left, right, expected in specs:
        for layer in layers:
            a = proj[metric][left][layer]
            b = proj[metric][right][layer]
            obs = signed_mean_diff(a, b)
            boots = []
            for _ in range(n_boot):
                ai = rng.integers(0, len(a), len(a))
                bi = rng.integers(0, len(b), len(b))
                boots.append(float(a[ai].mean() - b[bi].mean()))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            bad = np.sign(obs) != np.sign(expected) and abs(obs) > 1e-8
            if bad:
                warnings.append(f"{name} L{layer}: expected positive, observed {obs:+.3f}")
            rows.append(
                {
                    "axis": metric,
                    "contrast": name,
                    "left": left,
                    "right": right,
                    "layer": layer,
                    "signed_standardized_mean_diff": obs,
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "expected_sign": expected,
                    "opposite_sign": bool(bad),
                }
            )
    return rows, warnings


def compute_axis_aurocs(train_by_group, eval_by_group, dirs, stats, layers):
    rows = []
    for metric in ("harm", "refuse"):
        layer_scores, layer_labels = defaultdict(list), defaultdict(list)
        for layer in layers:
            if metric == "harm":
                pos_groups, neg_groups = HARM_POS, SAFE_NEG
                v = layer_get(dirs["v_harm"], layer).float()
                train_pos = [float(user_mean_off(s, layer) @ v) for g in pos_groups for s in train_by_group[g]]
                train_neg = [float(user_mean_off(s, layer) @ v) for g in neg_groups for s in train_by_group[g]]
            else:
                pos_groups, neg_groups = REFUSED_POS, ANSWERED_NEG
                v = layer_get(dirs["v_refuse"], layer).float()
                train_pos = [float(final_off(s, layer) @ v) for g in pos_groups for s in train_by_group[g]]
                train_neg = [float(final_off(s, layer) @ v) for g in neg_groups for s in train_by_group[g]]
            sign = 1.0 if np.mean(train_pos) >= np.mean(train_neg) else -1.0
            scores, labels = [], []
            center, scale = stats[metric][layer]
            for group in GROUPS:
                label = 1 if group in pos_groups else 0
                for sample in eval_by_group[group]:
                    h = user_mean_off(sample, layer) if metric == "harm" else final_off(sample, layer)
                    score = sign * ((float(h @ v) - center) / scale)
                    scores.append(score)
                    labels.append(label)
                    layer_scores["agg"].append(score)
                    layer_labels["agg"].append(label)
            rows.append({"axis": metric, "layer": layer, "auroc": auroc_score(scores, labels), "sign": sign})
        rows.append({"axis": metric, "layer": "mean_ACT_LAYERS", "auroc": auroc_score(layer_scores["agg"], layer_labels["agg"]), "sign": ""})
    return rows


def sample_feature(sample, q, layer, source: str = "off"):
    if source == "off":
        hu, hf = user_mean_off(sample, layer), final_off(sample, layer)
    else:
        hu, hf = user_mean_on(sample, layer), final_on(sample, layer)
    return torch.cat([hu @ q.T, hf @ q.T]).numpy()


def balanced_xy(by_group, pos_groups, neg_groups, q_by_layer, layers, rng):
    xs_pos, xs_neg = [], []
    for group in pos_groups:
        for s in by_group[group]:
            xs_pos.append(np.concatenate([sample_feature(s, q_by_layer[layer], layer, "off") for layer in layers]))
    for group in neg_groups:
        for s in by_group[group]:
            xs_neg.append(np.concatenate([sample_feature(s, q_by_layer[layer], layer, "off") for layer in layers]))
    n = min(len(xs_pos), len(xs_neg))
    if n == 0:
        return np.zeros((0, 1)), np.zeros(0)
    pi = rng.choice(len(xs_pos), n, replace=False)
    ni = rng.choice(len(xs_neg), n, replace=False)
    x = np.stack([xs_pos[i] for i in pi] + [xs_neg[i] for i in ni])
    y = np.asarray([1] * n + [0] * n)
    return x, y


def compute_nested_subspace_auroc(train_by_group, eval_by_group, dirs, layers, random_n, seed):
    rng = np.random.default_rng(seed)
    tasks = {
        "jb_corr_vs_jb_blocked": ({"jb_corr"}, {"jb_blocked"}),
        "or_corr_vs_benign_ans": ({"or_corr"}, {"benign_ans"}),
        "correction_vs_preservation": (set(CORRECTION_GROUPS), set(PRESERVATION_GROUPS)),
    }
    kinds = ["Q1D_refuse", "Q2D_harm_refuse", "Q4D_variation", "Q5D_full"]
    rows = []
    hidden = int(layer_get(dirs["v_harm"], layers[0]).numel())
    for kind in kinds:
        q_by_layer = {layer: subspace_basis(dirs, layer, kind) for layer in layers}
        dim = int(sum(q_by_layer[layer].shape[0] * 2 for layer in layers))
        for task, (pos, neg) in tasks.items():
            xtr, ytr = balanced_xy(train_by_group, pos, neg, q_by_layer, layers, rng)
            xte, yte = balanced_xy(eval_by_group, pos, neg, q_by_layer, layers, rng)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                auc = math.nan
            else:
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="liblinear")
                clf.fit(xtr, ytr)
                auc = auroc_score(clf.decision_function(xte), yte)
            rows.append({"task": task, "subspace": kind, "feature_dim": dim, "auroc": auc, "random_seed": ""})
    for rseed in range(random_n):
        q_by_layer = {}
        gen = torch.Generator().manual_seed(seed + 1000 + rseed)
        for layer in layers:
            mat = torch.randn(hidden, 5, generator=gen)
            q, _ = torch.linalg.qr(mat)
            q_by_layer[layer] = q.T
        for task, (pos, neg) in tasks.items():
            xtr, ytr = balanced_xy(train_by_group, pos, neg, q_by_layer, layers, rng)
            xte, yte = balanced_xy(eval_by_group, pos, neg, q_by_layer, layers, rng)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="liblinear")
            clf.fit(xtr, ytr)
            auc = auroc_score(clf.decision_function(xte), yte)
            rows.append({"task": task, "subspace": "Random5D", "feature_dim": 50, "auroc": auc, "random_seed": rseed})
    return rows


def compute_q5d_rank_diagnostics(dirs, layers):
    rows, cos_by_layer = [], {}
    for layer in layers:
        mat = original_direction_matrix(dirs, layer).float()
        norms = mat.norm(dim=1).numpy()
        cos = (mat @ mat.T).numpy()
        cos_by_layer[layer] = cos
        u, s, vt = torch.linalg.svd(mat, full_matrices=False)
        sv = s.numpy()
        q = orthonormal_basis([mat[i] for i in range(mat.shape[0])])
        orth_err = float((q @ q.T - torch.eye(q.shape[0])).abs().max())
        cond = float(sv[0] / max(sv[-1], 1e-12))
        eff = {thr: int(np.sum(sv >= sv[0] * thr)) for thr in (1e-2, 1e-3, 1e-4)}
        for i, val in enumerate(sv):
            rows.append(
                {
                    "layer": layer,
                    "stat": "singular_value",
                    "index": i + 1,
                    "value": float(val),
                    "condition_number": cond,
                    "effective_rank_1e-2": eff[1e-2],
                    "effective_rank_1e-3": eff[1e-3],
                    "effective_rank_1e-4": eff[1e-4],
                    "orthogonality_error": orth_err,
                }
            )
        for i, name in enumerate(SEMANTIC_LABELS):
            rows.append({"layer": layer, "stat": "direction_norm", "index": name, "value": float(norms[i])})
        for i, a in enumerate(SEMANTIC_LABELS):
            for j, b in enumerate(SEMANTIC_LABELS):
                rows.append({"layer": layer, "stat": "cosine", "index": f"{a}|{b}", "value": float(cos[i, j])})
    return rows, cos_by_layer


def get_eval_samples(eval_off: dict, eval_emb_off: list[dict]) -> list[dict]:
    meta = eval_off.get("eval_samples")
    if not isinstance(meta, list):
        raise RuntimeError("eval_off.json missing eval_samples.")
    if len(meta) != len(eval_emb_off):
        raise RuntimeError(f"eval_off/eval_emb_off length mismatch: {len(meta)} vs {len(eval_emb_off)}")
    out = []
    for i, (m, e) in enumerate(zip(meta, eval_emb_off)):
        group = m.get("group", e.get("group"))
        out.append(
            {
                "sample_id": m.get("id", e.get("id", i)),
                "idx": m.get("idx", e.get("idx", i)),
                "group": group,
                "prompt": m.get("prompt", e.get("prompt")),
                "span": tuple(e["span"]),
                "last_idx": int(e["last_idx"]),
                "off": e,
            }
        )
    return out


def config_value(config, key, default):
    return config[key] if isinstance(config, dict) and key in config else default


def checkpoint_identifier(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "sha256_16": sha256_file(path)[:16]}


def build_adapter(root, checkpoint, subspace_cache, args):
    sys.path.insert(0, str(root))
    if args.artifact_profile == "nlp_team17":
        from src.nlp_team17_adapter import NLPTeam17Adapter as AdapterClass
        adapter_profile = "nlp_team17"
    else:
        from src.moe_v85_adapter import V85OnePassMoEAdapter as AdapterClass
        adapter_profile = "v85"

    cfg = checkpoint.get("config", {})
    act_layers = [int(x) for x in config_value(cfg, "ACT_LAYERS", checkpoint.get("act_layers", ACT_LAYERS_DEFAULT))]
    default_jb_scale = 1.5 if args.artifact_profile == "nlp_team17" else 1.0
    default_or_scale = 2.5 if args.artifact_profile == "nlp_team17" else OR_SCALE_DEFAULT
    params = {
        "act_layers": act_layers,
        "hidden": int(config_value(cfg, "HIDDEN", HIDDEN_DEFAULT)),
        "flow_rank": int(config_value(cfg, "FLOW_RANK", FLOW_RANK_DEFAULT)),
        "t_dim": T_DIM_DEFAULT,
        "history_n": int(config_value(cfg, "HISTORY_N", HISTORY_N_DEFAULT)),
        "gate_rank": int(config_value(cfg, "GATE_RANK", GATE_RANK_DEFAULT)),
        "gate_t_dim": GATE_T_DIM_DEFAULT,
        "abstain_threshold": float(config_value(cfg, "ABSTAIN_THRESHOLD", ABSTAIN_THRESHOLD_DEFAULT)),
        "jb_scale": float(args.jb_scale if args.jb_scale is not None else config_value(cfg, "JB_SCALE", default_jb_scale)),
        "or_scale": float(args.or_scale if args.or_scale is not None else config_value(cfg, "OR_SCALE", default_or_scale)),
        "adapter_profile": adapter_profile,
        "checkpoint_config": cfg,
    }
    subspace = q_basis_from_cache(subspace_cache)
    adapter = AdapterClass(
        params["hidden"],
        act_layers=act_layers,
        subspace_basis=subspace,
        rank=params["flow_rank"],
        t_dim=params["t_dim"],
        history_n=params["history_n"],
        gate_rank=params["gate_rank"],
        gate_t_dim=params["gate_t_dim"],
        abstain_threshold=params["abstain_threshold"],
        jb_scale=params["jb_scale"],
        or_scale=params["or_scale"],
    )
    msg = adapter.load_state_dict(checkpoint["state_dict"], strict=False)
    if hasattr(adapter, "set_steering_scales"):
        adapter.set_steering_scales(
            jb_scale=params["jb_scale"],
            or_scale=params["or_scale"],
            abstain_threshold=params["abstain_threshold"],
        )
    print(f"load_state_dict: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    print(f"adapter profile={adapter_profile} steering scales: jb={params['jb_scale']} or={params['or_scale']} abstain={params['abstain_threshold']}")
    return adapter, params


def offset_spans_and_encode(tokenizer, prompts, spans, last_idxs, device):
    from src.model import apply_chat

    chat = apply_chat(tokenizer, prompts, add_generation_prompt=True)
    texts = [c["text"] for c in chat]
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    attn = enc["attention_mask"]
    pad_offsets = (attn.shape[1] - attn.sum(dim=1)).cpu().tolist()
    tmax = int(attn.shape[1])
    spans_adj, last_adj = [], []
    for (s, e), li, off in zip(spans, last_idxs, pad_offsets):
        off = int(off)
        spans_adj.append((int(s) + off, min(int(e) + off, tmax)))
        last_adj.append(min(int(li) + off, tmax - 1))
    return enc.to(device), spans_adj, last_adj


def capture_matches(capture, ckpt_id, act_layers, args):
    meta = capture.get("metadata", {})
    cfg = meta.get("adapter_config", {})
    expected_jb = float(args.jb_scale if args.jb_scale is not None else (1.5 if args.artifact_profile == "nlp_team17" else 1.0))
    expected_or = float(args.or_scale if args.or_scale is not None else (2.5 if args.artifact_profile == "nlp_team17" else OR_SCALE_DEFAULT))
    return (
        capture.get("schema") == "v85_eval_repr_on_capture_detailed_v1"
        and meta.get("checkpoint_sha256_16") == ckpt_id["sha256_16"]
        and [int(x) for x in meta.get("act_layers", [])] == [int(x) for x in act_layers]
        and cfg.get("adapter_profile", "v85") == args.artifact_profile
        and abs(float(cfg.get("jb_scale", expected_jb)) - expected_jb) < 1e-8
        and abs(float(cfg.get("or_scale", expected_or)) - expected_or) < 1e-8
    )


def capture_detailed(root, artifacts, artifact_candidates, eval_samples, checkpoint, subspace_cache, args, ckpt_id):
    cap_path = artifacts.get("capture_detailed") or default_artifact_path(root, artifact_candidates, "capture_detailed")
    act_guess = checkpoint.get("config", {}).get("ACT_LAYERS", checkpoint.get("act_layers", ACT_LAYERS_DEFAULT))
    if cap_path.exists() and not args.force_recapture:
        cap = load_pt(cap_path)
        if capture_matches(cap, ckpt_id, act_guess, args):
            print(f"Reusing detailed capture: {cap_path}")
            return cap
        print("Detailed capture metadata mismatch; recapturing.")

    sys.path.insert(0, str(root))
    from src.model import load_model

    adapter, adapter_cfg = build_adapter(root, checkpoint, subspace_cache, args)
    model, tokenizer = load_model(args.model_id, dtype=args.dtype, device_map=args.device_map)
    device = model.device
    adapter.to(device).eval()
    act_layers = adapter_cfg["act_layers"]
    records = []

    prompts = [s["prompt"] for s in eval_samples]
    spans = [s["span"] for s in eval_samples]
    last_idxs = [s["last_idx"] for s in eval_samples]
    off_records = [s["off"] for s in eval_samples]
    bs = max(1, args.batch_size)
    for start in range(0, len(eval_samples), bs):
        end = min(start + bs, len(eval_samples))
        enc, spans_adj, last_adj = offset_spans_and_encode(
            tokenizer, prompts[start:end], spans[start:end], last_idxs[start:end], device
        )
        with adapter.steer(model, spans_adj, last_adj, train_mode=False, capture=True) as state:
            with torch.inference_mode():
                model.model(**enc, use_cache=False)
        for bi, sample in enumerate(eval_samples[start:end]):
            s_adj, e_adj = spans_adj[bi]
            li = last_adj[bi]
            out = {
                "sample_id": sample["sample_id"],
                "idx": sample["idx"],
                "group": sample["group"],
                "span_original": sample["span"],
                "last_idx_original": sample["last_idx"],
                "span_adjusted": (int(s_adj), int(e_adj)),
                "last_idx_adjusted": int(li),
                "span_h_off": {},
                "last_h_off": {},
                "span_h_corr": {},
                "last_h_corr": {},
                "delta_jb_span": {},
                "delta_jb_last": {},
                "delta_or_span": {},
                "delta_or_last": {},
                "applied_jb_span": {},
                "applied_jb_last": {},
                "applied_or_span": {},
                "applied_or_last": {},
                "p_jb_raw": {},
                "p_or_raw": {},
                "p_jb_eff": {},
                "p_or_eff": {},
            }
            off = off_records[start + bi]
            for layer in act_layers:
                rec = state["records"][layer]
                h_corr = rec["h_corr"][bi]
                d_jb = rec["d_jb"][bi]
                d_or = rec["d_or"][bi]
                pjb = float(rec["p_jb"][bi].detach().float().cpu())
                por = float(rec["p_or"][bi].detach().float().cpu())
                out["span_h_off"][layer] = user_mean_off(off, layer).cpu().to(torch.bfloat16)
                out["last_h_off"][layer] = final_off(off, layer).cpu().to(torch.bfloat16)
                out["span_h_corr"][layer] = h_corr[s_adj:e_adj].float().mean(0).detach().cpu().to(torch.bfloat16)
                out["last_h_corr"][layer] = h_corr[li].float().detach().cpu().to(torch.bfloat16)
                out["delta_jb_span"][layer] = d_jb[s_adj:e_adj].float().mean(0).detach().cpu().to(torch.bfloat16)
                out["delta_jb_last"][layer] = d_jb[li].float().detach().cpu().to(torch.bfloat16)
                out["delta_or_span"][layer] = d_or[s_adj:e_adj].float().mean(0).detach().cpu().to(torch.bfloat16)
                out["delta_or_last"][layer] = d_or[li].float().detach().cpu().to(torch.bfloat16)
                out["applied_jb_span"][layer] = (pjb * d_jb[s_adj:e_adj].float().mean(0).detach().cpu()).to(torch.bfloat16)
                out["applied_jb_last"][layer] = (pjb * d_jb[li].float().detach().cpu()).to(torch.bfloat16)
                out["applied_or_span"][layer] = (por * d_or[s_adj:e_adj].float().mean(0).detach().cpu()).to(torch.bfloat16)
                out["applied_or_last"][layer] = (por * d_or[li].float().detach().cpu()).to(torch.bfloat16)
                out["p_jb_raw"][layer] = float(rec["p_jb_raw"][bi].detach().float().cpu())
                out["p_or_raw"][layer] = float(rec["p_or_raw"][bi].detach().float().cpu())
                out["p_jb_eff"][layer] = pjb
                out["p_or_eff"][layer] = por
            records.append(out)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    capture = {
        "schema": "v85_eval_repr_on_capture_detailed_v1",
        "metadata": {
            "checkpoint_path": str(artifacts["adapter"]),
            "checkpoint_sha256_16": ckpt_id["sha256_16"],
            "act_layers": act_layers,
            "steer_span_offset": "left_padding_batch_v1",
            "adapter_config": adapter_cfg,
        },
        "records": records,
    }
    save_pt(cap_path, capture)
    print(f"Saved detailed capture: {cap_path}")
    return capture


def records_from_eval_emb_on(eval_samples: list[dict], eval_emb_on: list[dict], layers: list[int]) -> list[dict]:
    if len(eval_samples) != len(eval_emb_on):
        raise RuntimeError(f"eval_emb_on length mismatch: {len(eval_emb_on)} vs eval samples {len(eval_samples)}")
    records = []
    for i, (sample, on) in enumerate(zip(eval_samples, eval_emb_on)):
        sid = on.get("id", on.get("sample_id", sample["sample_id"]))
        if sid != sample["sample_id"]:
            print(f"WARNING: sample id mismatch at {i}: eval_off={sample['sample_id']} eval_emb_on={sid}; using eval_off id")
        group = on.get("group", sample["group"])
        rec = {
            "sample_id": sample["sample_id"],
            "idx": on.get("idx", sample["idx"]),
            "group": group,
            "span_original": sample["span"],
            "last_idx_original": sample["last_idx"],
            "span_h_off": {},
            "last_h_off": {},
            "span_h_corr": {},
            "last_h_corr": {},
            "cache_source": "eval_emb_on",
        }
        for layer in layers:
            span_on = layer_get(on["span_h"], layer).float()
            rec["span_h_off"][layer] = user_mean_off(sample["off"], layer).cpu().to(torch.bfloat16)
            rec["last_h_off"][layer] = final_off(sample["off"], layer).cpu().to(torch.bfloat16)
            rec["span_h_corr"][layer] = (span_on.mean(0) if span_on.dim() > 1 else span_on).cpu().to(torch.bfloat16)
            rec["last_h_corr"][layer] = layer_get(on["last_h"], layer).float().cpu().to(torch.bfloat16)
        records.append(rec)
    return records


def coerce_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "refuse", "refused", "1"}:
            return True
        if lowered in {"false", "no", "answer", "answered", "0"}:
            return False
    return None


def behavior_label_from_wg(item):
    if isinstance(item, dict):
        for key in ("refusal", "refuse", "is_refusal", "refused"):
            if key in item:
                return coerce_optional_bool(item[key])
    return coerce_optional_bool(item)


def attach_eval_on_behavior(capture_records, eval_on):
    if not eval_on:
        return {}
    meta = eval_on.get("eval_samples", [])
    wg = eval_on.get("wg", [])
    out = {}
    for i, item in enumerate(wg):
        sid = meta[i].get("id", i) if i < len(meta) and isinstance(meta[i], dict) else i
        out[sid] = behavior_label_from_wg(item)
    matched = 0
    labeled = 0
    for rec in capture_records:
        if rec["sample_id"] in out:
            matched += 1
        rec["adapter_on_refused"] = out.get(rec["sample_id"])
        if rec["adapter_on_refused"] is not None:
            labeled += 1
    print(f"Attached eval_on refusal labels: {labeled}/{len(capture_records)} labeled, {matched}/{len(capture_records)} matched by sample_id")
    if labeled == 0:
        print("WARNING: eval_on.json was found, but no refusal labels were parsed. Check eval_on['wg'] keys and values.")
    return out


def desirable_success(group: str, refused: bool | None):
    if refused is None:
        return None
    if group in {"jb_corr", "jb_blocked", "harm_refuse"}:
        return bool(refused)
    return not bool(refused)


def transition_name(group: str, refused: bool | None):
    if refused is None and group in {"jb_corr", "or_corr"}:
        return f"{group}_unknown"
    if group == "jb_corr":
        return "jb_corr_to_refused" if refused is True else "jb_corr_still_answered"
    if group == "or_corr":
        return "or_corr_still_refused" if refused is True else "or_corr_to_answered"
    return group


def q_coords(sample, q_basis, layer, source):
    q = layer_get(q_basis, layer).float()
    if source == "off":
        h = user_mean_off(sample, layer) if "span_h" in sample else layer_get(sample["span_h_off"], layer).float()
    else:
        h = user_mean_on(sample, layer)
    return h @ q.T


def q_std(targets, group, layer):
    return torch.clamp(torch.as_tensor(layer_get(targets["g5_std"][group], layer), dtype=torch.float32), min=1.0)


def jb_target_dist(sample, targets, q_basis, layers, weights, source):
    vals = []
    for w, layer in zip(weights, layers):
        coord = q_coords(sample, q_basis, layer, source)
        tgt = torch.as_tensor(layer_get(targets["g5_mean"]["harm_refuse"], layer), dtype=torch.float32)
        std = q_std(targets, "harm_refuse", layer)
        vals.append(float(w) * float((((coord - tgt) / std) ** 2).mean()))
    return math.sqrt(max(sum(vals), 0.0))


def or_target_dist(sample, dirs, targets, layers, weights, source):
    vals = []
    for w, layer in zip(weights, layers):
        vr = layer_get(dirs["v_refuse"], layer).float()
        h = final_off(sample, layer) if source == "off" and "last_h" in sample else (
            layer_get(sample["last_h_off"], layer).float() if source == "off" else final_on(sample, layer)
        )
        tgt = layer_get(targets["or_vr_answered"], layer)
        vals.append(float(w) * abs((float(h @ vr) - float(tgt["mean"])) / max(float(tgt["std"]), 1.0)))
    return float(sum(vals))


def q5d_disp_norm(off_sample, on_sample, targets, q_basis, layers, weights):
    group = on_sample["group"]
    vals = []
    for w, layer in zip(weights, layers):
        off_c = q_coords(off_sample, q_basis, layer, "off")
        on_c = q_coords(on_sample, q_basis, layer, "on")
        vals.append(float(w) * float((((on_c - off_c) / q_std(targets, group, layer)) ** 2).mean()))
    return math.sqrt(max(sum(vals), 0.0))


def vector_norm(mapping, layer):
    return float(layer_get(mapping, layer).float().norm())


def applied_update_rows(records, layers):
    rows = []
    for rec in records:
        for layer in layers:
            jb = vector_norm(rec["applied_jb_span"], layer)
            orr = vector_norm(rec["applied_or_last"], layer)
            total_span = layer_get(rec["applied_jb_span"], layer).float() + layer_get(rec["applied_or_span"], layer).float()
            rows.append(
                {
                    "group": rec["group"],
                    "sample_id": rec["sample_id"],
                    "layer": layer,
                    "p_jb_raw": rec["p_jb_raw"][layer],
                    "p_or_raw": rec["p_or_raw"][layer],
                    "p_jb_eff": rec["p_jb_eff"][layer],
                    "p_or_eff": rec["p_or_eff"][layer],
                    "delta_jb_norm": vector_norm(rec["delta_jb_span"], layer),
                    "delta_or_norm": vector_norm(rec["delta_or_last"], layer),
                    "applied_jb_update_norm": jb,
                    "applied_or_update_norm": orr,
                    "applied_total_update_norm": float(total_span.norm()),
                }
            )
    return rows


def summarize_by_group_layer(rows, value_keys, n_boot, seed):
    rng = np.random.default_rng(seed)
    out = []
    for group in GROUPS:
        for layer in sorted({r["layer"] for r in rows}):
            sub = [r for r in rows if r["group"] == group and r["layer"] == layer]
            for key in value_keys:
                vals = [r[key] for r in sub]
                mean, lo, hi = bootstrap_mean_ci(vals, rng, n_boot)
                out.append({"group": group, "layer": layer, "metric": key, "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)})
    return out


def success_rate_rows(records, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for group in GROUPS:
        vals = []
        for rec in records:
            if rec["group"] != group:
                continue
            ok = desirable_success(group, rec.get("adapter_on_refused"))
            if ok is not None:
                vals.append(float(ok))
        mean, lo, hi = bootstrap_mean_ci(vals, rng, n_boot)
        rows.append({"group": group, "n": len(vals), "success_rate": mean, "ci_low": lo, "ci_high": hi})
    return rows


def transition_alignment_rows(off_by_id, records, dirs, targets, q_basis, layers, weights, n_boot, seed):
    rng = np.random.default_rng(seed)
    sample_rows = []
    for rec in records:
        group = rec["group"]
        if group not in {"jb_corr", "or_corr"}:
            continue
        off = off_by_id[rec["sample_id"]]
        if group == "jb_corr":
            d_off = jb_target_dist(off, targets, q_basis, layers, weights, "off")
            d_on = jb_target_dist(rec, targets, q_basis, layers, weights, "on")
            objective = "jb_to_harm_refuse_q5d"
        else:
            d_off = or_target_dist(off, dirs, targets, layers, weights, "off")
            d_on = or_target_dist(rec, dirs, targets, layers, weights, "on")
            objective = "or_to_answered_refusal"
        tr = transition_name(group, rec.get("adapter_on_refused"))
        sample_rows.append(
            {
                "sample_id": rec["sample_id"],
                "group": group,
                "transition": tr,
                "objective": objective,
                "distance_off": d_off,
                "distance_on": d_on,
                "delta_target_distance": d_on - d_off,
            }
        )
    summary = []
    for tr in ["jb_corr_to_refused", "jb_corr_still_answered", "or_corr_to_answered", "or_corr_still_refused"]:
        vals = [r["delta_target_distance"] for r in sample_rows if r["transition"] == tr]
        mean, lo, hi = bootstrap_mean_ci(vals, rng, n_boot)
        summary.append({"transition": tr, "metric": "delta_target_distance", "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)})
    return sample_rows, summary


def displacement_rows(off_by_id, records, targets, q_basis, layers, weights, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for rec in records:
        off = off_by_id[rec["sample_id"]]
        rows.append(
            {
                "sample_id": rec["sample_id"],
                "group": rec["group"],
                "transition": transition_name(rec["group"], rec.get("adapter_on_refused")),
                "q5d_displacement_norm": q5d_disp_norm(off, rec, targets, q_basis, layers, weights),
            }
        )
    summary = []
    for group in GROUPS:
        vals = [r["q5d_displacement_norm"] for r in rows if r["group"] == group]
        mean, lo, hi = bootstrap_mean_ci(vals, rng, n_boot)
        summary.append({"group": group, "metric": "q5d_displacement_norm", "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)})
    return rows, summary


def behavior_delta_rows(off_scores_by_id, on_scores_by_id, records):
    rows = []
    for rec in records:
        sid = rec["sample_id"]
        off = off_scores_by_id[sid]
        on = on_scores_by_id[sid]
        rows.append(
            {
                "sample_id": sid,
                "group": rec["group"],
                "transition": transition_name(rec["group"], rec.get("adapter_on_refused")),
                "x_off": float(off[0]),
                "y_off": float(off[1]),
                "x_on": float(on[0]),
                "y_on": float(on[1]),
                "delta_harm": float(on[0] - off[0]),
                "delta_refuse": float(on[1] - off[1]),
            }
        )
    return rows


def semantic_direction_rows(off_by_id, records, dirs, layers, weights, n_boot, seed):
    rng = np.random.default_rng(seed)
    sample_rows = []
    for rec in records:
        off = off_by_id[rec["sample_id"]]
        tr = transition_name(rec["group"], rec.get("adapter_on_refused"))
        for dkey, dlabel in zip(SEMANTIC_DIRS, SEMANTIC_LABELS):
            vals = []
            for w, layer in zip(weights, layers):
                v = unit(layer_get(dirs[dkey], layer))
                disp_user = user_mean_on(rec, layer) - user_mean_off(off, layer)
                disp_last = final_on(rec, layer) - final_off(off, layer)
                vals.append(float(w) * 0.5 * (float(disp_user @ v) + float(disp_last @ v)))
            sample_rows.append(
                {
                    "sample_id": rec["sample_id"],
                    "group": rec["group"],
                    "transition": tr,
                    "direction": dlabel,
                    "layer_weighted_displacement": float(sum(vals)),
                }
            )
    summary = []
    for tr in TRANSITION_ORDER:
        for dlabel in SEMANTIC_LABELS:
            vals = [r["layer_weighted_displacement"] for r in sample_rows if r["transition"] == tr and r["direction"] == dlabel]
            mean, lo, hi = bootstrap_mean_ci(vals, rng, n_boot)
            summary.append({"transition": tr, "direction": dlabel, "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)})
    return sample_rows, summary


def jb_variance_rows(off_by_id, records, behavior_delta, q_basis, targets, layers, weights):
    rows = []
    subsets = {
        "jb_corr_all": [r for r in records if r["group"] == "jb_corr"],
        "jb_corr_to_refused": [r for r in records if transition_name(r["group"], r.get("adapter_on_refused")) == "jb_corr_to_refused"],
        "jb_corr_still_answered": [r for r in records if transition_name(r["group"], r.get("adapter_on_refused")) == "jb_corr_still_answered"],
    }
    delta_by_id = {r["sample_id"]: r for r in behavior_delta}
    for subset, recs in subsets.items():
        if not recs:
            continue
        harm_off = np.asarray([delta_by_id[r["sample_id"]]["x_off"] for r in recs])
        harm_on = np.asarray([delta_by_id[r["sample_id"]]["x_on"] for r in recs])
        ref_off = np.asarray([delta_by_id[r["sample_id"]]["y_off"] for r in recs])
        ref_on = np.asarray([delta_by_id[r["sample_id"]]["y_on"] for r in recs])
        q_off, q_on, dist_off, dist_on = [], [], [], []
        for r in recs:
            off = off_by_id[r["sample_id"]]
            coords_off, coords_on = [], []
            for layer in layers:
                coords_off.append(q_coords(off, q_basis, layer, "off").numpy())
                coords_on.append(q_coords(r, q_basis, layer, "on").numpy())
            q_off.append(np.concatenate(coords_off))
            q_on.append(np.concatenate(coords_on))
            dist_off.append(jb_target_dist(off, targets, q_basis, layers, weights, "off"))
            dist_on.append(jb_target_dist(r, targets, q_basis, layers, weights, "on"))
        for metric, off_vals, on_vals in [
            ("harm_projection_variance", harm_off, harm_on),
            ("refuse_projection_variance", ref_off, ref_on),
            ("q5d_covariance_trace", np.asarray(q_off), np.asarray(q_on)),
            ("q5d_target_distance_variance", np.asarray(dist_off), np.asarray(dist_on)),
        ]:
            if metric == "q5d_covariance_trace":
                vo = float(np.trace(np.cov(off_vals.T))) if len(off_vals) > 1 else 0.0
                vn = float(np.trace(np.cov(on_vals.T))) if len(on_vals) > 1 else 0.0
            else:
                vo = float(np.var(off_vals, ddof=1)) if len(off_vals) > 1 else 0.0
                vn = float(np.var(on_vals, ddof=1)) if len(on_vals) > 1 else 0.0
            rows.append({"subset": subset, "metric": metric, "n": len(recs), "off": vo, "on": vn, "delta": vn - vo})
    return rows


def save_fig_pair(fig, pdf: Path, png: Path):
    paths = []
    if SAVE_PDF:
        fig.savefig(pdf, bbox_inches="tight")
        paths.append(pdf)
    fig.savefig(png, bbox_inches="tight", dpi=OUTPUT_DPI)
    paths.append(png)
    plt.close(fig)
    return paths


def group_from_display_name(name: str) -> str | None:
    name = str(name)
    if name in TRANSITION_DISPLAY:
        name = TRANSITION_DISPLAY[name]
    if "\n" in name:
        head = name.split("\n", 1)[0]
        if head in COLORS:
            return head
    if name in COLORS:
        return name
    for group in GROUPS:
        if name.startswith(f"{group}_"):
            return group
    return None


def color_for_display_name(name: str, default: str = "0.35") -> str:
    group = group_from_display_name(str(name))
    if group is None:
        return default
    return COLORS[group]


def color_group_ticklabels(ax, *, axis: str = "x") -> None:
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for label in labels:
        group = group_from_display_name(label.get_text())
        if group is not None:
            label.set_color(COLORS[group])


def display_transition_or_group(name: str) -> str:
    return TRANSITION_DISPLAY.get(str(name), str(name))


def random5d_summary(nested_rows, task: str) -> dict:
    vals = np.asarray([r["auroc"] for r in nested_rows if r["task"] == task and r["subspace"] == "Random5D"], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(vals.size)}


def nested_auroc_value(nested_rows, task: str, subspace: str) -> float:
    vals = [r["auroc"] for r in nested_rows if r["task"] == task and r["subspace"] == subspace]
    return float(np.mean(vals)) if vals else math.nan


def plot_behavior_geometry_panel(ax, behavior_scores, n_boot, seed, *, title, xlabel, ylabel, show_legend: bool):
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(seed)
    for group in PLOT_GROUPS:
        pts = behavior_scores[group]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=10,
            alpha=0.15,
            color=COLORS[group],
            linewidths=0,
            rasterized=True,
            label=DISPLAY_GROUP[group],
        )
        c, lo, hi = bootstrap_centroid_ci(pts, rng, n_boot)
        ax.errorbar(
            c[0],
            c[1],
            xerr=[[c[0] - lo[0]], [hi[0] - c[0]]],
            yerr=[[c[1] - lo[1]], [hi[1] - c[1]]],
            fmt="o",
            ms=5.6,
            color=COLORS[group],
            capsize=2,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax.annotate(DISPLAY_GROUP[group], c, xytext=(4, 3), textcoords="offset points", fontsize=ANNOTATION_SIZE, color=COLORS[group])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="0.92", lw=0.5)
    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=COLORS[group],
                markeredgecolor=COLORS[group],
                markersize=5.5,
                alpha=1.0,
                label=DISPLAY_GROUP[group],
            )
            for group in PLOT_GROUPS
        ]
        ax.legend(handles=handles, frameon=False, loc="lower right", handletextpad=0.4, borderaxespad=0.25)


def plot_contrast_panel(ax, contrast_rows, contrast_names, *, title, ylabel, legend_loc="lower left", legend_bbox=None):
    for name in contrast_names:
        sub = [r for r in contrast_rows if r["contrast"] == name]
        if not sub:
            continue
        xs = [r["layer"] for r in sub]
        ys = [r["signed_standardized_mean_diff"] for r in sub]
        lo = [r["ci_low"] for r in sub]
        hi = [r["ci_high"] for r in sub]
        color = CONTRAST_COLORS.get(name, "0.3")
        ax.plot(xs, ys, marker="o", lw=1.5, color=color, label=CONTRAST_LABELS.get(name, name))
        ax.fill_between(xs, lo, hi, alpha=0.14, color=color)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Intervention layer")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc=legend_loc, bbox_to_anchor=legend_bbox)
    ax.grid(color="0.93", lw=0.5)


def plot_nested_main_panel(ax, nested_rows):
    task = "correction_vs_preservation"
    x = np.arange(len(MAIN_SUBSPACE_ORDER))
    vals = [nested_auroc_value(nested_rows, task, subspace) for subspace in MAIN_SUBSPACE_ORDER]
    color = TASK_COLORS[task]
    rand = random5d_summary(nested_rows, task)
    if rand["n"]:
        ax.fill_between(
            [-0.15, len(MAIN_SUBSPACE_ORDER) - 0.85],
            [rand["ci_low"], rand["ci_low"]],
            [rand["ci_high"], rand["ci_high"]],
            color="0.6",
            alpha=0.18,
            label="Random 5D",
            zorder=0,
        )
        ax.axhline(rand["mean"], color="0.45", lw=1.0, ls="--", zorder=1)
    ax.plot(x, vals, marker="o", lw=1.8, color=color, label="Nested subspace probe", zorder=2)
    for xi, val in zip(x, vals):
        if np.isfinite(val):
            ax.annotate(f"{val:.3f}", (xi, val), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=ANNOTATION_SIZE)
    ax.set_xticks(x, [MAIN_SUBSPACE_LABELS[s] for s in MAIN_SUBSPACE_ORDER], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Probe AUROC")
    ax.set_title("Correction-need separability in nested subspaces", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(color="0.93", lw=0.5)


def plot_nested_appendix_panel(ax, nested_rows):
    x = np.arange(len(MAIN_SUBSPACE_ORDER))
    tasks = ["jb_corr_vs_jb_blocked", "or_corr_vs_benign_ans", "correction_vs_preservation"]
    for task in tasks:
        vals = [nested_auroc_value(nested_rows, task, subspace) for subspace in MAIN_SUBSPACE_ORDER]
        color = TASK_COLORS.get(task, "0.3")
        ax.plot(x, vals, marker="o", lw=1.5, color=color, label=TASK_LABELS.get(task, task), zorder=2)
        rand = random5d_summary(nested_rows, task)
        if rand["n"]:
            ax.fill_between(
                [-0.15, len(MAIN_SUBSPACE_ORDER) - 0.85],
                [rand["ci_low"], rand["ci_low"]],
                [rand["ci_high"], rand["ci_high"]],
                color=color,
                alpha=0.08,
                zorder=0,
            )
            ax.axhline(rand["mean"], color=color, lw=0.9, ls="--", alpha=0.75, zorder=1)
    ax.set_xticks(x, [MAIN_SUBSPACE_LABELS[s] for s in MAIN_SUBSPACE_ORDER], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Probe AUROC")
    ax.set_title("Nested subspace probe tasks", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(color="0.93", lw=0.5)


def representation_main_values(behavior_scores, contrast_rows, nested_rows, eval_by_group):
    panel_b_contrasts = ["jb_blocked_minus_jb_corr_jailbreak", "or_corr_minus_benign_ans_safe"]
    rows = []
    for group in GROUPS:
        rows.append({"panel": "A", "metric": "sample_count", "group": group, "value": len(eval_by_group[group]), "split": "held_out_eval_off"})
    for name in panel_b_contrasts:
        for r in [row for row in contrast_rows if row["contrast"] == name]:
            rows.append(
                {
                    "panel": "B",
                    "metric": "refusal_projection_difference",
                    "contrast": CONTRAST_LABELS[name],
                    "layer": r["layer"],
                    "mean": r["signed_standardized_mean_diff"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                }
            )
    task = "correction_vs_preservation"
    for subspace in MAIN_SUBSPACE_ORDER:
        rows.append(
            {
                "panel": "C",
                "metric": "probe_auroc",
                "task": task,
                "subspace": subspace,
                "label": MAIN_SUBSPACE_LABELS[subspace],
                "auroc": nested_auroc_value(nested_rows, task, subspace),
            }
        )
    rand = random5d_summary(nested_rows, task)
    rows.append(
        {
            "panel": "C",
            "metric": "random5d_control",
            "task": task,
            "subspace": "Random5D",
            "label": "Random 5D",
            "mean": rand["mean"],
            "ci_low": rand["ci_low"],
            "ci_high": rand["ci_high"],
            "n": rand["n"],
        }
    )
    json_obj = {
        "panel_a": {
            "split": "held_out_eval_off",
            "sample_count_by_group": {group: len(eval_by_group[group]) for group in GROUPS},
            "groups": GROUPS,
        },
        "panel_b": {
            "contrasts": {
                CONTRAST_LABELS[name]: [r for r in contrast_rows if r["contrast"] == name]
                for name in panel_b_contrasts
            }
        },
        "panel_c": {
            "task": "correction_vs_preservation",
            "positive_class": ["jb_corr", "or_corr"],
            "negative_class": ["jb_blocked", "harm_refuse", "benign_ans"],
            "subspace_auroc": {
                subspace: nested_auroc_value(nested_rows, task, subspace)
                for subspace in MAIN_SUBSPACE_ORDER
            },
            "random5d": rand,
        },
    }
    return rows, json_obj


def fig_representation_analysis_main(out_dir, behavior_scores, contrast_rows, nested_rows, eval_by_group, n_boot, seed):
    fig = plt.figure(figsize=FIGSIZE_MAIN_3PANEL, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.32, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    plot_behavior_geometry_panel(
        ax_a,
        behavior_scores,
        n_boot,
        seed,
        title="Base-model behavior geometry",
        xlabel="Harmfulness-related projection",
        ylabel="Refusal-related projection",
        show_legend=True,
    )
    plot_contrast_panel(
        ax_b,
        contrast_rows,
        ["jb_blocked_minus_jb_corr_jailbreak", "or_corr_minus_benign_ans_safe"],
        title="Refusal-related separation across layers",
        ylabel="Difference in refusal-related projection",
        legend_loc="lower center",
        legend_bbox=(0.5, 0.03),
    )
    plot_nested_main_panel(ax_c, nested_rows)
    pdf = out_dir / "fig_representation_analysis_main.pdf"
    png = out_dir / "fig_representation_analysis_main.png"
    paths = save_fig_pair(fig, pdf, png)
    rows, json_obj = representation_main_values(behavior_scores, contrast_rows, nested_rows, eval_by_group)
    csv_path = write_csv(out_dir / "fig_representation_analysis_main_values.csv", rows)
    json_path = out_dir / "fig_representation_analysis_main_values.json"
    save_json(json_path, json_obj)
    return paths + [csv_path, json_path], json_obj


def fig_representation_analysis_appendix(out_dir, contrast_rows, nested_rows):
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_APPENDIX_3PANEL, constrained_layout=True)
    harm_names = [
        "jb_corr_minus_benign_ans_answered",
        "harm_refuse_minus_or_corr_refused",
        "jb_blocked_minus_or_corr_aux",
    ]
    refuse_names = [
        "jb_blocked_minus_jb_corr_jailbreak",
        "harm_refuse_minus_jb_corr_aux",
        "or_corr_minus_benign_ans_safe",
    ]
    plot_contrast_panel(
        axes[0],
        [r for r in contrast_rows if r["axis"] == "harm"],
        harm_names,
        title="Harmfulness-related conditional contrasts",
        ylabel="Difference in harmfulness-related projection",
    )
    plot_contrast_panel(
        axes[1],
        [r for r in contrast_rows if r["axis"] == "refuse"],
        refuse_names,
        title="Refusal-related conditional contrasts",
        ylabel="Difference in refusal-related projection",
    )
    plot_nested_appendix_panel(axes[2], nested_rows)
    pdf = out_dir / "fig_representation_analysis_appendix.pdf"
    png = out_dir / "fig_representation_analysis_appendix.png"
    return save_fig_pair(fig, pdf, png)


def fig1_base_map(out_dir, behavior_scores, n_boot, seed):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)
    for group in PLOT_GROUPS:
        pts = behavior_scores[group]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=12,
            alpha=0.16,
            color=COLORS[group],
            linewidths=0,
            rasterized=True,
            label=DISPLAY_GROUP[group],
        )
        c, lo, hi = bootstrap_centroid_ci(pts, rng, n_boot)
        ax.errorbar(
            c[0],
            c[1],
            xerr=[[c[0] - lo[0]], [hi[0] - c[0]]],
            yerr=[[c[1] - lo[1]], [hi[1] - c[1]]],
            fmt="o",
            ms=6.2,
            color=COLORS[group],
            capsize=2,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax.annotate(DISPLAY_GROUP[group], c, xytext=(5, 4), textcoords="offset points", fontsize=ANNOTATION_SIZE, color=COLORS[group])
    ax.set_title("Base model behavior map")
    ax.set_xlabel("Harmfulness-related projection")
    ax.set_ylabel("Refusal-related projection")
    ax.grid(color="0.92", lw=0.5)
    ax.legend(frameon=False, loc="best")
    pdf = out_dir / "fig_1a_base_model_behavior_map.pdf"
    png = out_dir / "fig_1a_base_model_behavior_map.png"
    return save_fig_pair(fig, pdf, png)


def fig1_direction_validity(out_dir, contrast_rows):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True, constrained_layout=True)
    panel_specs = [
        (axes[0], "harm", "Harmfulness projection validity"),
        (axes[1], "refuse", "Refusal projection validity"),
    ]
    for ax, metric, title in panel_specs:
        rows = [r for r in contrast_rows if r["axis"] == metric]
        names = [name for name in CONTRAST_LABELS if any(r["contrast"] == name for r in rows)]
        for name in names:
            sub = [r for r in rows if r["contrast"] == name]
            xs = [r["layer"] for r in sub]
            ys = [r["signed_standardized_mean_diff"] for r in sub]
            lo = [r["ci_low"] for r in sub]
            hi = [r["ci_high"] for r in sub]
            color = CONTRAST_COLORS.get(name, "0.3")
            ax.plot(xs, ys, marker="o", lw=1.5, color=color, label=CONTRAST_LABELS.get(name, name))
            ax.fill_between(xs, lo, hi, alpha=0.14, color=color)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Intervention layer")
        ax.set_ylabel("Mean projection difference (standardized)")
        ax.legend(frameon=False)
        ax.grid(color="0.93", lw=0.5)
    pdf = out_dir / "fig_1b_behavior_direction_validity.pdf"
    png = out_dir / "fig_1b_behavior_direction_validity.png"
    return save_fig_pair(fig, pdf, png)


def fig1_nested_subspace(out_dir, nested_rows):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)
    tasks = sorted({r["task"] for r in nested_rows})
    x = np.arange(len(SUBSPACE_ORDER))
    for ti, task in enumerate(tasks):
        vals = []
        for subspace in SUBSPACE_ORDER:
            sub_vals = [r["auroc"] for r in nested_rows if r["task"] == task and r["subspace"] == subspace]
            vals.append(float(np.mean(sub_vals)) if sub_vals else np.nan)
        color = TASK_COLORS.get(task, f"C{ti}")
        ax.plot(x, vals, marker="o", lw=1.6, color=color, label=TASK_LABELS.get(task, task))
    ax.set_xticks(x, [SUBSPACE_LABELS[s] for s in SUBSPACE_ORDER], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Classification AUROC")
    ax.set_title("Nested subspace probe performance")
    ax.legend(frameon=False)
    ax.grid(color="0.93", lw=0.5)
    pdf = out_dir / "fig_1c_nested_subspace_probe_auroc.pdf"
    png = out_dir / "fig_1c_nested_subspace_probe_auroc.png"
    return save_fig_pair(fig, pdf, png)


def fig1(out_dir, behavior_scores, contrast_rows, nested_rows, n_boot, seed):
    paths = []
    paths += fig1_base_map(out_dir, behavior_scores, n_boot, seed)
    paths += fig1_direction_validity(out_dir, contrast_rows)
    paths += fig1_nested_subspace(out_dir, nested_rows)
    return paths


def fig_rank(out_dir, diag_rows, cos_by_layer, layers):
    fig = plt.figure(figsize=FIGSIZE_GRID, constrained_layout=True)
    gs = fig.add_gridspec(2, max(3, len(layers)))
    ax_s = fig.add_subplot(gs[0, :2])
    ax_r = fig.add_subplot(gs[0, 2:])
    for layer in layers:
        sv = [r["value"] for r in diag_rows if r["layer"] == layer and r["stat"] == "singular_value"]
        ax_s.plot(range(1, len(sv) + 1), sv, marker="o", label=f"L{layer}")
    ax_s.set_title("Q5D direction singular values")
    ax_s.set_xlabel("Index")
    ax_s.set_ylabel("Singular value")
    ax_s.legend(frameon=False)
    ranks = []
    for layer in layers:
        row = next(r for r in diag_rows if r["layer"] == layer and r["stat"] == "singular_value" and r["index"] == 1)
        ranks.append([row["effective_rank_1e-2"], row["effective_rank_1e-3"], row["effective_rank_1e-4"]])
    im = ax_r.imshow(np.asarray(ranks).T, aspect="auto", vmin=0, vmax=5, cmap="viridis")
    ax_r.set_yticks(range(3), ["1e-2", "1e-3", "1e-4"])
    ax_r.set_xticks(range(len(layers)), [str(l) for l in layers])
    ax_r.set_title("Effective rank by threshold")
    fig.colorbar(im, ax=ax_r, fraction=0.03, pad=0.02)
    for i, layer in enumerate(layers):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(cos_by_layer[layer], vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"L{layer} cosine")
        ax.set_xticks(range(5), SEMANTIC_LABELS, rotation=45, ha="right")
        ax.set_yticks(range(5), SEMANTIC_LABELS)
    pdf, png = out_dir / "fig_q5d_rank_and_direction_diagnostics.pdf", out_dir / "fig_q5d_rank_and_direction_diagnostics.png"
    return save_fig_pair(fig, pdf, png)


def boxplot_with_points(ax, values_by_name, title, ylabel):
    names = [n for n in values_by_name if len(values_by_name[n]) > 0]
    vals = [np.asarray(values_by_name[n], dtype=np.float64) for n in names]
    labels = [display_transition_or_group(n) for n in names]
    bp = ax.boxplot(vals, labels=labels, showfliers=False, patch_artist=True)
    for patch, name in zip(bp["boxes"], names):
        color = color_for_display_name(name)
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.18)
    for median in bp["medians"]:
        median.set_color("0.15")
        median.set_linewidth(1.1)
    rng = np.random.default_rng(123)
    for i, v in enumerate(vals, start=1):
        ax.scatter(
            i + rng.normal(0, 0.035, size=len(v)),
            v,
            s=9,
            alpha=0.35,
            color=color_for_display_name(names[i - 1]),
            linewidths=0,
        )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    color_group_ticklabels(ax, axis="x")
    ax.grid(axis="y", color="0.93", lw=0.5)


def fig2(out_dir, success_rows, align_sample, disp_rows, update_summary, layers):
    fig = plt.figure(figsize=FIGSIZE_GRID, constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    sub = gs[1, 1].subgridspec(1, 2)
    ax_d1 = fig.add_subplot(sub[0, 0])
    ax_d2 = fig.add_subplot(sub[0, 1])
    groups = [r["group"] for r in success_rows]
    rates = [r["success_rate"] for r in success_rows]
    lo = [r["success_rate"] - r["ci_low"] for r in success_rows]
    hi = [r["ci_high"] - r["success_rate"] for r in success_rows]
    ax_a.bar(groups, rates, yerr=[lo, hi], color=[COLORS[g] for g in groups], capsize=2)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_title("Desirable output behavior rate", fontweight="bold")
    ax_a.tick_params(axis="x", rotation=30)
    color_group_ticklabels(ax_a, axis="x")
    align_vals = defaultdict(list)
    for r in align_sample:
        align_vals[r["transition"]].append(r["delta_target_distance"])
    boxplot_with_points(ax_b, align_vals, "Transition-conditioned target alignment", "distance_on - distance_off")
    disp_vals = defaultdict(list)
    for r in disp_rows:
        disp_vals[r["group"]].append(r["q5d_displacement_norm"])
    boxplot_with_points(ax_c, disp_vals, "Subspace displacement norm", "Subspace displacement")
    mats = {}
    for metric in ["applied_jb_update_norm", "applied_or_update_norm"]:
        mat = np.full((len(GROUPS), len(layers)), np.nan)
        for gi, group in enumerate(GROUPS):
            for li, layer in enumerate(layers):
                vals = [r["mean"] for r in update_summary if r["group"] == group and r["layer"] == layer and r["metric"] == metric]
                if vals:
                    mat[gi, li] = vals[0]
        mats[metric] = mat
    for ax, metric, title in [(ax_d1, "applied_jb_update_norm", "Applied JB update norm"), (ax_d2, "applied_or_update_norm", "Applied OR update norm")]:
        im = ax.imshow(mats[metric], aspect="auto", cmap="magma")
        ax.set_xticks(range(len(layers)), [str(l) for l in layers])
        ax.set_yticks(range(len(GROUPS)), GROUPS)
        color_group_ticklabels(ax, axis="y")
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    pdf, png = out_dir / "fig_inference_correction_and_preservation.pdf", out_dir / "fig_inference_correction_and_preservation.png"
    return save_fig_pair(fig, pdf, png)


def fig2_cached(out_dir, success_rows, align_sample, disp_rows):
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE, constrained_layout=True)
    groups = [r["group"] for r in success_rows]
    rates = [r["success_rate"] for r in success_rows]
    lo = [r["success_rate"] - r["ci_low"] for r in success_rows]
    hi = [r["ci_high"] - r["success_rate"] for r in success_rows]
    axes[0].bar(groups, rates, yerr=[lo, hi], color=[COLORS[g] for g in groups], capsize=2)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Desirable output behavior rate", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)
    color_group_ticklabels(axes[0], axis="x")

    align_vals = defaultdict(list)
    for r in align_sample:
        align_vals[r["transition"]].append(r["delta_target_distance"])
    boxplot_with_points(axes[1], align_vals, "Transition-conditioned target alignment", "distance_on - distance_off")

    disp_vals = defaultdict(list)
    for r in disp_rows:
        disp_vals[r["group"]].append(r["q5d_displacement_norm"])
    boxplot_with_points(axes[2], disp_vals, "Subspace displacement norm", "Subspace displacement")

    pdf, png = out_dir / "fig_inference_correction_and_preservation.pdf", out_dir / "fig_inference_correction_and_preservation.png"
    return save_fig_pair(fig, pdf, png)


def transition_figures(out_dir, behavior_delta):
    paths = []
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)
    for tr in TRANSITION_ORDER:
        rows = [r for r in behavior_delta if r["transition"] == tr]
        if not rows:
            continue
        for r in rows:
            ax.annotate("", xy=(r["x_on"], r["y_on"]), xytext=(r["x_off"], r["y_off"]),
                        arrowprops=dict(arrowstyle="->", lw=0.6, color=TRANSITION_COLORS[tr], alpha=0.35))
        ax.scatter([r["x_off"] for r in rows], [r["y_off"] for r in rows], s=8, alpha=0.2, color=TRANSITION_COLORS[tr], label=tr)
    ax.set_xlabel("Harmfulness-related projection")
    ax.set_ylabel("Refusal-related projection")
    ax.set_title("Transition-conditioned behavior displacement")
    ax.legend(frameon=False)
    pdf, png = out_dir / "fig_transition_conditioned_behavior_displacement.pdf", out_dir / "fig_transition_conditioned_behavior_displacement.png"
    paths += save_fig_pair(fig, pdf, png)
    for metric, stem in [("delta_refuse", "fig_delta_refuse_by_transition"), ("delta_harm", "fig_delta_harm_by_transition")]:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)
        vals = defaultdict(list)
        for r in behavior_delta:
            vals[r["transition"]].append(r[metric])
        boxplot_with_points(ax, vals, stem, metric)
        pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
        paths += save_fig_pair(fig, pdf, png)
    return paths


def semantic_fig(out_dir, semantic_summary):
    mat = np.full((len(TRANSITION_ORDER), len(SEMANTIC_LABELS)), np.nan)
    for i, tr in enumerate(TRANSITION_ORDER):
        for j, d in enumerate(SEMANTIC_LABELS):
            vals = [r["mean"] for r in semantic_summary if r["transition"] == tr and r["direction"] == d]
            if vals:
                mat[i, j] = vals[0]
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(SEMANTIC_LABELS)), SEMANTIC_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(TRANSITION_ORDER)), TRANSITION_ORDER)
    color_group_ticklabels(ax, axis="y")
    ax.set_title("Semantic direction displacement by transition")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    pdf, png = out_dir / "fig_semantic_direction_displacement_by_transition.pdf", out_dir / "fig_semantic_direction_displacement_by_transition.png"
    return save_fig_pair(fig, pdf, png)


def routing_heatmap_fig(out_dir, update_summary, layers):
    mats = {}
    for metric in ["p_jb_eff", "p_or_eff"]:
        mat = np.full((len(GROUPS), len(layers)), np.nan)
        for gi, group in enumerate(GROUPS):
            for li, layer in enumerate(layers):
                vals = [r["mean"] for r in update_summary if r["group"] == group and r["layer"] == layer and r["metric"] == metric]
                if vals:
                    mat[gi, li] = vals[0]
        mats[metric] = mat
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, constrained_layout=True)
    for ax, metric, title in [(axes[0], "p_jb_eff", "Mean effective p_jb"), (axes[1], "p_or_eff", "Mean effective p_or")]:
        im = ax.imshow(mats[metric], aspect="auto", cmap="viridis", vmin=0)
        ax.set_xticks(range(len(layers)), [str(l) for l in layers])
        ax.set_yticks(range(len(GROUPS)), GROUPS)
        color_group_ticklabels(ax, axis="y")
        ax.set_title(title)
        ax.set_xlabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    pdf, png = out_dir / "fig_gate_routing_by_group.pdf", out_dir / "fig_gate_routing_by_group.png"
    return save_fig_pair(fig, pdf, png)


def ensure_eval_on(root, artifacts, artifact_candidates, eval_samples, checkpoint, subspace_cache, args):
    if artifacts.get("eval_on") and Path(artifacts["eval_on"]).exists():
        print(f"Reusing eval_on: {artifacts['eval_on']}")
        return load_json(Path(artifacts["eval_on"]))
    if args.skip_eval_on_generation:
        print("eval_on.json missing; skipping output-behavior-dependent analyses.")
        return None

    sys.path.insert(0, str(root))
    from src.classifier import WildGuard
    from src.model import generate as model_generate
    from src.model import load_model

    adapter, _ = build_adapter(root, checkpoint, subspace_cache, args)
    model, tokenizer = load_model(args.model_id, dtype=args.dtype, device_map=args.device_map)
    adapter.to(model.device).eval()
    prompts = [s["prompt"] for s in eval_samples]
    spans = [s["span"] for s in eval_samples]
    last_idxs = [s["last_idx"] for s in eval_samples]
    responses, p_jb_last, p_or_last = [], [], []
    bs = max(1, args.batch_size)
    for start in range(0, len(prompts), bs):
        end = min(start + bs, len(prompts))
        enc, spans_adj, last_adj = offset_spans_and_encode(tokenizer, prompts[start:end], spans[start:end], last_idxs[start:end], model.device)
        del enc
        with adapter.steer(model, spans_adj, last_adj, train_mode=False, capture=False) as state:
            chunk = model_generate(model, tokenizer, prompts[start:end], max_new_tokens=256, batch_size=end - start)
        responses.extend(chunk)
        if state["routing_log"]["p_jb"]:
            last_layer = max(state["routing_log"]["p_jb"].keys())
            p_jb_last.extend(state["routing_log"]["p_jb"][last_layer].tolist())
            p_or_last.extend(state["routing_log"]["p_or"][last_layer].tolist())
    wg = WildGuard()
    wg_results = wg.classify_batch(prompts, responses, batch_size=max(1, min(8, bs)))
    wg.unload()
    eval_on = {
        "eval_samples": [
            {"group": s["group"], "idx": s["idx"], "prompt": s["prompt"], "id": s["sample_id"]} for s in eval_samples
        ],
        "responses": responses,
        "wg": wg_results,
        "p_jb": p_jb_last,
        "p_or": p_or_last,
        "steer_span_offset": "left_padding_batch_v1",
    }
    out = default_artifact_path(root, artifact_candidates, "eval_on")
    save_json(out, eval_on)
    print(f"Saved eval_on: {out}")
    return eval_on


def make_behavior_id_maps(eval_samples, records, behavior_off, behavior_on, layers):
    off_by_id, on_by_id = {}, {}
    for group in GROUPS:
        for sample, point in zip([s for s in eval_samples if s["group"] == group], behavior_off[group]):
            off_by_id[sample["sample_id"]] = point
        for rec, point in zip([r for r in records if r["group"] == group], behavior_on[group]):
            on_by_id[rec["sample_id"]] = point
    return off_by_id, on_by_id


def make_report(path, ctx):
    lines = []
    lines.append("# Q5D Figure Analysis Report\n")
    lines.append("## Artifacts\n")
    for name, p in ctx["artifacts"].items():
        if p:
            lines.append(f"- `{name}`: `{p}`")
    lines.append(f"- checkpoint id: `{ctx['checkpoint_id']['sha256_16']}`\n")
    lines.append("## Evaluation Counts\n")
    for group, n in ctx["counts"].items():
        lines.append(f"- `{group}`: {n}")
    lines.append("\n## Behavior-Axis Validation\n")
    if ctx["contrast_warnings"]:
        lines.append("Opposite-sign conditional contrast warnings:")
        for w in ctx["contrast_warnings"]:
            lines.append(f"- {w}")
        lines.append("- Recommendation: do not make a strong harmfulness/refusal-axis motivation claim without qualifying these failures.")
    else:
        lines.append("- Conditional contrasts have the expected positive sign at all checked ACT layers.")
    lines.append("\n## AUROC\n")
    for row in ctx["auroc_rows"]:
        if row["layer"] == "mean_ACT_LAYERS":
            lines.append(f"- `{row['axis']}` aggregated AUROC: {row['auroc']:.3f}")
    lines.append("\n## Nested Subspace AUROC\n")
    for task in sorted({r["task"] for r in ctx["nested_rows"]}):
        q5 = next((r["auroc"] for r in ctx["nested_rows"] if r["task"] == task and r["subspace"] == "Q5D_full"), math.nan)
        q2 = next((r["auroc"] for r in ctx["nested_rows"] if r["task"] == task and r["subspace"] == "Q2D_harm_refuse"), math.nan)
        rand = [r["auroc"] for r in ctx["nested_rows"] if r["task"] == task and r["subspace"] == "Random5D"]
        rand_mean = float(np.mean(rand)) if rand else math.nan
        lines.append(f"- `{task}`: Q2D={q2:.3f}, Q5D={q5:.3f}, Random5D mean={rand_mean:.3f}")
        if np.isfinite(q5) and np.isfinite(q2) and q5 <= q2:
            lines.append("  - Warning: Q5D does not improve over Q2D; weaken Q5D necessity claim.")
        if np.isfinite(q5) and np.isfinite(rand_mean) and q5 <= rand_mean:
            lines.append("  - Warning: Q5D does not improve over Random5D; treat as diagnostic, not proof.")
    lines.append("\n## Q5D Rank Diagnostics\n")
    for layer, rank in ctx["rank_by_layer"].items():
        lines.append(f"- L{layer}: effective rank@1e-3 = {rank}")
        if rank < 5:
            lines.append("  - Warning: effective rank below 5; original directions may be redundant.")
    lines.append("\n## Inference Behavior\n")
    for row in ctx["success_rows"]:
        lines.append(f"- `{row['group']}` desirable rate: {row['success_rate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] n={row['n']}")
    lines.append("\n## Transition Target Alignment\n")
    for row in ctx["align_summary"]:
        lines.append(f"- `{row['transition']}` delta target distance: {row['mean']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] n={row['n']}")
        if row["transition"] == "jb_corr_to_refused" and np.isfinite(row["mean"]) and row["mean"] >= 0:
            lines.append("  - Warning: successful jailbreak transitions do not reduce target distance; avoid main representation-correction claim.")
    lines.append("\n## Preservation Displacement\n")
    for row in ctx["disp_summary"]:
        lines.append(f"- `{row['group']}` Q5D displacement: {row['mean']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
    jb = next((r for r in ctx["jb_diag"] if r["group"] == "jb_corr"), None)
    blocked = next((r for r in ctx["jb_diag"] if r["group"] == "jb_blocked"), None)
    if blocked and blocked["applied_jb_update_norm"] > 0.5 * max(jb["applied_jb_update_norm"], 1e-12):
        lines.append("\n- Warning: `jb_blocked` applied JB update norm is not small relative to `jb_corr`; preservation mechanism needs qualification.")
    lines.append("\n## Variance Diagnostic\n")
    all_var = [r for r in ctx["variance_rows"] if r["subset"] == "jb_corr_all"]
    sub_var = [r for r in ctx["variance_rows"] if r["subset"] != "jb_corr_all"]
    lines.append("- Inspect `jb_corr_variance_diagnostics.csv`: if all variance increases but subset variance does not, mixture explains the aggregate spread.")
    if all_var and sub_var:
        lines.append("- Automated recommendation: use transition-conditioned panels in main text if aggregate centroid movement is weak.")
    lines.append("\n## Panel Recommendation\n")
    lines.append("- Safer main panels: Figure 1A/1B if conditional contrasts hold, Figure 2A/2B/2D when output transitions and applied updates align.")
    lines.append("- Appendix diagnostics: nested subspace AUROC, Q5D rank diagnostics, raw routing probabilities, semantic-direction heatmap.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    global SAVE_PDF
    args = parse_args()
    root = args.root.resolve()
    SAVE_PDF = not args.png_only
    if args.output_dir is None:
        out_dir = root / "output/figures"
    else:
        out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts, artifact_candidates = resolve_artifacts(root, args)
    import_runtime_deps(root)
    setup_matplotlib()

    emb_train = load_pt(Path(artifacts["emb_train"]))
    eval_emb_off = load_pt(Path(artifacts["eval_emb_off"]))
    eval_off = load_json(Path(artifacts["eval_off"]))
    dirs = load_pt(Path(artifacts["directions"]))
    targets = load_pt(Path(artifacts["targets"]))
    subspace_cache = load_pt(Path(artifacts["subspace"]))
    checkpoint = load_pt(Path(artifacts["adapter"]))
    ckpt_id = checkpoint_identifier(Path(artifacts["adapter"]))
    cfg = checkpoint.get("config", {})
    layers = [int(x) for x in config_value(cfg, "ACT_LAYERS", checkpoint.get("act_layers", ACT_LAYERS_DEFAULT))]
    weights = weighted_layers(cfg, layers)
    q_basis = q_basis_from_cache(subspace_cache)

    eval_samples = get_eval_samples(eval_off, eval_emb_off)
    eval_off_records = []
    for s in eval_samples:
        rec = dict(s["off"])
        rec["sample_id"] = s["sample_id"]
        rec["idx"] = s["idx"]
        rec["prompt"] = s["prompt"]
        rec["group"] = s["group"]
        eval_off_records.append(rec)
    train_by_group = {g: list(emb_train.get(g, [])) for g in GROUPS}
    eval_by_group = group_samples(eval_off_records)
    require_groups("training cache", train_by_group)
    require_groups("eval off cache", eval_by_group)

    stats = train_projection_stats(train_by_group, dirs, layers)
    off_proj = behavior_projection(eval_by_group, dirs, stats, layers, "off")
    behavior_scores = aggregate_behavior(off_proj, layers)
    contrast_rows, contrast_warnings = compute_axis_contrasts(off_proj, layers, args.n_bootstrap, args.seed)
    auroc_rows = compute_axis_aurocs(train_by_group, eval_by_group, dirs, stats, layers)
    nested_rows = compute_nested_subspace_auroc(train_by_group, eval_by_group, dirs, layers, args.random5d_n, args.seed)
    rank_rows, cos_by_layer = compute_q5d_rank_diagnostics(dirs, layers)
    rank_by_layer = {
        layer: next(r["effective_rank_1e-3"] for r in rank_rows if r["layer"] == layer and r["stat"] == "singular_value" and r["index"] == 1)
        for layer in layers
    }

    paths = []
    paths += fig1(out_dir, behavior_scores, contrast_rows, nested_rows, args.n_bootstrap, args.seed)
    main_rep_paths, main_rep_values = fig_representation_analysis_main(
        out_dir, behavior_scores, contrast_rows, nested_rows, eval_by_group, args.n_bootstrap, args.seed
    )
    appendix_rep_paths = fig_representation_analysis_appendix(out_dir, contrast_rows, nested_rows)
    paths += main_rep_paths + appendix_rep_paths
    paths += fig_rank(out_dir, rank_rows, cos_by_layer, layers)
    write_csv(out_dir / "representation_axis_contrasts.csv", contrast_rows)
    write_csv(out_dir / "representation_axis_auroc.csv", auroc_rows)
    write_csv(out_dir / "nested_subspace_auroc.csv", nested_rows)
    save_json(out_dir / "q5d_rank_and_direction_diagnostics.json", {"rows": rank_rows, "rank_by_layer": rank_by_layer})
    write_csv(out_dir / "q5d_rank_and_direction_diagnostics.csv", rank_rows)

    if args.skip_detailed_inference:
        summary = {
            "metadata": {
                "seed": args.seed,
                "n_bootstrap": args.n_bootstrap,
                "checkpoint": ckpt_id,
                "artifacts": {k: str(v) for k, v in artifacts.items() if v},
                "artifact_profile": args.artifact_profile,
                "skip_detailed_inference": True,
            },
            "group_counts": {g: len(eval_by_group[g]) for g in GROUPS},
            "axis_contrast_warnings": contrast_warnings,
            "auroc": auroc_rows,
            "nested_subspace_auroc": nested_rows,
            "representation_analysis_main": main_rep_values,
            "q5d_rank_by_layer": rank_by_layer,
            "generated_files": [str(p) for p in paths],
        }
        save_json(out_dir / "representation_axes_to_q5d_summary.json", summary)
        print("\nGenerated representation-analysis files:")
        for p in paths + [
            out_dir / "representation_axis_contrasts.csv",
            out_dir / "representation_axis_auroc.csv",
            out_dir / "nested_subspace_auroc.csv",
            out_dir / "representation_axes_to_q5d_summary.json",
        ]:
            print(f"  {p}")
        print("\nSkipped detailed inference diagnostics (--skip-detailed-inference).")
        print("This mode does not create applied-update, routing, Q5D displacement, or transition-alignment figures.")
        return 0

    if args.use_eval_emb_on_cache:
        if not artifacts.get("eval_emb_on") or not Path(artifacts["eval_emb_on"]).exists():
            raise SystemExit("--use-eval-emb-on-cache requires eval_emb_on.pt. Pass --eval-emb-on-path or create the sweep cache first.")
        print(f"CPU-only cached inference analysis from eval_emb_on: {artifacts['eval_emb_on']}")
        eval_emb_on = load_pt(Path(artifacts["eval_emb_on"]))
        records = records_from_eval_emb_on(eval_samples, eval_emb_on, layers)
        eval_on = load_json(Path(artifacts["eval_on"])) if artifacts.get("eval_on") and Path(artifacts["eval_on"]).exists() else {}
        attach_eval_on_behavior(records, eval_on)

        off_by_id = {r["sample_id"]: r for r in eval_off_records}
        on_by_group = group_samples(records)
        require_groups("eval on cache", on_by_group)
        on_proj = behavior_projection(on_by_group, dirs, stats, layers, "on")
        behavior_on = aggregate_behavior(on_proj, layers)
        off_scores_by_id, on_scores_by_id = make_behavior_id_maps(eval_samples, records, behavior_scores, behavior_on, layers)

        success_rows = success_rate_rows(records, args.n_bootstrap, args.seed)
        align_sample, align_summary = transition_alignment_rows(off_by_id, records, dirs, targets, q_basis, layers, weights, args.n_bootstrap, args.seed)
        disp_rows, disp_summary = displacement_rows(off_by_id, records, targets, q_basis, layers, weights, args.n_bootstrap, args.seed)
        paths += fig2_cached(out_dir, success_rows, align_sample, disp_rows)
        write_csv(out_dir / "inference_behavior_success_rates.csv", success_rows)
        write_csv(out_dir / "inference_transition_target_alignment.csv", align_sample + align_summary)
        write_csv(out_dir / "inference_q5d_displacement.csv", disp_rows + disp_summary)

        behavior_delta = behavior_delta_rows(off_scores_by_id, on_scores_by_id, records)
        variance_rows = jb_variance_rows(off_by_id, records, behavior_delta, q_basis, targets, layers, weights)
        paths += transition_figures(out_dir, behavior_delta)
        write_csv(out_dir / "jb_corr_variance_diagnostics.csv", variance_rows)
        save_json(out_dir / "jb_corr_variance_diagnostics.json", {"rows": variance_rows})

        semantic_sample, semantic_summary = semantic_direction_rows(off_by_id, records, dirs, layers, weights, args.n_bootstrap, args.seed)
        paths += semantic_fig(out_dir, semantic_summary)
        write_csv(out_dir / "semantic_direction_displacement_by_transition.csv", semantic_sample + semantic_summary)

        summary = {
            "metadata": {
                "seed": args.seed,
                "n_bootstrap": args.n_bootstrap,
                "checkpoint": ckpt_id,
                "artifacts": {k: str(v) for k, v in artifacts.items() if v},
                "artifact_profile": args.artifact_profile,
                "use_eval_emb_on_cache": True,
                "skipped": ["hook_capture", "applied_update_norm", "routing_heatmap"],
            },
            "group_counts": {g: len(eval_by_group[g]) for g in GROUPS},
            "axis_contrast_warnings": contrast_warnings,
            "auroc": auroc_rows,
            "nested_subspace_auroc": nested_rows,
            "representation_analysis_main": main_rep_values,
            "q5d_rank_by_layer": rank_by_layer,
            "success_rates": success_rows,
            "transition_alignment_summary": align_summary,
            "displacement_summary": disp_summary,
            "generated_files": [str(p) for p in paths],
        }
        save_json(out_dir / "representation_axes_to_q5d_summary.json", summary)
        save_json(out_dir / "inference_correction_summary.json", summary)
        paths += [
            out_dir / "representation_axis_contrasts.csv",
            out_dir / "representation_axis_auroc.csv",
            out_dir / "nested_subspace_auroc.csv",
            out_dir / "representation_axes_to_q5d_summary.json",
            out_dir / "q5d_rank_and_direction_diagnostics.csv",
            out_dir / "q5d_rank_and_direction_diagnostics.json",
            out_dir / "inference_behavior_success_rates.csv",
            out_dir / "inference_transition_target_alignment.csv",
            out_dir / "inference_q5d_displacement.csv",
            out_dir / "inference_correction_summary.json",
            out_dir / "jb_corr_variance_diagnostics.csv",
            out_dir / "jb_corr_variance_diagnostics.json",
            out_dir / "semantic_direction_displacement_by_transition.csv",
        ]
        print("\nGenerated CPU-only cached analysis files:")
        for p in paths:
            print(f"  {p}")
        print("\nSkipped hook-only diagnostics: applied update norm and routing heatmaps.")
        return 0

    capture = capture_detailed(root, artifacts, artifact_candidates, eval_samples, checkpoint, subspace_cache, args, ckpt_id)
    records = capture["records"]
    eval_on = ensure_eval_on(root, artifacts, artifact_candidates, eval_samples, checkpoint, subspace_cache, args)
    attach_eval_on_behavior(records, eval_on)
    off_by_id = {r["sample_id"]: r for r in eval_off_records}
    on_by_group = group_samples(records)
    on_proj = behavior_projection(on_by_group, dirs, stats, layers, "on")
    behavior_on = aggregate_behavior(on_proj, layers)
    off_scores_by_id, on_scores_by_id = make_behavior_id_maps(eval_samples, records, behavior_scores, behavior_on, layers)

    success_rows = success_rate_rows(records, args.n_bootstrap, args.seed)
    align_sample, align_summary = transition_alignment_rows(off_by_id, records, dirs, targets, q_basis, layers, weights, args.n_bootstrap, args.seed)
    disp_rows, disp_summary = displacement_rows(off_by_id, records, targets, q_basis, layers, weights, args.n_bootstrap, args.seed)
    update_sample_rows = applied_update_rows(records, layers)
    update_summary = summarize_by_group_layer(
        update_sample_rows,
        ["p_jb_eff", "p_or_eff", "delta_jb_norm", "applied_jb_update_norm", "applied_or_update_norm", "applied_total_update_norm"],
        args.n_bootstrap,
        args.seed,
    )
    jb_diag = []
    for group in ["jb_corr", "jb_blocked"]:
        sub = [r for r in update_sample_rows if r["group"] == group]
        disp = [r["q5d_displacement_norm"] for r in disp_rows if r["group"] == group]
        succ = next((r["success_rate"] for r in success_rows if r["group"] == group), math.nan)
        jb_diag.append(
            {
                "group": group,
                "mean_effective_p_jb": float(np.mean([r["p_jb_eff"] for r in sub])) if sub else math.nan,
                "raw_jb_delta_norm": float(np.mean([r["delta_jb_norm"] for r in sub])) if sub else math.nan,
                "applied_jb_update_norm": float(np.mean([r["applied_jb_update_norm"] for r in sub])) if sub else math.nan,
                "q5d_displacement_norm": float(np.mean(disp)) if disp else math.nan,
                "desirable_behavior_rate": succ,
            }
        )
    paths += fig2(out_dir, success_rows, align_sample, disp_rows, update_summary, layers)
    paths += routing_heatmap_fig(out_dir, update_summary, layers)
    write_csv(out_dir / "inference_behavior_success_rates.csv", success_rows)
    write_csv(out_dir / "inference_transition_target_alignment.csv", align_sample + align_summary)
    write_csv(out_dir / "inference_q5d_displacement.csv", disp_rows + disp_summary)
    write_csv(out_dir / "inference_applied_update_norm.csv", update_sample_rows + update_summary)
    write_csv(out_dir / "jb_corr_vs_jb_blocked_update_diagnostics.csv", jb_diag)

    behavior_delta = behavior_delta_rows(off_scores_by_id, on_scores_by_id, records)
    variance_rows = jb_variance_rows(off_by_id, records, behavior_delta, q_basis, targets, layers, weights)
    paths += transition_figures(out_dir, behavior_delta)
    write_csv(out_dir / "jb_corr_variance_diagnostics.csv", variance_rows)
    save_json(out_dir / "jb_corr_variance_diagnostics.json", {"rows": variance_rows})

    semantic_sample, semantic_summary = semantic_direction_rows(off_by_id, records, dirs, layers, weights, args.n_bootstrap, args.seed)
    paths += semantic_fig(out_dir, semantic_summary)
    write_csv(out_dir / "semantic_direction_displacement_by_transition.csv", semantic_sample + semantic_summary)

    summary = {
        "metadata": {"seed": args.seed, "n_bootstrap": args.n_bootstrap, "checkpoint": ckpt_id, "artifacts": {k: str(v) for k, v in artifacts.items() if v}},
        "group_counts": {g: len(eval_by_group[g]) for g in GROUPS},
        "axis_contrast_warnings": contrast_warnings,
        "auroc": auroc_rows,
        "nested_subspace_auroc": nested_rows,
        "representation_analysis_main": main_rep_values,
        "q5d_rank_by_layer": rank_by_layer,
        "success_rates": success_rows,
        "transition_alignment_summary": align_summary,
        "displacement_summary": disp_summary,
        "jb_corr_vs_jb_blocked": jb_diag,
        "generated_files": [str(p) for p in paths],
    }
    save_json(out_dir / "representation_axes_to_q5d_summary.json", summary)
    save_json(out_dir / "inference_correction_summary.json", summary)
    report = make_report(
        out_dir / "q5d_figure_analysis_report.md",
        {
            "artifacts": artifacts,
            "checkpoint_id": ckpt_id,
            "counts": summary["group_counts"],
            "contrast_warnings": contrast_warnings,
            "auroc_rows": auroc_rows,
            "nested_rows": nested_rows,
            "rank_by_layer": rank_by_layer,
            "success_rows": success_rows,
            "align_summary": align_summary,
            "disp_summary": disp_summary,
            "jb_diag": jb_diag,
            "variance_rows": variance_rows,
        },
    )
    paths += [
        out_dir / "representation_axis_contrasts.csv",
        out_dir / "representation_axis_auroc.csv",
        out_dir / "nested_subspace_auroc.csv",
        out_dir / "representation_axes_to_q5d_summary.json",
        out_dir / "q5d_rank_and_direction_diagnostics.csv",
        out_dir / "q5d_rank_and_direction_diagnostics.json",
        out_dir / "inference_behavior_success_rates.csv",
        out_dir / "inference_transition_target_alignment.csv",
        out_dir / "inference_q5d_displacement.csv",
        out_dir / "inference_applied_update_norm.csv",
        out_dir / "jb_corr_vs_jb_blocked_update_diagnostics.csv",
        out_dir / "inference_correction_summary.json",
        out_dir / "jb_corr_variance_diagnostics.csv",
        out_dir / "jb_corr_variance_diagnostics.json",
        out_dir / "semantic_direction_displacement_by_transition.csv",
        report,
        artifacts.get("capture_detailed") or default_artifact_path(root, artifact_candidates, "capture_detailed"),
    ]

    print("\nGenerated files:")
    for p in paths:
        print(f"  {p}")
    print("\nRepresentation-analysis main/appendix figures:")
    rep_print = []
    if SAVE_PDF:
        rep_print += [
            out_dir / "fig_representation_analysis_main.pdf",
            out_dir / "fig_representation_analysis_appendix.pdf",
        ]
    rep_print += [
        out_dir / "fig_representation_analysis_main.png",
        out_dir / "fig_representation_analysis_appendix.png",
        out_dir / "fig_representation_analysis_main_values.csv",
        out_dir / "fig_representation_analysis_main_values.json",
    ]
    for p in rep_print:
        print(f"  {p}")
    print("\nPanel B layer-aggregated summary:")
    for name in ["jb_blocked_minus_jb_corr_jailbreak", "or_corr_minus_benign_ans_safe"]:
        sub = [r for r in contrast_rows if r["contrast"] == name]
        mean = float(np.mean([r["signed_standardized_mean_diff"] for r in sub])) if sub else math.nan
        lo = float(np.mean([r["ci_low"] for r in sub])) if sub else math.nan
        hi = float(np.mean([r["ci_high"] for r in sub])) if sub else math.nan
        print(f"  {CONTRAST_LABELS[name]:24s} mean={mean:+.3f}  mean_CI=[{lo:+.3f}, {hi:+.3f}]")
    print("\nPanel C correction-need probe:")
    for subspace in MAIN_SUBSPACE_ORDER:
        print(f"  {MAIN_SUBSPACE_LABELS[subspace]:16s} {nested_auroc_value(nested_rows, 'correction_vs_preservation', subspace):.3f}")
    rand = random5d_summary(nested_rows, "correction_vs_preservation")
    print(f"  Random 5D        mean={rand['mean']:.3f}  CI=[{rand['ci_low']:.3f}, {rand['ci_high']:.3f}]  n={rand['n']}")
    print("\nCurves removed from the main representation figure:")
    for label in [
        "jb_corr - benign_ans",
        "harm_refuse - or_corr",
        "jb_blocked - or_corr",
        "harm_refuse - jb_corr",
        "jb_corr vs jb_blocked",
        "or_corr vs benign_ans",
    ]:
        print(f"  {label}")
    print("\nGroup counts:")
    for g, n in summary["group_counts"].items():
        print(f"  {g:12s} {n}")
    harm_auc = next(r["auroc"] for r in auroc_rows if r["axis"] == "harm" and r["layer"] == "mean_ACT_LAYERS")
    ref_auc = next(r["auroc"] for r in auroc_rows if r["axis"] == "refuse" and r["layer"] == "mean_ACT_LAYERS")
    print(f"\nLayer-aggregated harmfulness AUROC: {harm_auc:.3f}")
    print(f"Layer-aggregated refusal AUROC: {ref_auc:.3f}")
    if contrast_warnings:
        print("\nOpposite-sign conditional contrast warnings:")
        for w in contrast_warnings:
            print(f"  {w}")
    print("\nNested subspace AUROC:")
    for r in nested_rows:
        if r["subspace"] != "Random5D":
            print(f"  {r['task']:30s} {r['subspace']:18s} {r['auroc']:.3f}")
    print("\nQ5D effective rank by layer:")
    for layer, rank in rank_by_layer.items():
        print(f"  L{layer}: {rank}")
    print("\nDesirable behavior rate:")
    for r in success_rows:
        print(f"  {r['group']:12s} {r['success_rate']:.3f}")
    print("\nTransition target alignment change:")
    for r in align_summary:
        print(f"  {r['transition']:24s} {r['mean']:+.3f}")
    print("\nPreservation Q5D displacement:")
    for r in disp_summary:
        if r["group"] in PRESERVATION_GROUPS:
            print(f"  {r['group']:12s} {r['mean']:.3f}")
    print("\nApplied JB update norm comparison:")
    for r in jb_diag:
        print(f"  {r['group']:12s} {r['applied_jb_update_norm']:.4f}")
    print("\nRecommendation: use Figure 1A/1B plus Figure 2A/2B/2D as main only if warnings in the markdown report are absent or acceptable; keep Q5D rank/random-subspace/semantic diagnostics in appendix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
