"""Our training-data adapters for RepBend / X-Boundary baselines.

The paper ports originally trained on each paper's own prompt/response pools.
For head-to-head comparison with the v-series adapters, these helpers expose the
same manifest-derived groups used by our over-refusal pipeline:

  jb_corr / or_corr / jb_blocked / harm_refuse / benign_ans

When ``cache/manifest.jsonl`` is not present, callers can fall back to the
project ``src.dataset`` loaders. The fallback never uses paper-specific data.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "cache" / "manifest.jsonl"
DEFAULT_EVAL_INDICES = REPO_ROOT / "cache" / "v07" / "eval_indices.json"

DEFAULT_REFUSAL_RESPONSE = (
    "I can't help with instructions, code, or guidance that would facilitate "
    "harmful or illegal activity."
)
DEFAULT_SAFE_COMPLETION = (
    "This is a safe request. I can help with a concise, direct answer."
)


def ensure_repo_root_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _has_text(row: dict, key: str) -> bool:
    return isinstance(row.get(key), str) and bool(row[key].strip())


def _prompt(row: dict) -> str:
    return str(row.get("prompt", "")).strip()


def _response(row: dict) -> str:
    return str(row.get("response", "")).strip()


def _is_safe_complied(row: dict) -> bool:
    return (
        row.get("refuse_label") is False
        and row.get("prompt_harm") is False
        and row.get("response_harm") is not True
        and _has_text(row, "prompt")
        and _has_text(row, "response")
    )


def _is_overrefused(row: dict) -> bool:
    return (
        row.get("category") == "safe_sensitive"
        and row.get("refuse_label") is True
        and row.get("prompt_harm") is False
        and row.get("response_harm") is not True
        and _has_text(row, "prompt")
        and _has_text(row, "response")
    )


def _is_harmful_refused(row: dict) -> bool:
    return (
        row.get("refuse_label") is True
        and (
            row.get("prompt_harm") is True
            or row.get("category") in {"harmful", "harmful_jailbreak"}
        )
        and _has_text(row, "prompt")
        and _has_text(row, "response")
    )


def _is_harmful_complied(row: dict) -> bool:
    return (
        row.get("refuse_label") is False
        and row.get("response_harm") is True
        and (
            row.get("prompt_harm") is True
            or row.get("category") in {"harmful", "harmful_jailbreak"}
        )
        and _has_text(row, "prompt")
        and _has_text(row, "response")
    )


def _group_key(row: dict) -> str | None:
    if _is_harmful_complied(row):
        return "jb_corr"
    if _is_overrefused(row):
        return "or_corr"
    if row.get("category") == "harmful_jailbreak" and row.get("refuse_label") is True:
        return "jb_blocked"
    if row.get("category") == "harmful" and row.get("refuse_label") is True:
        return "harm_refuse"
    if row.get("category") == "benign" and _is_safe_complied(row):
        return "benign_ans"
    return None


def _load_eval_indices(path: Path) -> dict[str, set[int]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "eval" in raw:
        raw = raw["eval"]
    return {str(k): {int(i) for i in v} for k, v in raw.items()}


def load_manifest_rows(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    eval_indices_path: str | Path | None = DEFAULT_EVAL_INDICES,
    exclude_eval: bool = True,
) -> list[dict]:
    """Load our manifest rows, optionally excluding the v-series eval split."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not exclude_eval or eval_indices_path is None:
        return rows

    eval_idx = _load_eval_indices(Path(eval_indices_path))
    if not eval_idx:
        return rows

    grouped: dict[str, list[dict]] = {g: [] for g in eval_idx}
    keep: list[dict] = []
    for row in rows:
        g = _group_key(row)
        if g in grouped:
            grouped[g].append(row)
        else:
            keep.append(row)

    for g, group_rows in grouped.items():
        banned = eval_idx.get(g, set())
        keep.extend(row for i, row in enumerate(group_rows) if i not in banned)
    return keep


def _sample_with_replacement(items: list[Any], n: int, *, rng: random.Random) -> list[Any]:
    if not items:
        return []
    if len(items) >= n:
        return rng.sample(items, n)
    return [rng.choice(items) for _ in range(n)]


def _pairs(rows: list[dict], *, response: str | None = None) -> list[tuple[str, str]]:
    out = []
    for row in rows:
        p = _prompt(row)
        r = response if response is not None else _response(row)
        if p and r:
            out.append((p, r))
    return out


def manifest_repbend_pools(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    num_examples: int,
    seed: int = 42,
    eval_indices_path: str | Path | None = DEFAULT_EVAL_INDICES,
    exclude_eval: bool = True,
) -> dict[str, list[tuple[str, str]]]:
    rows = load_manifest_rows(
        manifest_path,
        eval_indices_path=eval_indices_path,
        exclude_eval=exclude_eval,
    )
    safe = [r for r in rows if _is_safe_complied(r)]
    unsafe_refused = [r for r in rows if _is_harmful_refused(r)]
    unsafe_complied = [r for r in rows if _is_harmful_complied(r)]

    rng = random.Random(seed)
    pools = {
        "ss": _pairs(safe),
        "us": _pairs(unsafe_refused),
        "uu": _pairs(unsafe_complied),
    }
    missing = [k for k, v in pools.items() if not v]
    if missing:
        raise ValueError(
            f"manifest has empty RepBend pool(s): {missing}. "
            "Need safe complied, harmful refused, and harmful complied rows."
        )
    return {k: _sample_with_replacement(v, num_examples, rng=rng)
            for k, v in pools.items()}


def manifest_xboundary_pools(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    num_retain: int,
    boundary_data_size: int,
    seed: int = 3333,
    eval_indices_path: str | Path | None = DEFAULT_EVAL_INDICES,
    exclude_eval: bool = True,
    boundary_completion: str = DEFAULT_SAFE_COMPLETION,
) -> dict[str, list]:
    rows = load_manifest_rows(
        manifest_path,
        eval_indices_path=eval_indices_path,
        exclude_eval=exclude_eval,
    )
    erase = [r for r in rows if _is_harmful_complied(r)]
    retain = [r for r in rows if _is_safe_complied(r)] + [
        r for r in rows if _is_harmful_refused(r)
    ]
    boundary = [r for r in rows if _is_overrefused(r)]

    rng = random.Random(seed)
    erase_pairs = _pairs(erase)
    retain_pairs = _pairs(retain)
    boundary_pairs = [
        {
            "prompt": _prompt(r),
            "overrefusal_response": _response(r),
            "retain_response": boundary_completion,
        }
        for r in boundary
        if _prompt(r) and _response(r)
    ]
    missing = []
    if not erase_pairs:
        missing.append("erase")
    if not retain_pairs:
        missing.append("retain")
    if boundary_data_size > 0 and not boundary_pairs:
        missing.append("boundary")
    if missing:
        raise ValueError(f"manifest has empty X-Boundary pool(s): {missing}")

    return {
        "erase": _sample_with_replacement(erase_pairs, max(1, num_retain), rng=rng),
        "retain": _sample_with_replacement(retain_pairs, max(1, num_retain), rng=rng),
        "boundary": _sample_with_replacement(
            boundary_pairs, max(0, boundary_data_size), rng=rng
        ),
    }


def src_repbend_pools(*, num_examples: int, seed: int = 42) -> dict[str, list[tuple[str, str]]]:
    """Fallback based on project dataset loaders only, not paper data."""
    ensure_repo_root_on_path()
    from src.dataset import load_advbench, load_alpaca, load_harmbench

    rng = random.Random(seed)
    benign = [
        (s.prompt, str(s.meta.get("output", "")).strip())
        for s in load_alpaca(max(num_examples * 2, num_examples), seed=seed)
        if s.prompt and str(s.meta.get("output", "")).strip()
    ]
    adv = [
        (s.prompt, str(s.meta.get("target", "")).strip())
        for s in load_advbench(max(num_examples * 2, num_examples), seed=seed)
        if s.prompt and str(s.meta.get("target", "")).strip()
    ]
    harmful_prompts = [s.prompt for s in load_harmbench(num_examples, seed=seed) if s.prompt]
    harmful_prompts += [p for p, _ in adv]
    refused = [(p, DEFAULT_REFUSAL_RESPONSE) for p in harmful_prompts]

    pools = {"ss": benign, "us": refused, "uu": adv}
    missing = [k for k, v in pools.items() if not v]
    if missing:
        raise ValueError(f"src fallback has empty RepBend pool(s): {missing}")
    return {k: _sample_with_replacement(v, num_examples, rng=rng)
            for k, v in pools.items()}


def src_xboundary_pools(
    *,
    num_retain: int,
    boundary_data_size: int,
    seed: int = 3333,
    boundary_completion: str = DEFAULT_SAFE_COMPLETION,
) -> dict[str, list]:
    """Fallback based on project dataset loaders only, not paper data."""
    ensure_repo_root_on_path()
    from src.dataset import (
        load_advbench,
        load_alpaca,
        load_harmbench,
        load_or_bench_orr_train,
        load_xstest,
    )

    rng = random.Random(seed)
    adv = [
        (s.prompt, str(s.meta.get("target", "")).strip())
        for s in load_advbench(max(num_retain, 512), seed=seed)
        if s.prompt and str(s.meta.get("target", "")).strip()
    ]
    benign = [
        (s.prompt, str(s.meta.get("output", "")).strip())
        for s in load_alpaca(max(num_retain, 512), seed=seed)
        if s.prompt and str(s.meta.get("output", "")).strip()
    ]
    harmful_refuse = [
        (s.prompt, DEFAULT_REFUSAL_RESPONSE)
        for s in load_harmbench(max(128, num_retain // 4), seed=seed)
        if s.prompt
    ]
    safe_sensitive = load_xstest(boundary_data_size, seed=seed, safe=True)
    if len(safe_sensitive) < boundary_data_size:
        safe_sensitive += load_or_bench_orr_train(
            boundary_data_size - len(safe_sensitive), seed=seed
        )
    boundary = [
        {
            "prompt": s.prompt,
            "overrefusal_response": DEFAULT_REFUSAL_RESPONSE,
            "retain_response": boundary_completion,
        }
        for s in safe_sensitive
        if s.prompt
    ]

    pools = {
        "erase": adv,
        "retain": benign + harmful_refuse,
        "boundary": boundary,
    }
    missing = [k for k, v in pools.items() if k != "boundary" and not v]
    if boundary_data_size > 0 and not boundary:
        missing.append("boundary")
    if missing:
        raise ValueError(f"src fallback has empty X-Boundary pool(s): {missing}")
    return {
        "erase": _sample_with_replacement(pools["erase"], max(1, num_retain), rng=rng),
        "retain": _sample_with_replacement(pools["retain"], max(1, num_retain), rng=rng),
        "boundary": _sample_with_replacement(
            pools["boundary"], max(0, boundary_data_size), rng=rng
        ),
    }

