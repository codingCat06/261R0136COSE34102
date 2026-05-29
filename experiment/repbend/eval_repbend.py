"""eval_repbend.py — 학습된 RepBend LoRA adapter 를 base 위에 얹어 eval split
응답 생성. dump 결과 schema `[{id, prompt, response, refused, source,
category, expected}]` — X-Boundary 와 동일 (v-series head-to-head 호환).

WildGuard judge 는 *별 셀에서* 호출 (memory 'no-gpt4o-judge' — GPT-4o 절대
금지). 이 스크립트는 응답 생성 + greedy deviation 측정 + base 비교까지만.

사용:
  python experiment/repbend/eval_repbend.py \
      --ckpt experiment/output/repbend_3b/ckpt \
      --out experiment/output/resp_repbend_3b.json \
      [--per-source-n 100] [--train-n 64] [--max-new-tokens 384] [--batch-size 8]

검증:
  - dump 후 schema 검증 (id/prompt/response/source/category/expected 필드).
  - --compare-base 면 동일 prompt 에 base 응답도 같이 생성 → greedy deviation
    측정 (학습이 *실제로* 모델 행동을 바꿨는지). diff=0 이면 hard-fail.
  - 입력 = src.dataset.load_{xstest,harmbench,advbench,or_bench,alpaca}() 의
    eval_idx (= full[train_n: train_n + per_source_n]).  v-series 와 동일 split.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
GEN_MAX_NEW_TOKENS = 384
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_N = 64           # over_refuse 표준 (notebooks/01 변수)
DEFAULT_PER_SOURCE_N = 100     # eval_idx = [train_n: train_n + per_source_n]


# ─────────────────────── eval split (5 dataset, X-Boundary 동일) ───────────────────────

def load_eval_pool(*, per_source_n: int, train_n: int = DEFAULT_TRAIN_N) -> list[dict]:
    """5 dataset 의 eval_idx 만 합쳐 list 반환. X-Boundary eval_xboundary 와 동일 패턴.

    schema: [{id, prompt, source, expected, category}]
      expected ∈ {refuse, answer}; category ∈ {harmful, safe_sensitive, benign}
    """
    sys.path.insert(0, str(REPO_ROOT))
    from src.dataset import (
        load_advbench, load_alpaca, load_harmbench, load_or_bench, load_xstest,
    )

    def _eval_slice(loader, source, **kw):
        full = loader(n=train_n + per_source_n, **kw)
        return full[train_n: train_n + per_source_n], source

    pool: list[dict] = []
    for loader, source, kw in [
        (load_xstest, "xstest_safe", {"safe": True}),
        (load_harmbench, "harmbench", {}),
        (load_advbench, "advbench", {}),
        (load_or_bench, "or_bench", {"hard_only": True}),
        (load_alpaca, "alpaca", {}),
    ]:
        slice_, src_name = _eval_slice(loader, source, **kw)
        for s in slice_:
            pool.append(dict(
                id=s.id, prompt=s.prompt, source=src_name,
                expected=s.expected, category=s.category,
            ))
    print(f"[eval-pool] n={len(pool)}  (5 dataset eval_idx, per_source_n={per_source_n})")
    return pool


def _chat_texts(tok, prompts):
    """src.model.apply_chat 와 동일: single user turn, add_generation_prompt=True."""
    return [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
            for p in prompts]


def generate(model, tok, prompts, *, batch_size: int, max_new_tokens: int) -> list[str]:
    """src.model.generate 와 동일 디코딩 (greedy do_sample=False, left pad)."""
    texts = _chat_texts(tok, prompts)
    outs = []
    bs = max(1, batch_size)
    n_batches = (len(texts) + bs - 1) // bs
    dev = next(model.parameters()).device
    for bi, i in enumerate(range(0, len(texts), bs)):
        chunk = texts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        prompt_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        outs.extend(tok.batch_decode(gen[:, prompt_len:], skip_special_tokens=True))
        print(f"  [gen] batch {bi + 1}/{n_batches} ({len(outs)}/{len(texts)})", flush=True)
    if len(outs) != len(prompts):
        raise RuntimeError(f"응답 {len(outs)} ≠ prompt {len(prompts)}")
    return outs


def load_repbend_model(ckpt_dir: str):
    """base Llama-3.2-3B + 학습된 LoRA adapter merge.

    learning-rate 0 이거나 학습 안 한 ckpt 라도 LoRA weight 가 zero init 됐으면
    base 와 *bit-identical* 한 forward 가 나와야 함 (smoke_test 의 'no-op'
    가정). 그래서 우리는 merge_and_unload 하지 않고 PEFT model 로 유지 —
    `--disable-adapter` 옵션으로 base 직접 호출도 가능.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(ckpt_dir if Path(ckpt_dir).exists()
                                         else MODEL_ID,
                                         token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval()
    return model, tok


def generate_base(prompts, *, batch_size: int, max_new_tokens: int) -> list[str]:
    """plain base 모델 (LoRA 없음) — smoke / --compare-base 용."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    base.eval()
    out = generate(base, tok, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens)
    del base
    import gc; gc.collect(); torch.cuda.empty_cache()
    return out


def save_atomic(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _validate_schema(per_sample: list[dict]):
    """schema 고정: {id, prompt, response, source, category, expected} +
    refused (WildGuard eval 후 채움). X-Boundary 와 동일 schema.
    """
    required = {"id", "prompt", "response", "source", "category", "expected"}
    for i, r in enumerate(per_sample):
        miss = required - set(r.keys())
        if miss:
            raise RuntimeError(f"schema violation @{i}: missing {miss}, keys={list(r.keys())}")
        if not isinstance(r["response"], str):
            raise RuntimeError(f"schema violation @{i}: response not str (type={type(r['response'])})")
    print(f"  [schema] OK ({len(per_sample)} rows, keys={sorted(per_sample[0].keys())})")


def main():
    ap = argparse.ArgumentParser(description="RepBend eval — LoRA adapter inference (5 dataset eval_idx)")
    ap.add_argument("--ckpt", required=True, help="train_repbend.py 가 저장한 ckpt 폴더")
    ap.add_argument("--out", default=str(REPO_ROOT / "experiment" / "output" / "resp_repbend_3b.json"))
    ap.add_argument("--per-source-n", type=int, default=DEFAULT_PER_SOURCE_N,
                    help="dataset 당 eval sample 수 (X-Boundary 와 통일 권장)")
    ap.add_argument("--train-n", type=int, default=DEFAULT_TRAIN_N,
                    help="over_refuse 표준 train_n (notebooks/01) — eval_idx = [train_n:]")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    ap.add_argument("--smoke", type=int, default=0,
                    help=">0 이면 앞 N개만 + base 와 deviation 측정")
    ap.add_argument("--compare-base", action="store_true",
                    help="동일 prompt 에 base 응답도 생성 → greedy deviation 측정")
    args = ap.parse_args()

    t0 = time.time()
    items = load_eval_pool(per_source_n=args.per_source_n, train_n=args.train_n)
    if args.smoke > 0:
        items = items[:args.smoke]
        # resp_ prefix → smoke_ 로 (run_eval glob 회피)
        _op = Path(args.out)
        if _op.name.startswith("resp_"):
            args.out = str(_op.with_name("smoke_" + _op.name[len("resp_"):]))
        print(f"[SMOKE] 앞 {len(items)}개만 → {args.out}")

    prompts = [it["prompt"] for it in items]
    model, tok = load_repbend_model(args.ckpt)
    responses = generate(model, tok, prompts,
                          batch_size=args.batch_size,
                          max_new_tokens=args.max_new_tokens)

    per_sample = [{
        "id": it.get("id"),
        "prompt": it["prompt"],
        "response": resp,
        "refused": None,                              # WildGuard 후 채움
        "source": it["source"],                       # xstest_safe / harmbench / advbench / or_bench / alpaca
        "category": it["category"],                   # harmful / safe_sensitive / benign
        "expected": it["expected"],                   # refuse / answer
    } for it, resp in zip(items, responses)]
    _validate_schema(per_sample)

    # ── greedy deviation (학습이 실제로 행동을 바꿨는지) ──
    deviation = None
    if args.compare_base or args.smoke > 0:
        print("\n[compare-base] base 응답 생성 중 ...")
        del model; import gc; gc.collect(); torch.cuda.empty_cache()
        base_resps = generate_base(prompts, batch_size=args.batch_size,
                                    max_new_tokens=args.max_new_tokens)
        # token-level Levenshtein(== difflib) 평균 + 완전동일 비율
        n_diff = sum(1 for a, b in zip(responses, base_resps) if a != b)
        ratios = [difflib.SequenceMatcher(None, a, b).ratio()
                  for a, b in zip(responses, base_resps)]
        avg_sim = sum(ratios) / len(ratios) if ratios else 1.0
        deviation = {
            "n_total": len(responses),
            "n_different": n_diff,
            "frac_different": n_diff / max(1, len(responses)),
            "avg_string_similarity": avg_sim,
            "avg_deviation": 1.0 - avg_sim,
        }
        print(f"  base 와 응답 차이: {n_diff}/{len(responses)} ({100 * n_diff / len(responses):.1f}%)")
        print(f"  평균 유사도: {avg_sim:.3f}  (deviation = {1 - avg_sim:.3f})")
        # ⚠ smoke: 차이 0 = 학습 무의미 또는 LoRA 가 silent no-op → hard fail
        if args.smoke > 0 and n_diff == 0:
            raise RuntimeError(
                "[SMOKE FAIL] base 와 응답 완전 일치 — 학습 효과 0 이거나 LoRA "
                "merge silent no-op. ckpt 확인.")
        if args.smoke > 0:
            print(f"  [SMOKE PASS] {n_diff}/{len(responses)} 응답이 base 와 다름.")
        # base 응답도 sample 별 dump (선택적)
        for r, b in zip(per_sample, base_resps):
            r["base_response"] = b

    result = {
        "per_sample": per_sample,
        "config": {
            "phase": "responses",
            "model_id": MODEL_ID,
            "ckpt": str(Path(args.ckpt).resolve()),
            "gen_batch": args.batch_size,
            "gen_max_new_tokens": args.max_new_tokens,
            "n_test": len(items),
            "per_source_n": args.per_source_n,
            "train_n": args.train_n,
            "smoke": args.smoke if args.smoke > 0 else None,
            "method_meta": {
                "method": "RepBend (LoRA + activation-bending loss)",
                "paper": "Representation Bending for Large Language Model Safety (arXiv:2504.01550)",
                "repo": "github.com/AIM-Intelligence/RepBend",
                "loss": "L = α·L_safe − β·L_unsafe + γ·L_cos + ε·L_kl",
                "loss_coefficients": "α=0.5, β=0.5, γ=0.1, ε=0.3, η=0.0",
                "lora_config": "r=16, alpha=16, dropout=0.05, target_modules=all linear",
                "target_layers": "18..27 (3B-adjusted; author 20..30 for 8B)",
                "training_data": "allenai/wildguardmix wildguardtrain (response_all mode, n=1024)",
                "eval_data": "5 dataset eval_idx = [train_n=64 : train_n+per_source_n] of "
                              "load_xstest(safe)/harmbench/advbench/or_bench(hard_only)/alpaca "
                              "(X-Boundary 와 동일, v-series head-to-head 호환)",
                "decoding": "greedy do_sample=False (matches over_refuse harness)",
                "chat_template": "src.model.apply_chat (single user turn, add_generation_prompt=True)",
            },
            "deviation": deviation,
        },
    }

    # 미리보기 (저장 전 — 실패해도 보이게)
    for r in per_sample[:2]:
        print(f"\n--- [{r['source']} / {r['category']} / expect={r['expected']}] "
              f"{str(r['prompt'])[:80]!r}")
        print(f"  → {str(r['response'])[:200]!r}")

    save_atomic(args.out, result)
    dt = time.time() - t0
    print(f"\n[done] n={len(items)}  →  {args.out}  ({dt:.0f}s)")
    print(f"[next] WildGuard judge: 노트북에서 `from src.classifier import classify_refuse;`")
    print(f"         refused_list = classify_refuse(prompts, responses, unload_after=True)")


if __name__ == "__main__":
    sys.exit(main())
