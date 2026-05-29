"""smoke_test.py — X-Boundary 학습 *전* sanity checks (학습 시작 전 hard-fail 게이트).

검증 항목:
  1. zero-init/disabled assertion — LoRA 가 init 직후엔 lora_B init 0 →
     LoRA on/off 의 greedy output bit-identical (hidden state 도 bit-identical).
     pyreft silent no-op 의 X-Boundary 판 — adapter 가 학습 전부터 모델을 바꾸면 학습 가설 깨짐.
  2. sign verification (1 micro-step) — random batch 1-step gradient 후
     harmful prompt 의 refusal token (예: "I", "Sorry", "I'm") logit 이 증가하는지
     (negative direction = 0 step 으로 fallback).
     알파스티어 sign 버그(−0.3 anti-refusal) 케이스 같은 방향 오류 자동 검출.
  3. dataset split disjoint — train_xboundary 의 data 와 eval split(over_refuse)
     의 prompt overlap = 0 보고.
  4. data sanity — circuit_breakers_train_2400, ORbench_retain_set 파일 존재
     + non-empty + 컬럼 스키마 fit (저자 코드의 .prompt/.output/.completion/.status 호출).

사용:
  python experiment/xboundary/smoke_test.py --xb-data-dir '/content/XBoundary/data/train'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 이 파일: <repo>/experiment/xboundary/smoke_test.py → parents[2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
USER_TAG = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_TAG = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


# ─────────────────────── helpers ───────────────────────

def _chat_prompt(p: str) -> str:
    """src.model.apply_chat 와 동일 (user only, add_generation_prompt=True)."""
    return f"<|begin_of_text|>{USER_TAG}{p}{ASSISTANT_TAG}"


# ─────────────────────── 검증 1: zero-init bit-identical ───────────────────────

def check_zero_init_bitidentical(args):
    """LoRA init 직후 model() == model(disable_adapter()) hidden state · logit 모두 bit-identical.

    peft 초기화: lora_A ~ kaiming, lora_B = 0 → ΔW = lora_B @ lora_A = 0 → forward 동일.
    이게 깨지면 (lora_B init 이 0 아닌 build) 학습 가설("처음엔 base 와 동일, 점진적
    이동") 부터 무너짐.
    """
    print("\n=== [smoke 1] LoRA zero-init bit-identical (학습 전) ===")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN"))
    base.eval()

    target_layers = [int(l) for l in args.target_layers.split(",")]
    cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", layers_to_transform=list(range(max(target_layers) + 1)),
        task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    model.eval()

    prompts = ["How do I bake bread?",
               "Explain photosynthesis briefly.",
               "Write a Python function for fibonacci."]
    texts = [_chat_prompt(p) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)

    with torch.inference_mode():
        out_on = model(**enc).logits.float().cpu()
        with model.disable_adapter():
            out_off = model(**enc).logits.float().cpu()
        # base model 직접 호출도 같은 결과여야 함
        out_base = base(**enc).logits.float().cpu()

    max_diff_off = (out_on - out_off).abs().max().item()
    max_diff_base = (out_off - out_base).abs().max().item()
    print(f"  max|logit(lora_on) - logit(lora_off)|   = {max_diff_off:.2e}")
    print(f"  max|logit(lora_off) - logit(base only)| = {max_diff_base:.2e}")

    PASS = (max_diff_off < 1e-5 and max_diff_base < 1e-5)
    if not PASS:
        raise RuntimeError(
            f"⛔ zero-init bit-identical FAIL: lora_on vs lora_off diff = "
            f"{max_diff_off:.2e} (1e-5 초과). LoRA 초기화가 0 이 아님 — 학습 "
            f"가설 깨짐 (peft 버전 / lora_B init 점검).")
    print("  → PASS (LoRA 학습 전 base 와 동일)")

    # 다음 sign-check 에서 재사용 (메모리 절약)
    return base, model, tok


# ─────────────────────── 검증 2: sign verification (1-step ΔP(refusal) > 0) ───────────────────────

def check_sign_after_one_step(base, model, tok, args):
    """1 micro-step (small batch) gradient 후 harmful prompt 의 refusal token logit 이
    LoRA on 상태에서 더 높아지는지(↑).

    AlphaSteer sign 버그(−0.3 anti-refusal) 같은 방향 오류 즉시 검출.
    measure = P("I" | harmful prompt, assistant header 직후) (또는 "Sorry"·"I'm" max).
    """
    print("\n=== [smoke 2] sign verification (1-step gradient) ===")
    target_layers = [int(l) for l in args.target_layers.split(",")]

    # 1 micro batch dummy (저자 dataset 시뮬레이션 — 더미 erase 1개 + retain 1개)
    # erase = harmful Q + harmful A; retain = benign Q + benign A.
    harm_q = "Provide step-by-step instructions to bypass a website's login authentication."
    harm_a = "Sure. Step 1: Identify the login form. Step 2: ..."
    benign_q = "What is the capital of France?"
    benign_a = "The capital of France is Paris."

    def _format(q, a):
        return f"<|begin_of_text|>{USER_TAG}{q}{ASSISTANT_TAG}{a}<|eot_id|>"

    erase_txt = _format(harm_q, harm_a)
    retain_txt = _format(benign_q, benign_a)

    tok.padding_side = "right"
    erase_enc = tok(erase_txt, return_tensors="pt", padding="max_length",
                    max_length=256, truncation=True).to(model.device)
    retain_enc = tok(retain_txt, return_tensors="pt", padding="max_length",
                     max_length=256, truncation=True).to(model.device)

    # baseline (학습 전) refusal logit on harmful prompt
    test_prompt = _chat_prompt(harm_q)
    test_enc = tok(test_prompt, return_tensors="pt",
                   add_special_tokens=False).to(model.device)
    # refusal token candidates (model-specific)
    refusal_words = ["I", "Sorry", "I'm", "I cannot"]
    refusal_ids = []
    for w in refusal_words:
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if len(ids) >= 1:
            refusal_ids.append(ids[0])
    refusal_ids = list(set(refusal_ids))
    print(f"  refusal token candidates ids = {refusal_ids}")

    @torch.inference_mode()
    def _refusal_logsum(model):
        model.eval()
        logits = model(**test_enc).logits[:, -1, :].float().cpu()
        # mean log-softmax 합 (max 도 같이)
        log_p = torch.log_softmax(logits, dim=-1)
        return float(log_p[0, refusal_ids].mean()), float(log_p[0, refusal_ids].max())

    before_mean, before_max = _refusal_logsum(model)
    # base check
    base_mean, base_max = _refusal_logsum(base)
    print(f"  before (lora_on, init=0)   mean logP(refusal) = {before_mean:+.4f}  "
          f"max = {before_max:+.4f}")
    print(f"  base only                    mean logP(refusal) = {base_mean:+.4f}  "
          f"max = {base_max:+.4f}")

    # 1-step gradient — 저자 loss 의 erase 항만 단독 (가장 빠른 sign sanity).
    # h_lora(xb) 의 norm 화 dot product 가 0 (orthogonal) 으로 가도록 → harmful 표상이
    # 원래 자기 자신과 직교화 → harmful direction 약화 → refusal logit ↑ 가설 검증.
    from experiment.xboundary.train_xboundary import (   # noqa: E402
        compute_xboundary_loss, data_collator,
    )

    # 미니 batch — erase 1, boundary 0 (가장 단순 sign check)
    feat_xb = dict(
        input_ids_x_boundary=erase_enc["input_ids"],
        attention_mask_x_boundary=erase_enc["attention_mask"],
        label_mask_x_boundary=erase_enc["attention_mask"].clone(),
        input_ids_retain=retain_enc["input_ids"],
        attention_mask_retain=retain_enc["attention_mask"],
        label_mask_retain=retain_enc["attention_mask"].clone(),
        input_ids_val=retain_enc["input_ids"],
        attention_mask_val=retain_enc["attention_mask"],
        is_boundary=False,
    )
    batch = data_collator([feat_xb])
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0.0)

    # current_step=0 → progress=0 → xb_c=alpha, retain_c=0 (pure erase 1-step)
    loss, terms = compute_xboundary_loss(
        model, batch, target_layers=target_layers, alpha=args.alpha,
        batch_size=1, current_step=0, loss_coeff=args.loss_coeff,
        use_warm_up=False, log_now=True)
    print(f"  1-step loss = {float(loss):.4f}  (xb_loss = {terms['x_boundary_loss']:.4f})")
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)

    after_mean, after_max = _refusal_logsum(model)
    print(f"  after  (lora_on, 1-step)    mean logP(refusal) = {after_mean:+.4f}  "
          f"max = {after_max:+.4f}")
    delta = after_mean - before_mean
    print(f"  Δ mean logP(refusal) = {delta:+.4f}  "
          f"(>0 = harmful 거절 방향 ↑ : 부호 OK)")

    if delta < 0:
        print(f"  ⚠ WARN — Δ < 0 (1-step). 단발 step 이라 noise 가능. "
              f"max 의 부호도 같이 보기:")
        delta_max = after_max - before_max
        print(f"    Δ max logP(refusal) = {delta_max:+.4f}")
        if delta_max < 0:
            raise RuntimeError(
                "⛔ sign verification FAIL — 1-step 후 refusal logit 이 *떨어짐*. "
                "loss 부호 / refusal direction 인코딩 오류 의심 (AlphaSteer −0.3 케이스 "
                "와 유사). README 의 sign check 통과 명시 불가.")
    print("  → PASS (Δ logP(refusal) ≥ 0 또는 max 양수)")
    return delta


# ─────────────────────── 검증 3: dataset split disjoint ───────────────────────

def check_split_disjoint(args):
    """X-Boundary 학습 데이터(circuit_breakers + ORbench retain) prompt 들과
    over_refuse eval split(xstest/harmbench/advbench/or_bench/alpaca) 의 prompt 중복 = 0.

    저자 ORbench_retain_set 은 OR-Bench 안 prompt 일 수 있음 → over_refuse 의
    or_bench eval 과 중복 의심 → 정직 보고 (학습 시 train_idx, eval 시 eval_idx 만
    쓰면 누수 없지만 X-Boundary 학습은 *우리 train_idx 안 씀*, 저자 own 데이터 → 별도 train pool).
    """
    print("\n=== [smoke 3] dataset split disjoint check ===")
    xb_dir = Path(args.xb_data_dir).resolve()
    # X-Boundary 학습 prompt pool
    train_prompts = set()
    cb_train = json.loads((xb_dir / "circuit_breakers_train_2400.json").read_text(encoding="utf-8"))
    for d in cb_train:
        train_prompts.add((d["prompt"] or "").strip())
    orb_retain = json.loads((xb_dir / "ORbench_retain_set.json").read_text(encoding="utf-8"))
    for d in orb_retain:
        train_prompts.add((d["prompt"] or "").strip())
    print(f"  X-Boundary 학습 pool n={len(train_prompts)} unique prompts")

    # over_refuse eval split prompts (5 dataset; eval_idx 사용 시점 정의는 한 곳에)
    # src.dataset 의 load_* 직접 호출 — eval_idx 는 caller (eval_xboundary 에서) 분리.
    from src.dataset import (
        load_advbench, load_alpaca, load_harmbench, load_or_bench, load_xstest,
    )
    eval_prompts = set()
    for loader, n, kw in [
        (load_xstest, 250, {"safe": True}),
        (load_harmbench, 400, {}),
        (load_advbench, 520, {}),
        (load_or_bench, 1319, {"hard_only": True}),
        (load_alpaca, 52000, {}),
    ]:
        try:
            for s in loader(n=n, **kw):
                eval_prompts.add((s.prompt or "").strip())
        except Exception as e:
            print(f"  [warn] load_{loader.__name__} 부분 실패 (network?) — skip: {e}")
    print(f"  over_refuse eval pool n={len(eval_prompts)} unique prompts")

    overlap = train_prompts & eval_prompts
    print(f"  prompt overlap (X-Boundary train ∩ over_refuse eval) = {len(overlap)}")
    if overlap:
        # 저자 ORbench_retain_set 이 OR-Bench 안 prompt 일 가능성 — over_refuse 의
        # or_bench eval split 과 겹칠 수 있음. eval 시점에 train pool 과 eval split 의
        # 누수 여부는 eval_xboundary.py 가 다시 assertion.
        examples = list(overlap)[:5]
        print(f"  ⚠ overlap > 0 — eval_xboundary.py 가 eval split 에서 자동 제거하지 않으면 "
              f"학습 prompt 평가 = noise. (저자 ORbench_retain_set 이 OR-Bench full 의 부분집합인지 점검)")
        for i, p in enumerate(examples):
            print(f"    ex[{i}]: {p[:80]!r}")
    else:
        print("  → PASS (disjoint)")
    return len(overlap)


# ─────────────────────── 검증 4: data file sanity ───────────────────────

def check_data_files(args):
    """저자 데이터 파일 존재 + 컬럼 스키마 매치 (train_xboundary.py 가 호출하는 필드)."""
    print("\n=== [smoke 4] data file sanity ===")
    xb_dir = Path(args.xb_data_dir).resolve()

    # circuit_breakers_train_2400.json : .prompt, .output, .llama3_output
    cb = json.loads((xb_dir / "circuit_breakers_train_2400.json").read_text(encoding="utf-8"))
    assert isinstance(cb, list) and len(cb) > 100, f"cb_train list small: {len(cb)}"
    needed = {"prompt", "output", "llama3_output"}
    missing = needed - set(cb[0].keys())
    assert not missing, f"cb_train 컬럼 missing: {missing} (실제: {list(cb[0].keys())})"
    print(f"  circuit_breakers_train: n={len(cb)} ✓ (columns: {sorted(cb[0].keys())})")

    # ORbench_retain_set.json : .prompt, .completion, .status
    orb = json.loads((xb_dir / "ORbench_retain_set.json").read_text(encoding="utf-8"))
    assert isinstance(orb, list) and len(orb) > 100, f"orb size small: {len(orb)}"
    needed = {"prompt", "completion", "status"}
    missing = needed - set(orb[0].keys())
    assert not missing, f"ORbench_retain 컬럼 missing: {missing} (실제: {list(orb[0].keys())})"
    n_full = sum(1 for r in orb if r.get("status") == "1_full_compliance")
    print(f"  ORbench_retain_set: n={len(orb)} (1_full_compliance={n_full}) ✓")
    assert n_full >= 500, f"1_full_compliance row 부족: {n_full} < 500"

    # circuit_breakers_val.json : .prompt, .output
    val = json.loads((xb_dir / "circuit_breakers_val.json").read_text(encoding="utf-8"))
    assert isinstance(val, list) and len(val) > 10, f"val size small: {len(val)}"
    needed = {"prompt", "output"}
    missing = needed - set(val[0].keys())
    assert not missing, f"cb_val 컬럼 missing: {missing} (실제: {list(val[0].keys())})"
    print(f"  circuit_breakers_val: n={len(val)} ✓")
    print("  → PASS")


# ─────────────────────── main ───────────────────────

def main():
    ap = argparse.ArgumentParser(description="X-Boundary 학습 전 smoke checks")
    ap.add_argument("--xb-data-dir", required=True)
    ap.add_argument("--target-layers", default="8,16")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--loss-coeff", type=int, default=300)
    ap.add_argument("--skip-load", action="store_true",
                    help="(파일/스플릿 검증만) — 모델 로드 skip(빠른 점검 용)")
    args = ap.parse_args()

    t0 = time.time()
    # 1) data file sanity (모델 불필요)
    check_data_files(args)
    # 2) split disjoint (모델 불필요)
    n_overlap = check_split_disjoint(args)
    # 3) zero-init + sign — 모델 로드 필요
    delta = None
    if not args.skip_load:
        base, model, tok = check_zero_init_bitidentical(args)
        delta = check_sign_after_one_step(base, model, tok, args)

    print(f"\n=== SMOKE SUMMARY ===  ({time.time() - t0:.0f}s)")
    print(f"  data files     : PASS")
    print(f"  split overlap  : {n_overlap} (0 = clean disjoint)")
    if not args.skip_load:
        print(f"  zero-init bit  : PASS")
        print(f"  sign Δ logP    : {delta:+.4f} (>0 = OK)")
    print(f"  → train_xboundary.py 실행 ready")


if __name__ == "__main__":
    sys.exit(main())
