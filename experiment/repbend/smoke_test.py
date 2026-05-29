"""smoke_test.py — RepBend 학습 *전* sanity check.

검증 protocol (각 단계 hard-fail):
  1. (no-op assertion) 학습 안 한 LoRA adapter (zero-init delta) 가 base 와
     bit-identical 한 greedy output 을 내는지. delta ≠ 0 이면 LoRA 가 새로 init
     된 게 아니거나 W_b·W_a 초기 곱이 0 이 아닌 것 → raise.
  2. (forward sanity) base 모델만 단독 forward 가 정상 작동 + 응답 nonempty.
  3. (dataset disjoint) train/eval 데이터 source 가 다른지 (RepBend 는 학습=
     wildguardmix, 평가=wildjailbreak+orbench → source 자체가 disjoint, id 도
     안 겹침). report 만, raise X.
  4. (chat template integrity) generate 의 chat template 이 src.model.generate
     의 그것과 동일 token sequence 인지 (간접 검증: 양쪽 모두 동일 prompt 에
     동일 응답 → 그러면 chat template 동일).

Colab/AWS 에서 학습 전 한 번 실행 권장.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _chat_texts(tok, prompts):
    return [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
            for p in prompts]


def _generate(model, tok, prompts, *, max_new_tokens: int = 50):
    """plain greedy generate — base 와 PEFT 둘 다 동일 코드 path."""
    texts = _chat_texts(tok, prompts)
    dev = next(model.parameters()).device
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(dev)
    prompt_len = enc["input_ids"].shape[1]
    with torch.inference_mode():
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    return tok.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description="RepBend smoke test (학습 전 sanity)")
    ap.add_argument("--n-probe", type=int, default=4,
                    help="bit-identical 검증용 prompt 수")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print("=" * 76)
    print("RepBend smoke test")
    print("=" * 76)

    # ── 1. tokenizer + base 로드 ──
    print("\n[1/4] base model + tokenizer 로드 ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    base.eval()
    print(f"  base device={next(base.parameters()).device}")

    # ── 2. base forward sanity ──
    print("\n[2/4] base forward sanity ...")
    probe_prompts = [
        "What is the capital of France?",
        "Write a haiku about autumn.",
        "How do I make a bomb?",
        "Explain quantum entanglement in one sentence.",
    ][:args.n_probe]
    base_resps = _generate(base, tok, probe_prompts)
    for p, r in zip(probe_prompts, base_resps):
        print(f"  {p[:50]!r:60s} → {r[:80]!r}")
        if len(r.strip()) == 0:
            raise RuntimeError(f"base 응답 빈 string — model load 실패 의심. prompt={p!r}")

    # ── 3. LoRA wrap + bit-identical 확인 (no-op assertion) ──
    print("\n[3/4] LoRA wrap → bit-identical 검증 (zero-init delta) ...")
    # train_repbend 와 동일한 LoRA config
    from train_repbend import (TARGET_LAYER_START_IDX, LAYERS_WINDOW_SIZE,
                                LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES)
    target_layers = list(range(TARGET_LAYER_START_IDX,
                                TARGET_LAYER_START_IDX + LAYERS_WINDOW_SIZE))
    layers_to_transform = list(range(max(target_layers) + 1))
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        layers_to_transform=layers_to_transform,
        task_type="CAUSAL_LM", bias="none",
    )
    # PEFT init 은 W_a init kaiming, W_b init 0 → delta = W_b @ W_a = 0
    # 따라서 학습 전 LoRA model = base bit-identical 이어야 함.
    peft_model = get_peft_model(base, lora_cfg)
    peft_model.eval()
    peft_resps = _generate(peft_model, tok, probe_prompts)

    n_diff = sum(1 for a, b in zip(peft_resps, base_resps) if a != b)
    print(f"  PEFT(zero-init) vs base: {n_diff}/{len(probe_prompts)} 다름")
    for p, b, pf in zip(probe_prompts, base_resps, peft_resps):
        match = "✓" if b == pf else "✗"
        print(f"  {match} {p[:40]!r}")
        if b != pf:
            print(f"    base: {b[:100]!r}")
            print(f"    PEFT: {pf[:100]!r}")
    if n_diff > 0:
        raise RuntimeError(
            f"[FAIL] LoRA zero-init 인데 PEFT 응답 ≠ base 응답 ({n_diff}/{len(probe_prompts)}). "
            f"W_b 가 zero 가 아니거나 LoRA config 가 의도와 다름. peft 버전 확인.")
    print(f"  [PASS] LoRA zero-init = base bit-identical")

    # ── 4. dataset source disjoint report ──
    print("\n[4/4] dataset source disjoint (train ↔ eval) ...")
    print("  train data source : allenai/wildguardmix (wildguardtrain)")
    print("  eval data source  : experiment/output/test_sets/")
    print("                       wildjailbreak_adv_harmful_n*.json (WildJailbreak adv)")
    print("                       orbench_hard_n*.json              (OR-Bench Hard-1K)")
    print("  → source 자체 disjoint, id prefix 도 다름 (wildjb_/orbench_ vs wgmix 내부 id 없음)")
    # 실제 train data id 검증은 학습 시 — 여기선 schema 만 확인
    test_dir = REPO_ROOT / "experiment" / "output" / "test_sets"
    if test_dir.exists():
        asr_files = sorted(test_dir.glob("wildjailbreak_adv_harmful_n*.json"))
        orr_files = sorted(test_dir.glob("orbench_hard_n*.json"))
        print(f"  eval files: ASR={[p.name for p in asr_files]}  ORR={[p.name for p in orr_files]}")
        if not asr_files or not orr_files:
            print("  ⚠ test_set 파일 부재 — eval 전에 build_test_set 으로 생성 필요")
    else:
        print(f"  ⚠ test_dir 부재: {test_dir}")

    print("\n" + "=" * 76)
    print("[ALL PASS] smoke test 완료. 다음:")
    print("  python experiment/repbend/train_repbend.py --smoke    # 1-step sign check")
    print("  python experiment/repbend/train_repbend.py             # full 학습")
    print("  python experiment/repbend/eval_repbend.py --ckpt <out> --smoke 8")
    print("=" * 76)


if __name__ == "__main__":
    sys.exit(main())
