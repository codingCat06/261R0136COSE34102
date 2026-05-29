"""train_xboundary.py — X-Boundary (arXiv:2502.09990) LoRA 학습을 Llama-3.2-3B-Instruct 로 포팅.

이 파일은 **격리 standalone** (Colab 학습 런타임 전용). 프로젝트 `src` / `experiment.common`
을 *import 안 함* — peft/accelerate 환경이 메인 노트북 transformers 와 충돌 가능.
대신 (a) Llama chat template 을 코드로 복제(src.model.generate 와 동일), (b) 학습
데이터는 우리 manifest/cache 를 우선 사용하고 manifest 가 없을 때만 `src.dataset`
fallback 을 사용한다. paper 전용 train json / ultrachat retain 은 기본 경로에서 제외.

저자 lorra_x_boundary.py 의 핵심 algorithm 충실 포팅:
  - LoRA (r=16, alpha=16, dropout=0.05, target = q/k/v/o/gate/up/down_proj) — 저자 동일
  - target_layers = "10,20" → 3B(28 layer) 비례 환산 → "8,16" (저자 8B 의 10/32, 20/32 ≈ 8/28, 16/28)
  - transform_layers = -1 → "0..max(target_layers)" 모든 LoRA layer transform
  - drop_layers_after = max(target_layers) — 학습 시 그 위 layer 는 forward 안 함 (속도)
  - loss : retain (Frobenius masked) + x_boundary (erase + separate)
      progress = step / loss_coeff (저자 default 300; max_steps 180 < 300 → 학습 끝까지 < 1)
      retain_coeff = alpha * progress
      x_boundary_coeff = alpha * (1 - progress)
      erase    (is_boundary=False, harmful): ReLU(<h_lora_xb_norm, h_orig_xb_norm>)
      separate (is_boundary=True,  overrefuse): ReLU(<h_lora_xb_norm, mean(h_orig_retain_norm)>)
      retain : ||h_lora_retain - h_orig_retain||_2 (masked token only)
  - "reference" hidden = base model 의 LoRA 비활성(disable_adapter) forward — 저자와 동일
  - optimizer = AdamW, lr=1e-4 constant, weight_decay=0, bf16, batch 16, max_steps 180

이전 데이터 (저자 fetch_xboundary.sh 가 clone 한 data/train/):
  - circuit_breakers_train_2400.json — erase set (harmful prompt + harmful completion)
      column: prompt, output (harmful completion, target=멀리하기) + llama3_output (refusal,
      use_refusal_retain 일 때 retain set 에 합쳐짐)
  - ORbench_retain_set.json — boundary set (over-refusal prompt + complied response)
      column: prompt, completion, status (== "1_full_compliance" 만 사용)
  - circuit_breakers_val.json — eval (log_now 시 cos sim 측정)
  - **multi-turn (SafeMT) 미사용** — over_refuse single-turn setup

⚠ 의도적 deviation (README 에 명시):
  - 저자 8B 의 layer 10/20 → 3B 비례환산 8/16 (28 layer 기준 동일 상대위치).
  - SafeMT multi-turn 데이터 미포함 — 우리 setup single-turn.
  - retain 의 ultrachat_200k 는 더 이상 기본 학습 데이터가 아님.
  - chat template = Llama-3.2 instruct (저자 코드의 'llama' 분기와 동일 user/assistant tag).

사용:
  python experiment/xboundary/train_xboundary.py \
      --xb-data-dir '/content/XBoundary/data/train' \
      --output-dir 'experiment/output/xboundary_3b_adapter' \
      [--max-steps 180] [--batch-size 16] [--target-layers 8,16] [--smoke]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.methods.our_training_data import (  # noqa: E402
    DEFAULT_EVAL_INDICES,
    DEFAULT_MANIFEST,
    DEFAULT_SAFE_COMPLETION,
    manifest_xboundary_pools,
    src_xboundary_pools,
)

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
SEED = 3333  # 저자 lorra_x_boundary.py 와 동일


# ─────────────────────── chat template (src.model.apply_chat 와 동일) ───────────────────────
# Llama-3.2 instruct = Llama-3 chat template — 저자 코드 'llama' 분기와 토큰 동일.
USER_TAG = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_TAG = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
ONE_SHOT_TEMPLATE = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"


# ─────────────────────── dataset (저자 XBoundaryDataset 포팅) ───────────────────────

class XBoundaryDataset(Dataset):
    """저자 xb_train_dataset.XBoundaryDataset 의 single-turn(Llama) 포팅.

    구성:
      - x_boundary_orig: list (string 또는 dict)
          string  → erase 샘플 (harmful Q + harmful A; is_boundary=False)
          dict    → boundary 샘플 ({overrefusalQA, retainQA}; is_boundary=True)
                    overrefusalQA = (over-refusal Q + refusal A) — 이걸 separate 하라
                    retainQA      = (over-refusal Q + complied A) — 이건 retain 시켜라
      - orig_s_retain: 우리 benign answer + proper harmful refusal retain
      - val_orig: x_boundary 샘플 prefix (현재 loss 에서는 log placeholder)

    저자 __getitem__ 의 토큰화는 그대로(left pad request, right pad response → concat).
    """

    def __init__(self, tokenizer, *, num_examples: int = 10000,
                 use_refusal_retain: bool = True, boundary_data_size: int = 500,
                 model_name_or_path: str = MODEL_ID,
                 data_source: str = "manifest",
                 manifest_path: str | Path = DEFAULT_MANIFEST,
                 eval_indices_path: str | Path | None = DEFAULT_EVAL_INDICES,
                 exclude_eval: bool = True,
                 boundary_completion: str = DEFAULT_SAFE_COMPLETION):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = 1024
        self.use_refusal_retain = use_refusal_retain
        self.boundary_data_size = boundary_data_size
        self.user_tag = USER_TAG
        self.assistant_tag = ASSISTANT_TAG
        self.sep_token = ""           # 저자 llama 분기와 동일

        # 저자 코드: llama 분기는 switch_select=[0,1] (어떤 샘플은 instruction 비움 — augmentation)
        switch_select = [0, 1]

        if data_source == "manifest":
            pools = manifest_xboundary_pools(
                manifest_path,
                num_retain=num_examples,
                boundary_data_size=boundary_data_size,
                seed=SEED,
                eval_indices_path=eval_indices_path,
                exclude_eval=exclude_eval,
                boundary_completion=boundary_completion,
            )
            source_desc = f"manifest:{manifest_path}"
        elif data_source == "src":
            pools = src_xboundary_pools(
                num_retain=num_examples,
                boundary_data_size=boundary_data_size,
                seed=SEED,
                boundary_completion=boundary_completion,
            )
            source_desc = "src.dataset fallback"
        else:
            raise ValueError(f"unknown data_source={data_source!r}; expected manifest|src")

        # ─── erase set: harmful prompt + harmful/complied response ───
        x_boundary_orig = []
        for prompt, cb_output in pools["erase"]:
            switch = np.random.choice(switch_select)
            instr = prompt if switch == 0 else ""
            x_boundary_orig.append(ONE_SHOT_TEMPLATE.format(
                user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                instruction=instr, response=cb_output))
        random.shuffle(x_boundary_orig)

        # ─── retain set: 우리 benign answer + proper harmful refusals ───
        orig_s = []
        for prompt, response in pools["retain"]:
            switch = np.random.choice(switch_select)
            instr = prompt if switch == 0 else ""
            txt = ONE_SHOT_TEMPLATE.format(
                user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                instruction=instr, response=response)
            orig_s.append(txt)
        random.shuffle(orig_s)

        # ─── boundary (our over-refusal) set ───
        if boundary_data_size > 0:
            for d in pools["boundary"]:
                retain_qa = ONE_SHOT_TEMPLATE.format(
                    user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                    instruction=d["prompt"], response=d["retain_response"])
                overrefusal_qa = ONE_SHOT_TEMPLATE.format(
                    user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                    instruction=d["prompt"], response=d["overrefusal_response"])
                x_boundary_orig.append({
                    "overrefusalQA": overrefusal_qa, "retainQA": retain_qa})
            random.shuffle(x_boundary_orig)
            print(f"[data] boundary set 추가 — total x_boundary 길이 {len(x_boundary_orig)}")

        # use_refusal_retain 은 API 호환용. 우리 retain pool 에 proper refusal 이 이미 포함됨.
        _ = use_refusal_retain

        # ─── val: loss logging placeholder. compute_loss 에서는 현재 사용하지 않음. ───
        val_orig = list(x_boundary_orig[: max(1, min(64, len(x_boundary_orig)))])
        val_orig = [
            v["overrefusalQA"] if isinstance(v, dict) else v
            for v in val_orig
        ]

        self.x_boundary_orig = x_boundary_orig
        self.orig_s_retain = orig_s
        self.val_orig = val_orig
        print(f"[data] source={source_desc}  x_boundary={len(self.x_boundary_orig)}  "
              f"retain={len(self.orig_s_retain)}  val={len(self.val_orig)}")

    def __len__(self):
        return min(len(self.orig_s_retain), len(self.x_boundary_orig))

    def __getitem__(self, i):
        """저자 __getitem__ 그대로 — request(left-pad) + response(right-pad) concat (max_len 1024)."""
        orig_s_retain = self.orig_s_retain[i]
        x_boundary_orig = self.x_boundary_orig[i]
        val_orig = self.val_orig[i % len(self.val_orig)]

        cb_kw = dict(max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        full_kw = dict(max_length=1024, padding="max_length", truncation=True, return_tensors="pt")

        # ─── x_boundary side ───
        is_boundary = False
        if isinstance(x_boundary_orig, dict):
            orig_s_retain = x_boundary_orig["retainQA"]
            x_boundary_orig = x_boundary_orig["overrefusalQA"]
            is_boundary = True
        cb_req, cb_resp = x_boundary_orig.split("<SEPARATOR>")

        self.tokenizer.padding_side = "left"
        tok_req_xb = self.tokenizer(cb_req, **cb_kw)
        self.tokenizer.padding_side = "right"
        tok_resp_xb = self.tokenizer(cb_resp, add_special_tokens=False, **cb_kw)
        self.tokenizer.padding_side = "left"

        ids_xb = torch.cat([tok_req_xb["input_ids"], tok_resp_xb["input_ids"]], dim=1)
        attn_xb = torch.cat([tok_req_xb["attention_mask"], tok_resp_xb["attention_mask"]], dim=1)
        if is_boundary:
            # separate: response 부분만 push (저자: response 토큰만 0 mask)
            # ⚠ 저자 코드 그대로: label_mask = (req_attn, zeros(resp_attn.shape))
            #   → 즉 request 부분만 mask=1 → loss 가 request 토큰에 적용됨.
            #   "over-refusal 의 거절 응답이 retain mean 과 직교하게" 라는 paper 설명과 위치 어긋남이
            #   있지만, 저자 official 코드를 그대로 따름(implementation fidelity).
            label_mask_xb = torch.cat(
                [tok_req_xb["attention_mask"],
                 torch.zeros(tok_resp_xb["attention_mask"].shape)], dim=1)
        else:
            label_mask_xb = attn_xb       # erase: 전체 토큰

        # ─── retain side ───
        tok_retain = self.tokenizer(
            orig_s_retain.replace("<SEPARATOR>", self.sep_token), **full_kw)
        ids_retain = tok_retain["input_ids"]
        attn_retain = tok_retain["attention_mask"]
        label_mask_retain = attn_retain

        # ─── val ───
        tok_val = self.tokenizer(
            val_orig.replace("<SEPARATOR>", self.sep_token), **full_kw)

        return dict(
            input_ids_x_boundary=ids_xb,
            attention_mask_x_boundary=attn_xb,
            label_mask_x_boundary=label_mask_xb,
            input_ids_retain=ids_retain,
            attention_mask_retain=attn_retain,
            label_mask_retain=label_mask_retain,
            input_ids_val=tok_val["input_ids"],
            attention_mask_val=tok_val["attention_mask"],
            is_boundary=is_boundary,
        )


# ─────────────────────── data collator (저자 동일) ───────────────────────

def data_collator(batch):
    out = {}
    for feat in batch:
        for k, v in feat.items():
            out.setdefault(k, []).append(v)
    for k, v in out.items():
        if isinstance(v[0], torch.Tensor):
            out[k] = torch.cat(v, dim=0)
        elif isinstance(v[0], bool):
            out[k] = torch.tensor(v)
        elif isinstance(v[0], int):
            out[k] = torch.tensor(v)
        else:
            raise ValueError(f"unsupported collator type {type(v[0])}")
    return out


# ─────────────────────── X-Boundary loss (저자 compute_loss 포팅) ───────────────────────

def compute_xboundary_loss(model, inputs, target_layers, alpha, batch_size,
                           current_step, loss_coeff=300, use_warm_up=False,
                           log_now=False):
    """저자 lorra_x_boundary.compute_loss 1:1 포팅 (저자 코드 line 단위 대응).

    핵심 식:
      progress     = current_step / loss_coeff  (warmup<30 → 0.001)
      retain_c     = alpha * progress
      xb_c         = alpha * (1 - progress)
      retain_loss  = ‖h_lora(retain)·mask - h_orig(retain)·mask‖_2     (mean)
      x_boundary_loss = per-sample sum / total_mask:
          is_boundary=False: ReLU(<ĥ_lora(xb), ĥ_orig(xb)>)
          is_boundary=True:  ReLU(<ĥ_lora(xb), mean(ĥ_orig(retain))>)
      loss = retain_c * retain_loss + xb_c * x_boundary_loss
    """
    progress = current_step / loss_coeff
    if use_warm_up and current_step < 30:
        progress = 0.001

    retain_coeff = alpha * progress
    x_boundary_coeff = alpha * (1 - progress)

    retain_ids = inputs["input_ids_retain"]
    retain_attn = inputs["attention_mask_retain"]
    retain_lmask = inputs["label_mask_retain"]
    xb_ids = inputs["input_ids_x_boundary"]
    xb_attn = inputs["attention_mask_x_boundary"]
    xb_lmask = inputs["label_mask_x_boundary"]
    is_boundary = inputs["is_boundary"]

    retain_in = dict(input_ids=retain_ids, attention_mask=retain_attn, output_hidden_states=True)
    xb_in = dict(input_ids=xb_ids, attention_mask=xb_attn, output_hidden_states=True)

    # boundary sample 이 batch 에 하나라도 있으면 separate term 의 target 으로 retain reference 가 필요.
    # 저자 코드는 warm_up=True 로 step<30 progress=0.001 강제해 retain_coeff>0 으로 우회 → 우리는 명시.
    has_boundary = bool(is_boundary.any().item()) if x_boundary_coeff > 0 else False
    need_retain_ref = (retain_coeff > 0) or has_boundary

    # ─── reference (LoRA off) ───
    # disable_adapter() 가 peft model 의 LoRA 비활성 (저자 동일)
    layers_xb_attn_mask = xb_lmask.repeat(len(target_layers), 1, 1).unsqueeze(-1)
    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
            if need_retain_ref:
                orig_retain_out = model(**retain_in)["hidden_states"]
                orig_retain_hidden = torch.stack(orig_retain_out).detach()
                layers_retain_attn_mask = retain_lmask.repeat(
                    len(orig_retain_out), 1, 1).unsqueeze(-1)
                orig_retain_hidden_mask = orig_retain_hidden * layers_retain_attn_mask
                del orig_retain_out
                gc.collect()
            if x_boundary_coeff > 0:
                xb_out = model(**xb_in)["hidden_states"]
                xb_hidden = torch.stack([xb_out[l].detach() for l in target_layers])
                del xb_out
                gc.collect()
    model.train()

    retain_loss = torch.tensor(0.0, device=xb_ids.device)
    x_boundary_loss = torch.tensor(0.0, device=xb_ids.device)

    # ─── retain side ───
    if retain_coeff > 0:
        lora_retain_out = model(**retain_in)["hidden_states"]
        lora_retain_hidden_mask = torch.stack(lora_retain_out) * layers_retain_attn_mask
        retain_loss = torch.norm(
            lora_retain_hidden_mask - orig_retain_hidden_mask,
            dim=-1, p=2, dtype=torch.float).nanmean()

    # ─── x_boundary side ───
    if x_boundary_coeff > 0:
        lora_xb_out = model(**xb_in)["hidden_states"]
        lora_xb_hidden = torch.stack([lora_xb_out[l] for l in target_layers])

        norm_lora_xb = lora_xb_hidden / (
            torch.norm(lora_xb_hidden, dim=-1, keepdim=True, dtype=torch.float))
        norm_xb = xb_hidden / (
            torch.norm(xb_hidden, dim=-1, keepdim=True, dtype=torch.float))

        # target_orig_retain : retain 측 reference 의 target_layers 만 (boundary 케이스용).
        # batch 에 boundary sample 없으면 사용 안 되니 skip → has_boundary=False 면 retain ref 자체가 없을 수도.
        if has_boundary:
            target_orig_retain = torch.stack(
                [orig_retain_hidden[l] for l in target_layers]).permute(1, 0, 2, 3)
            norm_target_orig_retain = target_orig_retain / (
                torch.norm(target_orig_retain, dim=-1, keepdim=True, dtype=torch.float))
            target_layer_retain_mask = torch.stack(
                [layers_retain_attn_mask[l] for l in target_layers]).permute(1, 0, 2, 3)
            norm_target_orig_retain = norm_target_orig_retain * target_layer_retain_mask

        norm_lora_xb = norm_lora_xb.permute(1, 0, 2, 3)
        layers_xb_attn_mask = layers_xb_attn_mask.permute(1, 0, 2, 3)
        norm_xb = norm_xb.permute(1, 0, 2, 3)

        xb_loss_acc = 0.0
        for i in range(batch_size):
            if is_boundary[i]:
                # separate: <h_lora_xb, mean(h_orig_retain)>
                mean_retain = (norm_target_orig_retain[i].sum(dim=-2)
                               / target_layer_retain_mask[i].sum(dim=-2).clamp(min=1))
                xb_len = norm_lora_xb[i].shape[1]
                mean_retain = mean_retain.unsqueeze(-2).repeat(1, xb_len, 1)
                inner = (norm_lora_xb[i] * mean_retain) * layers_xb_attn_mask[i]
                xb_loss_acc = xb_loss_acc + torch.relu(inner.sum(dim=-1)).sum()
            else:
                # erase: <h_lora_xb, h_orig_xb>
                inner = (norm_lora_xb[i] * norm_xb[i]) * layers_xb_attn_mask[i]
                xb_loss_acc = xb_loss_acc + torch.relu(inner.sum(dim=-1)).sum()

        x_boundary_loss = xb_loss_acc / layers_xb_attn_mask.sum().clamp(min=1)

    loss = retain_coeff * retain_loss + x_boundary_coeff * x_boundary_loss

    if log_now:
        print(f"  [step {current_step}] progress={progress:.4f}  "
              f"retain_c={retain_coeff:.4f} xb_c={x_boundary_coeff:.4f}  "
              f"retain={float(retain_loss):.4f} xb={float(x_boundary_loss):.4f}  "
              f"total={float(loss):.4f}", flush=True)

    return loss, dict(
        retain_loss=float(retain_loss),
        x_boundary_loss=float(x_boundary_loss),
        progress=progress,
        retain_coeff=retain_coeff,
        x_boundary_coeff=x_boundary_coeff,
    )


# ─────────────────────── model load + LoRA ───────────────────────

def load_model_with_lora(target_layers, *, drop_layers_after, lora_r=16, lora_alpha=16,
                         lora_dropout=0.05):
    """저자 train() 의 model 로드 + LoraConfig 그대로.

    drop_layers_after : 우리는 학습 시 max(target_layers) 위 layer 는 forward 안 함 (속도).
    inference 시엔 full layer 필요 → save 시 base full model + LoRA adapter 만 저장 →
    eval_xboundary.py 가 full base + adapter merge.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    full_num_layers = config.num_hidden_layers     # 3B = 28
    if drop_layers_after is not None and drop_layers_after < full_num_layers - 1:
        config.num_hidden_layers = drop_layers_after + 1
        print(f"[model] drop_layers_after={drop_layers_after} → "
              f"training 시 num_hidden_layers={config.num_hidden_layers} "
              f"(원본 {full_num_layers}). 저장 시 full layer 복원.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=os.environ.get("HF_TOKEN"),
        model_max_length=8192, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, device_map="auto",
        torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN"))

    layers_to_transform = list(range(max(target_layers) + 1))     # transform_layers=-1 의미
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", layers_to_transform=layers_to_transform,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()        # gradient_checkpointing 호환
    model.print_trainable_parameters()
    return model, tokenizer, full_num_layers


# ─────────────────────── save (LoRA adapter 만) ───────────────────────

def save_lora_adapter(model, output_dir: Path, *, target_layers,
                      transform_layers_max, full_num_layers,
                      lora_r, lora_alpha, lora_dropout,
                      training_data: dict | None = None):
    """LoRA adapter 만 저장 (base model 은 HF hub 에서 다시 받음).

    저자 코드는 drop_layers_after 한 후 base + adapter 둘 다 저장하지만, 우리는
    base 안 바꾸므로(public model) LoRA 만. eval 시 from_pretrained(base) +
    PeftModel.from_pretrained(adapter) 으로 복원.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    meta = dict(
        method="X-Boundary (arXiv:2502.09990) — ported to Llama-3.2-3B-Instruct",
        paper="Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks "
              "without Compromising Usability (EMNLP 2025 Findings)",
        repo="github.com/AI45Lab/X-Boundary",
        base_model=MODEL_ID,
        target_layers=target_layers,
        transform_layers="0..{}".format(transform_layers_max),
        base_full_num_layers=full_num_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        training_data=training_data or {},
        single_turn_only=True,
        note=("저자 multi-turn (SafeMT) 미사용 — over_refuse single-turn setup. "
              "ASR 절대 수치는 저자 reported 와 비교 불가."),
    )
    (output_dir / "xboundary_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] LoRA adapter → {output_dir} (+ xboundary_meta.json)")


# ─────────────────────── train loop ───────────────────────

def train_loop(model, tokenizer, dataset, *, target_layers, alpha=10, max_steps=180,
               batch_size=16, lr=1e-4, loss_coeff=300, use_warm_up=False,
               log_every=10, return_history=False):
    """저자 CustomTrainer 의 학습 루프 직접 구현 (HF Trainer 의존 회피 — Colab 격리).

    AdamW, constant lr — 저자 lr_scheduler_type='constant' 와 동일.
    """
    from torch.optim import AdamW

    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=lr, weight_decay=0.0)

    n = len(dataset)
    print(f"[train] dataset n={n}  max_steps={max_steps}  batch={batch_size}  lr={lr}  "
          f"alpha={alpha}  target_layers={target_layers}  loss_coeff={loss_coeff}")

    indices = list(range(n))
    random.shuffle(indices)
    cursor = 0

    history = []
    t0 = time.time()
    for step in range(max_steps):
        batch_idx = []
        for _ in range(batch_size):
            if cursor >= n:
                random.shuffle(indices)
                cursor = 0
            batch_idx.append(indices[cursor])
            cursor += 1
        batch = data_collator([dataset[i] for i in batch_idx])
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        log_now = (step % log_every == 0)
        loss, terms = compute_xboundary_loss(
            model, batch, target_layers=target_layers, alpha=alpha,
            batch_size=batch_size, current_step=step,
            loss_coeff=loss_coeff, use_warm_up=use_warm_up, log_now=log_now)

        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        if log_now or step == max_steps - 1:
            elapsed = time.time() - t0
            print(f"[train] step {step}/{max_steps}  loss={float(loss):.4f}  "
                  f"retain={terms['retain_loss']:.4f}  xb={terms['x_boundary_loss']:.4f}  "
                  f"({elapsed:.0f}s)", flush=True)
            history.append({"step": step, "loss": float(loss), **terms})

    print(f"[train] done — {time.time() - t0:.0f}s")
    if return_history:
        return history


# ─────────────────────── main ───────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="X-Boundary (arXiv:2502.09990) LoRA training, ported to Llama-3.2-3B")
    ap.add_argument("--xb-data-dir", default=None,
                    help="legacy 호환용. 현재 기본 학습은 paper data/train 을 사용하지 않음.")
    ap.add_argument("--data-source", choices=["manifest", "src"], default="manifest",
                    help="학습 데이터. manifest=cache/manifest.jsonl 의 우리 5-group response pair, "
                         "src=src.dataset loader fallback. paper data 는 사용하지 않음.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="--data-source manifest 일 때 사용할 우리 manifest.jsonl")
    ap.add_argument("--eval-indices", default=str(DEFAULT_EVAL_INDICES),
                    help="있으면 v-series eval split 을 학습에서 제외")
    ap.add_argument("--include-eval-split", action="store_true",
                    help="manifest 사용 시 eval_indices 제외를 끔. 보통 사용하지 않음.")
    ap.add_argument("--boundary-completion", default=DEFAULT_SAFE_COMPLETION,
                    help="manifest 의 over-refusal row 에 정답 completion 이 없을 때 retainQA 로 쓸 안전 응답")
    ap.add_argument("--output-dir", default="experiment/output/xboundary_3b_adapter",
                    help="LoRA adapter 저장 경로")
    ap.add_argument("--target-layers", default="8,16",
                    help="3B 비례환산 (저자 8B '10,20' → 28L 의 8,16). comma-sep.")
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="lorra_alpha (loss 가중)")
    ap.add_argument("--max-steps", type=int, default=180)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--loss-coeff", type=int, default=300)
    ap.add_argument("--boundary-data-size", type=int, default=500,
                    help="저자 boundary_data_size=500 (over-refusal pair 수)")
    ap.add_argument("--use-warm-up", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: max_steps=5, batch_size=2, dataset prefix 작게")
    ap.add_argument("--no-drop-layers", action="store_true",
                    help="drop_layers_after 비활성 (전체 layer forward — 느림)")
    ap.add_argument("--history-out", default=None,
                    help="loss history JSON 저장 경로(생략 시 output_dir/history.json)")
    args = ap.parse_args()

    # seed (저자 lorra_x_boundary.py 와 동일)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    target_layers = [int(l) for l in args.target_layers.split(",")]
    drop_after = None if args.no_drop_layers else max(target_layers)

    if args.smoke:
        args.max_steps = 5
        args.batch_size = 2
        args.boundary_data_size = 4
        print(f"[SMOKE] max_steps={args.max_steps} batch={args.batch_size} "
              f"boundary={args.boundary_data_size}")

    model, tokenizer, full_num_layers = load_model_with_lora(
        target_layers, drop_layers_after=drop_after,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout)

    dataset = XBoundaryDataset(
        tokenizer,
        boundary_data_size=args.boundary_data_size,
        use_refusal_retain=True,
        num_examples=2000 if args.smoke else 10000,
        model_name_or_path=MODEL_ID,
        data_source=args.data_source,
        manifest_path=args.manifest,
        eval_indices_path=args.eval_indices,
        exclude_eval=not args.include_eval_split,
        boundary_completion=args.boundary_completion)

    history = train_loop(
        model, tokenizer, dataset, target_layers=target_layers,
        alpha=args.alpha, max_steps=args.max_steps,
        batch_size=args.batch_size, lr=args.lr,
        loss_coeff=args.loss_coeff, use_warm_up=args.use_warm_up,
        return_history=True)

    out = Path(args.output_dir)
    save_lora_adapter(
        model, out, target_layers=target_layers,
        transform_layers_max=max(target_layers),
        full_num_layers=full_num_layers,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        training_data={
            "data_source": args.data_source,
            "manifest": args.manifest if args.data_source == "manifest" else None,
            "exclude_eval": not args.include_eval_split,
            "boundary_completion_fallback": args.boundary_completion,
        })

    hist_p = Path(args.history_out) if args.history_out else (out / "history.json")
    hist_p.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] adapter → {out}  history → {hist_p}")


if __name__ == "__main__":
    sys.exit(main())
