"""train_repbend.py — RepBend (Representation Bending, arXiv:2504.01550) 의
LoRA fine-tuning + activation-bending loss 를 Llama-3.2-3B-Instruct 위에 재구현.

⚠ 격리 런타임 (Colab/AWS) 전용 standalone. project `src.*` 를 *import 하지
않음* — peft / trl 설치가 main 노트북 런타임을 오염시킬 수 있어 분리.

저자 official code (github.com/AIM-Intelligence/RepBend) 의 핵심:
  L = α·L_safe − β·L_unsafe + γ·L_cos + ε·L_kl  (η·L_safeunsafe 는 default 0)
    L_safe   = ‖h_M'(safe)  − h_M(safe)‖₂   (KEEP — original 과 같게)
    L_unsafe = ‖h_M'(unsafe) − h_M(unsafe)‖₂ (PUSH — original 과 멀게, 부호 -β)
    L_cos    = mean(1 − cos(h_M'(unsafe))    (DIVERSIFY — unsafe 끼리 서로 멀게)
    L_kl     = KL(M'(retain) ‖ M(retain))    (RETAIN — safe prompt 일반능력 보존)

  M  = base LLM (LoRA disable_adapter() 로 호출)
  M' = M + LoRA  (학습 가능)

저자 default (train.sh, rep_bending 분기):
  α=0.5  β=0.5  γ=0.1  ε=0.3  η=0.0  lr=1e-5  bs=4  grad_accum=4  max_steps=450
  loss_mode='response_all' (response token 들만 hidden 비교)
  target_layer_start_idx=20  layers_window_size=11   → unsafe layers = [20..30]
  alpha_mode='all'    → safe layers = 전체 (0..N)
  lora_r=16  lora_alpha=16  lora_dropout=0.05  target_modules=all linear
  dataset = allenai/wildguardmix wildguardtrain (num_examples=10000)

우리 적용 시 변경 (README 의 'deviation' 표에 명시):
  - base model: Llama-3-8B (저자) → **Llama-3.2-3B-Instruct (우리)**.
    layer 수 32 → 28. target_layer_start_idx=20+layers_window_size=11 그대로
    쓰면 [20..30] 인데 3B 는 layer 0..27 만 존재 → start_idx=18, window=10 으로
    조정 (마지막 10 layer = [18..27]). 'last 1/3' 비율(저자 11/32≈34%) 유지
    (우리 10/28≈36%).
  - dataset: **우리 manifest/cache 데이터 우선**. `cache/manifest.jsonl` 에서
    safe-complied / harmful-refused / harmful-complied response pair 를 구성한다.
    manifest 가 없으면 paper data 가 아니라 `src.dataset` 기반 fallback 만 사용.
  - num_examples: 저자 10k → **우리 1024** (메모리 'comparison-baselines-llama31'
    의 common.SCALE 정책: harmful 512 + benign 512). RepBend 는 safe/unsafe pair
    가 같이 들어가므로 num_examples 가 한 쪽 (safe) 의 크기 = unsafe 와 같은 수.
    데이터 효율 비교 정합성 + Colab GPU/시간 예산.
  - max_steps: 저자 450 → **300** (1024 sample / bs 16 = 64 step/epoch, ~5 epoch).

사용:
  python experiment/repbend/train_repbend.py \
      --out-dir experiment/output/repbend_3b/ckpt \
      --num-examples 1024 \
      --max-steps 300 \
      [--smoke]   # 8 sample 만, 1 step, signed-gradient 검증
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.methods.our_training_data import (  # noqa: E402
    DEFAULT_EVAL_INDICES,
    DEFAULT_MANIFEST,
    manifest_repbend_pools,
    src_repbend_pools,
)

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
SEED = 42

# 우리 spec (3B = 28 layer). README 의 deviation 표 참조.
TARGET_LAYER_START_IDX = 18           # 저자 20 → 우리 18 (layer 0..27 안에서 마지막 10)
LAYERS_WINDOW_SIZE = 10               # 저자 11 → 우리 10
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# loss coefficient (저자 그대로)
ALPHA = 0.5   # safe representation 보존 가중 (M' safe = M safe)
BETA  = 0.5   # unsafe representation push 가중 (M' unsafe ≠ M unsafe, 부호 -)
GAMMA = 0.1   # unsafe diversity (서로 다르게)
EPS   = 0.3   # KL retain (safe prompt 일반능력)

# 학습 hyperparam (저자 + 우리 데이터 규모로 조정)
LR = 1e-5                              # 저자 train.sh learning_rate=1e-5 동일
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_STEPS_DEFAULT = 300                # 저자 450, 우리 1024 sample → 300
MAX_SEQ_LEN = 1024                     # 저자 4096 → 우리 1024 (3B GPU 예산)
MAX_GRAD_NORM = 1.0                    # 저자는 HF SFTTrainer default(=1.0) 사용. -β·L_unsafe 가
                                       # unbounded 라서 grad-clip 없으면 발산 — paper 도 clamp 안
                                       # 하고 이 grad-clip 으로만 막음. 우리도 동일하게.


# ─────────────────────── seeding (저자 train.py 동일) ───────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────── dataset ───────────────────────
# 저자 RepBendingDataset 의 'response_all' 분기와 동일:
#   - safe sample  : (prompt_harm=unharmful, response_refusal=compliance, response_harm=unharmful)
#                    + (prompt_harm=harmful,   response_refusal=refusal,    response_harm=unharmful)
#   - unsafe sample: (prompt_harm=harmful,    response_refusal=compliance)
# 즉 (a) ss = safe-prompt 안전응답, (b) us = unsafe-prompt 안전응답(refusal), (c) uu = unsafe-prompt 위험응답.
# response_all mode: hidden 비교는 response 토큰들만.

def _chat_format(tokenizer, instruction: str, response: str, model_name: str) -> str:
    """저자 dataset.py 의 'llama-3' 분기와 동일 포맷 (3.2 도 동일 special token)."""
    user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
    assistant_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return f"{user_tag}{instruction}{assistant_tag}{response}"


class RepBendDataset(Dataset):
    """우리 manifest/src dataset 에서 safe / unsafe / mixed sample 추출.

    저자 RepBendingDataset 의 response_all 모드 그대로 — sample 마다 다음 4 종 같이 반환:
      ids_safe_sample            : safe prompt + safe response          (ss)
      ids_unsafe_sample          : unsafe prompt + unsafe response      (uu)
      ids_unsafe_request_safe_response   : unsafe prompt + safe(=refusal) response  (us)
      ids_unsafe_request_unsafe_response : 위 ids_unsafe_sample 와 같음 (저자 코드도 한 쌍씩 사용)
      ids_retain                 : safe prompt + safe response의 prompt-only 부분 (KL 항)

    response_all loss_mode 에서는 response 부분 mask 만 hidden 비교에 씀.
    """

    def __init__(self, tokenizer, num_examples: int, max_length: int,
                 *, data_source: str = "manifest",
                 manifest_path: str | Path = DEFAULT_MANIFEST,
                 eval_indices_path: str | Path | None = DEFAULT_EVAL_INDICES,
                 exclude_eval: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if data_source == "manifest":
            pools = manifest_repbend_pools(
                manifest_path,
                num_examples=num_examples,
                seed=SEED,
                eval_indices_path=eval_indices_path,
                exclude_eval=exclude_eval,
            )
            source_desc = f"manifest:{manifest_path}"
        elif data_source == "src":
            pools = src_repbend_pools(num_examples=num_examples, seed=SEED)
            source_desc = "src.dataset fallback"
        else:
            raise ValueError(f"unknown data_source={data_source!r}; expected manifest|src")

        self.ss = pools["ss"]
        self.us = pools["us"]
        self.uu = pools["uu"]
        print(f"[RepBendDataset] source={source_desc}  "
              f"ss={len(self.ss)} us={len(self.us)} uu={len(self.uu)}")

    def __len__(self):
        return len(self.ss)

    def _encode(self, prompt: str, response: str):
        """prompt 와 response 따로 토큰화 — response_all mode 에서 response mask 만 쓰기 위해."""
        prompt_text = _chat_format(self.tokenizer, prompt, "", "llama-3")  # response 빈 칸
        full_text = _chat_format(self.tokenizer, prompt, response, "llama-3")

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False,
                                    truncation=True, max_length=self.max_length).input_ids
        full_ids = self.tokenizer(full_text, add_special_tokens=False,
                                  truncation=True, max_length=self.max_length).input_ids
        # full_ids 가 prompt_ids 로 시작하지 않으면 (rare encoding edge) prompt mask 길이만 보존
        prompt_len = len(prompt_ids) if full_ids[:len(prompt_ids)] == prompt_ids else min(len(prompt_ids), len(full_ids))
        return full_ids, prompt_len

    def __getitem__(self, idx):
        ss_p, ss_r = self.ss[idx]
        uu_p, uu_r = self.uu[idx]
        us_p, us_r = self.us[idx]

        ss_ids, ss_plen = self._encode(ss_p, ss_r)
        uu_ids, uu_plen = self._encode(uu_p, uu_r)
        us_ids, us_plen = self._encode(us_p, us_r)
        # retain = ss_ids 그대로 (저자도 safe prompt 위에서 KL)
        return {
            "ss_ids": ss_ids, "ss_plen": ss_plen,
            "uu_ids": uu_ids, "uu_plen": uu_plen,
            "us_ids": us_ids, "us_plen": us_plen,
        }


# ─────────────────────── collator (left pad) ───────────────────────

def collate(batch, pad_id: int):
    """left-pad (left-pad 면 last token = T-1 균일 → 저자 left padding 정합).

    반환: (ids, mask, response_mask) 짝 3개 — ss/uu/us.
    response_mask = 응답 토큰만 1, prompt/pad 는 0.

    ⚠ ss/uu/us 같은 `max_len` 으로 통일 — compute_repbend_loss 에서
    `torch.cat([ss_ids, us_ids], dim=0)` 가 sequence dim 일치 필요.
    """
    def pack(items, key_ids, key_plen, max_len):
        ids_list = [it[key_ids] for it in items]
        plen_list = [it[key_plen] for it in items]
        ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(items), max_len), dtype=torch.long)
        resp = torch.zeros((len(items), max_len), dtype=torch.long)
        for i, (x, plen) in enumerate(zip(ids_list, plen_list)):
            # left pad: 뒤쪽에 실토큰
            ids[i, max_len - len(x):] = torch.tensor(x, dtype=torch.long)
            mask[i, max_len - len(x):] = 1
            # response = prompt_len 이후 = max_len - len(x) + plen 이후
            resp_start = max_len - len(x) + plen
            resp[i, resp_start:] = 1
            # pad 영역(앞) 은 이미 0 — resp_mask 도 0
        return ids, mask, resp

    max_len = max(
        max(len(it["ss_ids"]) for it in batch),
        max(len(it["uu_ids"]) for it in batch),
        max(len(it["us_ids"]) for it in batch),
    )
    ss_ids, ss_mask, ss_resp = pack(batch, "ss_ids", "ss_plen", max_len)
    uu_ids, uu_mask, uu_resp = pack(batch, "uu_ids", "uu_plen", max_len)
    us_ids, us_mask, us_resp = pack(batch, "us_ids", "us_plen", max_len)
    return {
        "ss_ids": ss_ids, "ss_mask": ss_mask, "ss_resp": ss_resp,
        "uu_ids": uu_ids, "uu_mask": uu_mask, "uu_resp": uu_resp,
        "us_ids": us_ids, "us_mask": us_mask, "us_resp": us_resp,
    }


# ─────────────────────── model + lora ───────────────────────

def load_base_and_lora():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    # LoRA layers_to_transform = 0..max(target_layers) (저자 train.py 동일)
    target_layers = list(range(TARGET_LAYER_START_IDX,
                                TARGET_LAYER_START_IDX + LAYERS_WINDOW_SIZE))
    layers_to_transform = list(range(max(target_layers) + 1))
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        layers_to_transform=layers_to_transform,
        task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tok, target_layers


# ─────────────────────── 핵심: RepBend loss ───────────────────────
# 저자 trainer.py 의 _calc_loss 와 동일 식.

def _hidden_at(model, input_ids, attention_mask, target_layers: list[int]):
    """forward output_hidden_states=True → 지정 layer hidden stack.
    return: [L_target, B, T, H] tensor (fp32 안전성 위해 caller 가 dtype 취급).
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple of [B, T, H] for layers 0..N (N+1개)
    stacked = torch.stack([hs[l] for l in target_layers], dim=0)  # [L, B, T, H]
    return stacked, out.logits   # logits 는 KL retain 용


def _safe_norm_loss(lora_h, base_h, mask_3d):
    """저자 _calc_safe_loss / _calc_unsafe_loss 동일:
       sum_{layer,batch,token} ‖h_M' − h_M‖₂ / sum(mask).
    mask_3d: [L, B, T] (token 별 0/1, response 토큰만 1)."""
    diff = (lora_h - base_h).to(torch.float32)
    # ‖.‖₂ along H dim → [L, B, T]
    norms = torch.norm(diff, dim=-1, p=2)
    # token mask 적용
    return (norms * mask_3d).sum() / mask_3d.sum().clamp(min=1)


def _cosine_loss(lora_h_unsafe, request_last_idx_per_batch):
    """저자 _calc_cosine_loss: layer 별 unsafe 응답 시작 직전 token (= request 의 마지막
    token) hidden 들의 pairwise cosine similarity → (1 − cos) mean. unsafe 끼리 서로
    멀어지게.

    lora_h_unsafe: [L, B, T, H]
    request_last_idx_per_batch: [B] (left pad 라 max_len - 1 - response_len 같은 게 아니라
       저자는 mask_unsafe_sample_request.shape[-1] - 1 을 썼는데 — 그건 *request 의 전체
       길이* 의 -1, 즉 request 마지막 토큰. left-pad 환경에서 우리는 batch 별로 다름).
    """
    L, B, T, H = lora_h_unsafe.shape
    # batch 별 다른 idx — gather
    idx = request_last_idx_per_batch.to(lora_h_unsafe.device)  # [B]
    # expand to [L, B, 1, H] → gather along T
    idx_exp = idx.view(1, B, 1, 1).expand(L, B, 1, H)
    last = torch.gather(lora_h_unsafe, dim=2, index=idx_exp).squeeze(2)  # [L, B, H]
    last_fp = last.to(torch.float32)
    normed = last_fp / (last_fp.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    cos_loss = lora_h_unsafe.new_zeros(()).float()
    for l in range(L):
        sim = normed[l] @ normed[l].T  # [B, B]
        # 대각 제외
        eye = torch.eye(B, device=sim.device, dtype=torch.bool)
        cos_loss = cos_loss + (1 - sim[~eye]).mean()
    return cos_loss / L


def _kl_loss(lora_logits, base_logits, mask, temp: float = 2.0):
    """저자 _calc_kl_loss 동일 — retain prompt 위에서 KL(M' ‖ M) (temp scaled)."""
    m3 = mask.unsqueeze(-1).to(lora_logits.dtype)
    p = F.log_softmax((lora_logits * m3) / temp, dim=-1)
    q = F.softmax((base_logits * m3) / temp, dim=-1)
    return F.kl_div(p, q, reduction="batchmean") * (temp ** 2)


def compute_repbend_loss(model, batch, target_layers, *,
                          alpha=ALPHA, beta=BETA, gamma=GAMMA, eps=EPS):
    """저자 _compute_loss 의 response_all 모드 그대로 (single GPU 단순화 + dataset 합성).

    저자 코드는 (offline 모드 기준)
      safe_ids   = cat(ids_safe_sample,   ids_unsafe_request_safe_response)   # ss + us
      unsafe_ids = cat(ids_unsafe_sample, ids_unsafe_request_unsafe_response) # uu + uu(=second copy)
    우리는 ss + us / uu + us 로 (저자 분기에서 ids_unsafe_request_unsafe_response =
    ids_unsafe_sample 같은 식, 우리도 동일).
    """
    device = next(model.parameters()).device
    dev = lambda x: x.to(device)

    # safe = ss + us
    safe_ids = torch.cat([dev(batch["ss_ids"]),  dev(batch["us_ids"])],  dim=0)
    safe_mask = torch.cat([dev(batch["ss_mask"]), dev(batch["us_mask"])], dim=0)
    safe_resp = torch.cat([dev(batch["ss_resp"]), dev(batch["us_resp"])], dim=0)

    # unsafe = uu + uu (저자도 두 번)
    unsafe_ids = torch.cat([dev(batch["uu_ids"]), dev(batch["uu_ids"])], dim=0)
    unsafe_mask = torch.cat([dev(batch["uu_mask"]), dev(batch["uu_mask"])], dim=0)
    unsafe_resp = torch.cat([dev(batch["uu_resp"]), dev(batch["uu_resp"])], dim=0)

    # safe layers = 'all' (저자 alpha_mode='all'). target_layers_safe = 0..N+1
    # PEFT model 의 config 접근: model.config (forward 됨) 또는 model.module.config (DDP wrap),
    # 또는 model.base_model.model.config (최내부). 우선순위로 fallback.
    if hasattr(model, "module") and hasattr(model.module, "config"):
        n_layers = model.module.config.num_hidden_layers
    elif hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
    else:
        n_layers = model.base_model.model.config.num_hidden_layers  # PEFT 최내부
    target_layers_safe = list(range(n_layers + 1))     # output_hidden_states 는 N+1 개

    # === M (base = adapter disabled) — eval, no_grad ===
    model.eval()
    with model.disable_adapter():
        with torch.no_grad():
            base_safe_h, _ = _hidden_at(model, safe_ids, safe_mask, target_layers_safe)
            base_unsafe_h, _ = _hidden_at(model, unsafe_ids, unsafe_mask, target_layers)
            _, base_retain_logits = _hidden_at(model, dev(batch["ss_ids"]), dev(batch["ss_mask"]),
                                                 [n_layers])  # logits 만 필요

    # === M' (LoRA active) — train ===
    model.train()
    lora_safe_h, _ = _hidden_at(model, safe_ids, safe_mask, target_layers_safe)
    lora_unsafe_h, _ = _hidden_at(model, unsafe_ids, unsafe_mask, target_layers)
    _, lora_retain_logits = _hidden_at(model, dev(batch["ss_ids"]), dev(batch["ss_mask"]),
                                        [n_layers])

    # response mask 를 [L, B, T] 로 broadcast
    safe_mask_3d = safe_resp.unsqueeze(0).repeat(len(target_layers_safe), 1, 1).float()
    unsafe_mask_3d = unsafe_resp.unsqueeze(0).repeat(len(target_layers), 1, 1).float()

    # === L_safe ===
    L_safe = _safe_norm_loss(lora_safe_h, base_safe_h, safe_mask_3d)
    # === L_unsafe ===
    L_unsafe = _safe_norm_loss(lora_unsafe_h, base_unsafe_h, unsafe_mask_3d)
    # === L_cos === (저자 코드: 'mask_unsafe_sample_request.shape[-1] - 1' = request last token.
    #   우리는 left-pad 라 request 마지막 token = unsafe_resp == 1 의 가장 앞쪽 idx − 1.
    #   batch 별로 다름 → gather.)
    # request_last_idx = first idx where unsafe_resp == 1, minus 1
    # (resp_start = max_len - len(x) + plen → resp_start - 1 = request 마지막 token)
    first_resp = unsafe_resp.float().argmax(dim=1)  # [B] (resp_start)
    req_last = (first_resp - 1).clamp(min=0)        # [B]
    L_cos = _cosine_loss(lora_unsafe_h, req_last)
    # === L_kl ===
    L_kl = _kl_loss(lora_retain_logits, base_retain_logits, dev(batch["ss_mask"]))

    total = alpha * L_safe - beta * L_unsafe + gamma * L_cos + eps * L_kl
    return total, {
        "L_safe": L_safe.item(),
        "L_unsafe": L_unsafe.item(),
        "L_cos": L_cos.item(),
        "L_kl": L_kl.item(),
        "total": total.item(),
    }


# ─────────────────────── ⚠ 검증: 1-step sign check ───────────────────────

def sign_check(model, tokenizer, target_layers, batch, lr: float = LR,
                max_grad_norm: float = MAX_GRAD_NORM):
    """학습 1 step 전후 — harmful prompt 에 'I' (refusal 시작 token 후보 X)
    vs 'Sure' 같은 *compliance* token logit 차이가 줄어드는지 (= refusal 쪽으로
    움직이는지) 측정. AlphaSteer 의 sign 버그 같은 case 방지.

    실제로 'refusal token logit ↑' 정확 검증은 어렵지만 — RepBend 는 hidden 을
    바꾸는 거라, *unsafe loss term* 부호가 맞으면 (학습 후 L_unsafe 가 *증가*
    해야 함, 부호 -β·L_unsafe 라 minimization 이 L_unsafe 키움) 으로 sign 검증.
    """
    model.train()
    # before: no_grad 로 graph 누적 회피 (값만 필요)
    with torch.no_grad():
        _, comp = compute_repbend_loss(model, batch, target_layers)
    L_unsafe_before = comp["L_unsafe"]
    L_safe_before = comp["L_safe"]
    # 1-step (학습 loop 와 동일 grad-clip)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr)
    total, _ = compute_repbend_loss(model, batch, target_layers)
    total.backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_grad_norm)
    optim.step()
    optim.zero_grad()
    # after: no_grad
    with torch.no_grad():
        _, comp_after = compute_repbend_loss(model, batch, target_layers)
    L_unsafe_after = comp_after["L_unsafe"]
    L_safe_after = comp_after["L_safe"]
    delta_unsafe = L_unsafe_after - L_unsafe_before
    delta_safe = L_safe_after - L_safe_before
    print(f"\n[SIGN CHECK]")
    print(f"  L_unsafe: {L_unsafe_before:.4f} → {L_unsafe_after:.4f} (Δ={delta_unsafe:+.4f})  "
          f"[기대 Δ > 0 — unsafe push 효과]")
    print(f"  L_safe  : {L_safe_before:.4f} → {L_safe_after:.4f} (Δ={delta_safe:+.4f})  "
          f"[기대 Δ ≈ 0 또는 ↓ — safe 보존]")
    # ⚠ hard fail: unsafe Δ 가 음수면 sign 잘못된 것 (loss 가 unsafe 를 *줄임*)
    # 단 1 step 으로는 노이즈 가능 — Δ < -0.1 일 때만 fail (강한 음수만)
    if delta_unsafe < -0.1:
        raise RuntimeError(
            f"sign check FAIL: L_unsafe Δ={delta_unsafe:+.4f} < -0.1 — "
            f"loss 부호 잘못된 것. -β·L_unsafe minimization 은 L_unsafe ↑ 방향이어야 함. "
            f"check β sign, target_layers, response_mask alignment.")
    print(f"  [PASS] sign convention OK\n")


# ─────────────────────── main ───────────────────────

def main():
    ap = argparse.ArgumentParser(description="RepBend Llama-3.2-3B LoRA training")
    ap.add_argument("--out-dir", default="experiment/output/repbend_3b/ckpt",
                    help="LoRA adapter 저장 경로")
    ap.add_argument("--num-examples", type=int, default=1024)
    ap.add_argument("--data-source", choices=["manifest", "src"], default="manifest",
                    help="학습 데이터. manifest=cache/manifest.jsonl 의 우리 5-group response pair, "
                         "src=src.dataset loader fallback. paper data 는 사용하지 않음.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="--data-source manifest 일 때 사용할 우리 manifest.jsonl")
    ap.add_argument("--eval-indices", default=str(DEFAULT_EVAL_INDICES),
                    help="있으면 v-series eval split 을 학습에서 제외")
    ap.add_argument("--include-eval-split", action="store_true",
                    help="manifest 사용 시 eval_indices 제외를 끔. 보통 사용하지 않음.")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    ap.add_argument("--lr", type=float, default=LR,
                    help=f"AdamW learning rate (저자 paper / train.sh: 1e-5, 우리 default={LR})")
    ap.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM,
                    help=f"gradient clipping (저자 HF default 1.0; -β·L_unsafe unbounded → 필수). "
                         f"default={MAX_GRAD_NORM}. 0 또는 음수면 clipping disable.")
    ap.add_argument("--divergence-stop", type=float, default=10.0,
                    help="L_unsafe EMA(α=0.1)가 처음 5 step 평균의 N× 넘으면 early-stop. "
                         "0/음수면 disable. default=10.")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke mode: 8 sample, 1 step sign check 후 exit (학습 안 함)")
    args = ap.parse_args()

    set_seed(SEED)
    t0 = time.time()

    print(f"[setup] MODEL_ID={MODEL_ID}")
    print(f"[setup] target_layers=[{TARGET_LAYER_START_IDX}..{TARGET_LAYER_START_IDX + LAYERS_WINDOW_SIZE - 1}] "
          f"(layers_window={LAYERS_WINDOW_SIZE})")
    print(f"[setup] LoRA r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
    print(f"[setup] loss: α={ALPHA} β={BETA} γ={GAMMA} ε={EPS}")
    print(f"[setup] num_examples={args.num_examples}  max_steps={args.max_steps}  "
          f"bs={args.batch_size} grad_accum={args.grad_accum}  lr={args.lr}  "
          f"max_grad_norm={args.max_grad_norm if args.max_grad_norm > 0 else 'disabled'}  "
          f"divergence_stop={args.divergence_stop if args.divergence_stop > 0 else 'disabled'}")
    print(f"[setup] data_source={args.data_source}  manifest={args.manifest}  "
          f"exclude_eval={not args.include_eval_split}")

    model, tok, target_layers = load_base_and_lora()
    dataset = RepBendDataset(
        tok,
        num_examples=args.num_examples if not args.smoke else 8,
        max_length=MAX_SEQ_LEN,
        data_source=args.data_source,
        manifest_path=args.manifest,
        eval_indices_path=args.eval_indices,
        exclude_eval=not args.include_eval_split,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=lambda b: collate(b, tok.pad_token_id))

    if args.smoke:
        # ⚠ smoke: 첫 batch 1 step 으로 sign check
        batch = next(iter(loader))
        sign_check(model, tok, target_layers, batch, lr=args.lr, max_grad_norm=args.max_grad_norm)
        print(f"[done] smoke OK ({time.time() - t0:.0f}s) — full 학습은 --smoke 빼고 다시 실행.")
        return

    # === full training loop ===
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    model.train()
    step = 0
    accum = 0
    losses_log = []
    grad_norm_log = []                     # clip 전 grad norm 추적용 (clip 발동 빈도 진단)
    L_unsafe_ema = None                    # divergence 감지용 EMA(α=0.1)
    L_unsafe_baseline = None               # 처음 5 step 의 단순 평균 (baseline)
    L_unsafe_initial_buf = []
    stopped_early = False
    while step < args.max_steps and not stopped_early:
        for batch in loader:
            total, comp = compute_repbend_loss(model, batch, target_layers)
            (total / args.grad_accum).backward()
            accum += 1
            if accum >= args.grad_accum:
                # grad-clip (paper 의 안정화 메커니즘 — HF SFTTrainer default=1.0)
                if args.max_grad_norm > 0:
                    gnorm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.max_grad_norm)
                    grad_norm_log.append(float(gnorm))
                optim.step()
                optim.zero_grad()
                accum = 0
                step += 1
                losses_log.append({"step": step, **comp,
                                    "grad_norm": grad_norm_log[-1] if grad_norm_log else None})

                # divergence guard (EMA vs baseline)
                lu = comp["L_unsafe"]
                if step <= 5:
                    L_unsafe_initial_buf.append(lu)
                    if step == 5:
                        L_unsafe_baseline = sum(L_unsafe_initial_buf) / 5.0
                L_unsafe_ema = lu if L_unsafe_ema is None else 0.9 * L_unsafe_ema + 0.1 * lu
                if (args.divergence_stop > 0 and L_unsafe_baseline is not None
                        and L_unsafe_ema > args.divergence_stop * L_unsafe_baseline):
                    print(f"\n  ⚠ DIVERGENCE STOP at step {step}: "
                          f"L_unsafe_ema={L_unsafe_ema:.4f} > {args.divergence_stop}× "
                          f"baseline({L_unsafe_baseline:.4f}). "
                          f"lr 낮추거나 max_grad_norm 낮추거나 max_steps 줄여. "
                          f"(저자 paper recipe: lr=1e-5, max_steps=450, grad-clip=1.0)")
                    stopped_early = True
                    break

                if step % 10 == 0:
                    gn = grad_norm_log[-1] if grad_norm_log else float("nan")
                    print(f"  [step {step}/{args.max_steps}] "
                          f"L_safe={comp['L_safe']:.4f} L_unsafe={comp['L_unsafe']:.4f} "
                          f"L_cos={comp['L_cos']:.4f} L_kl={comp['L_kl']:.4f} "
                          f"total={comp['total']:.4f}  gnorm={gn:.3f}")
                if step >= args.max_steps:
                    break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "train_log.json").write_text(json.dumps({
        "losses": losses_log,
        "config": {
            "MODEL_ID": MODEL_ID,
            "TARGET_LAYER_START_IDX": TARGET_LAYER_START_IDX,
            "LAYERS_WINDOW_SIZE": LAYERS_WINDOW_SIZE,
            "ALPHA": ALPHA, "BETA": BETA, "GAMMA": GAMMA, "EPS": EPS,
            "LR": args.lr, "BATCH_SIZE": args.batch_size, "GRAD_ACCUM": args.grad_accum,
            "MAX_STEPS": args.max_steps, "NUM_EXAMPLES": args.num_examples,
            "MAX_GRAD_NORM": args.max_grad_norm, "DIVERGENCE_STOP": args.divergence_stop,
            "STOPPED_EARLY": stopped_early, "STEPS_ACTUAL": step,
            "LORA_R": LORA_R, "LORA_ALPHA": LORA_ALPHA, "LORA_DROPOUT": LORA_DROPOUT,
            "DATA_SOURCE": args.data_source,
            "MANIFEST": args.manifest if args.data_source == "manifest" else None,
            "EXCLUDE_EVAL": not args.include_eval_split,
        },
    }, indent=2))
    print(f"[done] {time.time() - t0:.0f}s — saved {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
