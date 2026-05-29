# X-Boundary — Llama-3.2-3B 포팅 baseline

**무엇**: X-Boundary (Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn
Jailbreaks without Compromising Usability, arXiv:2502.09990, EMNLP 2025 Findings, Lu et al.)
의 학습 algorithm 을 **저자 official code (github.com/AI45Lab/X-Boundary) 의
`src/lorra_x_boundary.py` 의 loss 와 dataset 그대로** 우리 셋업
(Llama-3.2-3B-Instruct, over_refuse single-turn eval) 에 포팅. **이건 baseline 재현**이지
저자 결과 transfer 가 아님.

**아닌 것**: 저자 8B 모델 직접 사용 X / multi-turn 평가 X / GPT-4o judge X.

**저자 코드와 우리 코드의 mapping** — 저자 official `lorra_x_boundary.py:compute_loss`
의 line 별로 우리 `train_xboundary.compute_xboundary_loss` 에 1:1 포팅:
- `retain_coeff = alpha * progress` (저자 동일)
- `x_boundary_coeff = alpha * (1 - progress)` (저자 동일)
- erase loss (`is_boundary=False`): `ReLU(<ĥ_lora(xb), ĥ_orig(xb)>)` (저자 동일)
- separate loss (`is_boundary=True`): `ReLU(<ĥ_lora(xb), mean(ĥ_orig(retain))>)` (저자 동일)
- retain loss: `||h_lora(retain)·mask - h_orig(retain)·mask||_2` Frobenius (저자 동일)

---

## 알고리즘 요약 (이걸 학습)

X-Boundary 의 핵심 = LoRA adapter 학습, loss = **retain + erase + separate** 3-항.

(1) **reference model** = base 가 LoRA disable_adapter() 한 forward (학습 가능 weight 0).
모든 reference hidden state 는 `model.disable_adapter()` 블록 안에서 추출 (no_grad).

(2) **progress schedule** (저자 `loss_coeff=300`, `max_steps=180` → 학습 끝까지 progress<1).
```
progress = step / loss_coeff
retain_coeff      = α · progress
x_boundary_coeff  = α · (1 - progress)
```
초반엔 boundary loss (erase + separate) 가 dominant, 후반엔 retain 이 dominant.

(3) **erase loss** (harmful pair, `is_boundary=False`):
```
ĥ_lora(xb) := h_lora(xb) / ||h_lora(xb)||
ĥ_orig(xb) := h_orig(xb) / ||h_orig(xb)||
L_erase = ReLU( <ĥ_lora(xb), ĥ_orig(xb)> )   (per-layer, per-token, summed/normalized)
```
→ harmful 표상을 reference (LoRA off) 와 직교화 (cos < 0 가는 건 ReLU 가 0 으로 자름).

(4) **separate loss** (over-refusal pair, `is_boundary=True`):
```
ĥ_orig(retain) := mean_over_tokens( h_orig(retain) / ||h_orig(retain)|| )
L_sep = ReLU( <ĥ_lora(xb), ĥ_orig(retain)> )
```
→ over-refusal 의 거절 응답 표상을 *retain (benign) 평균 표상* 과 직교화.
("over-refusal 의 거절 출력이 benign 입력 분포와 묶이지 않도록.")

(5) **retain loss**:
```
L_retain = || (h_lora(retain) - h_orig(retain)) · mask ||_2   (Frobenius, mean)
```
→ benign 응답 표상 보존 (LoRA on/off 거리 0 으로 가까이).

(6) **합산**:
```
L = retain_coeff · L_retain + x_boundary_coeff · L_xb
L_xb = (Σ erase + Σ separate) / total_mask
```

---

## 우리 셋업과의 차이 (정직 보고)

| 항목                  | 저자 (paper)                                              | 우리 (port)                                                    |
|---|---|---|
| base model            | Llama-3-8B-Instruct (32 layer, d=4096), Qwen2-7B-Instruct | **Llama-3.2-3B-Instruct (28 layer, d=3072)**                  |
| target_layers         | "10,20"                                                   | **"8,16"** (비례 환산 — 10/32 ≈ 8/28, 20/32 ≈ 16/28)            |
| transform_layers      | "-1" (0..max(target))                                     | "-1" (0..16) — 동일                                            |
| LoRA r / α / dropout  | 16 / 16 / 0.05                                            | 16 / 16 / 0.05 — 동일                                          |
| target_modules        | q/k/v/o/gate/up/down                                      | 동일                                                          |
| optim / lr / wd       | AdamW / 1e-4 constant / 0                                 | 동일                                                          |
| batch / max_steps     | 16 / 180                                                  | 동일                                                          |
| lorra_alpha (α)       | 10                                                        | 동일                                                          |
| loss_coeff            | 300                                                       | 동일                                                          |
| boundary_data_size    | 500                                                       | 동일                                                          |
| dataset (erase)       | circuit_breakers_train_2400.json (저자 own)               | **동일** (저자 fetch)                                          |
| dataset (boundary)    | ORbench_retain_set.json (저자 own)                        | **동일** (저자 fetch)                                          |
| dataset (retain)      | ultrachat_200k + refusal_retain                           | **동일**                                                       |
| **multi-turn**        | **포함** (SafeMT_train_600)                                | **❌ 미포함** — over_refuse single-turn setup                  |
| evaluation suite      | HarmBench / OKTest / PHtest / ORbench / xstest_v2 / 다회 attack | **xstest_safe / harmbench / advbench / or_bench / alpaca eval_idx** |
| judge                 | LLM-as-judge (GPT-4 등)                                    | **WildGuard** (project memory `no-gpt4o-judge`)                |

**핵심 deviation 두 개:**
1. **single-turn 만** — 저자 paper 의 핵심 contribution 은 "multi-turn jailbreak 도 막는다" 인데
   우리는 over_refuse single-turn setup → ASR 절대 수치는 저자 reported 수치와
   **직접 비교 불가**. 우리 metric 은 *우리 eval split 내에서의* 상대 비교만.
2. **judge 다름** — 저자 LLM-as-judge / 우리 WildGuard. ASR/ORR 의 절대값 다름.
   우리 표 안에선 commandv/AlphaSteer 와 같은 WildGuard 라 cross-baseline 비교 OK.

---

## 실행 순서 (Colab)

### A. 데이터·코드 fetch (1 회만)
```bash
!bash experiment/xboundary/fetch_xboundary.sh /content/XBoundary
# → /content/XBoundary/data/train/{circuit_breakers_train_2400.json,
#                                   circuit_breakers_val.json,
#                                   ORbench_retain_set.json}
```

### B. 격리 런타임 (학습) — 별도 Colab 런타임 권장
```bash
!pip -q install -r experiment/xboundary/requirements_xboundary.txt
```

### C. SMOKE 먼저 (학습 *전* — 통과 후 full 학습)
```bash
# HF_TOKEN 필요 (gated Llama-3.2-3B)
!python experiment/xboundary/smoke_test.py \
    --xb-data-dir /content/XBoundary/data/train
# 통과 = 4 검증 모두 PASS:
#   (1) data file sanity   PASS
#   (2) split disjoint     overlap=N (0 이상이면 eval_xboundary 가 자동 제거)
#   (3) zero-init bit      PASS (LoRA on/off 동일 — 학습 가설 OK)
#   (4) sign Δ logP        > 0 (1-step 후 refusal token logit ↑)
```

### D. 학습 (180 step, ~30-60min A100 추정)
```bash
!python experiment/xboundary/train_xboundary.py \
    --xb-data-dir /content/XBoundary/data/train \
    --output-dir experiment/output/xboundary_3b_adapter
# 산출:
#   experiment/output/xboundary_3b_adapter/adapter_model.safetensors
#   experiment/output/xboundary_3b_adapter/adapter_config.json
#   experiment/output/xboundary_3b_adapter/xboundary_meta.json
#   experiment/output/xboundary_3b_adapter/history.json (loss 곡선)
```

### E. eval — SMOKE (학습 후, deviation hard-fail)
```bash
!python experiment/xboundary/eval_xboundary.py \
    --adapter-dir experiment/output/xboundary_3b_adapter \
    --xb-data-dir /content/XBoundary/data/train \
    --out experiment/output/smoke_xboundary_3b.json \
    --smoke 8
# trained vs base deviation > 0 확인 (=학습 됐다).
```

### F. eval — full (5 dataset × 100 sample)
```bash
!python experiment/xboundary/eval_xboundary.py \
    --adapter-dir experiment/output/xboundary_3b_adapter \
    --xb-data-dir /content/XBoundary/data/train \
    --out experiment/output/resp_xboundary_3b.json \
    --per-source-n 100
# 산출 = WildGuard 라벨링까지 마친 metric (ASR/ORR/CR) + per_sample dump.
```

---

## 검증 protocol (hard-fail assertions)

| # | 항목 | 위치 | 통과 기준 |
|---|---|---|---|
| 1 | smoke (학습 전, zero-init bit) | `smoke_test.py:check_zero_init_bitidentical` | LoRA on/off logit max-diff < 1e-5 — **PASS 확인 (`peft` 의 lora_B init 이 0 인 한 무조건 통과; 실제 GPU 실행으로 한 번 확정 필요)** |
| 2 | sign verification (1-step) | `smoke_test.py:check_sign_after_one_step` | Δ mean logP(refusal token) ≥ 0 (또는 max ≥ 0) — AlphaSteer −0.3 케이스 자동 검출 |
| 3 | greedy deviation (학습 후) | `eval_xboundary.py:--smoke` | trained vs base 응답 차이 > 0 — 학습 안 됨 자동 검출 |
| 4 | train/eval split disjoint | `smoke_test.py:check_split_disjoint` + `eval_xboundary.py:_xb_train_overlap_set` | X-Boundary 학습 prompt 와 over_refuse eval prompt overlap — *검출 + 자동 제거*. ORbench_retain_set 이 OR-Bench full 의 부분집합이라 nontrivial overlap 발생 가능; eval 시 강제 제거 |
| 5 | resp JSON schema | `eval_xboundary.label_and_score` | `[{id, prompt, response, refused, source, category, expected}]` 고정 |
| 6 | WildGuard PROMPT 무수정 | `src.classifier.classify_refuse` 직접 호출 | `src/classifier.py` 의 PROMPT/`_parse` 절대 변경 X |
| 7 | step ≠ same-model gotcha | LoRA = 같은 base + adapter | `eval_xboundary.load_eval_model` 가 `from_pretrained(MODEL_ID)` + `PeftModel.from_pretrained(adapter)` — train 시 `drop_layers_after` 로 잘랐어도 eval 은 full base + adapter merge (full 28 layer 복원). 같은 base, weight transfer 0 |
| 8 | single-turn 정직 보고 | 이 README + `xboundary_meta.json` + `resp_*.json:config.method_meta.deviation` | "저자는 multi-turn 평가 / 우리는 single-turn — ASR 절대 비교 불가" 명시 |
| 9 | summary cell (notebook 마지막) | `xboundary.ipynb` 의 마지막 cell | 모든 핵심 출력 재 print (loss curve / smoke verdict / metric) |

검증 #1·#2·#3 은 GPU 필요 → 사용자 Colab 실행으로 확정.
**환경 부재로 #1·#2·#3 의 실시간 PASS 확정은 사용자 첫 Colab run 에서 수행** (smoke_test.py 가 자동 hard-fail). README 의 검증 결과 칸은 사용자 첫 run 후 채움.

---

## 외부 모르는 점 (추정)

1. **`is_boundary=True` 의 separate loss 의 마스크 위치**: 저자 official 코드는 boundary 케이스에서
   `label_mask = (request_attn, zeros(response_attn))` — 즉 **request 토큰만** loss 에 적용.
   paper 설명 ("over-refusal 의 *거절 응답* 표상이 retain 평균과 직교") 과 implementation 의
   토큰 위치가 어긋남이 있을 수 있음. 우리는 **implementation 충실**을 택했음
   (저자 line 그대로 포팅). paper 텍스트에 맞춰 response 토큰만 mask 하려면
   `train_xboundary.py:XBoundaryDataset.__getitem__` 의 `label_mask_xb` 분기를
   reverse (`torch.cat([zeros(req), resp_attn])`). reviewer 가 paper-vs-code 충돌을
   지적하면 그쪽으로 switch.
2. **`loss_coeff=300` 의 의미**: 저자 default. `max_steps=180` < 300 → progress 끝나도 < 1.
   즉 학습 종료 시점에도 `retain_coeff < α`, `x_boundary_coeff > 0`. (저자 의도 유지.)
3. **`drop_layers_after`**: 우리 `train_xboundary.py` 는 학습 시 max(target_layers)=16 위
   layer 모두 잘라 forward 안 함 (저자 동일 — 속도). 단 LoRA adapter 의
   `layers_to_transform` 도 [0..16] 라 그 위 layer 에는 LoRA 없음. **eval 시 base full
   28 layer 복원 + adapter merge** (저자 utils.save_model_and_tokenizer 가 base 도 잘라서
   저장하는 것과 다른 우리 선택: base 는 HF hub 에서 다시 받음).
4. **저자 SafeMT_train_600 미포함**: multi-turn 학습 데이터로 single-turn baseline 학습을
   안 하는 게 정직. 저자가 single-turn 학습+multi-turn 평가도 했는지는 paper 본문 확인 필요.

---

## 산출물 위치

- 학습 adapter: `experiment/output/xboundary_3b_adapter/`
  - `adapter_model.safetensors` — LoRA weights
  - `adapter_config.json` — peft config (target_modules, layers_to_transform)
  - `xboundary_meta.json` — paper/repo/deviation 기록
  - `history.json` — step 별 loss curve (retain / x_boundary / total)
- 평가 결과: `experiment/output/resp_xboundary_3b.json`
  - `per_sample` — [{id, prompt, response, refused, source, category, expected}]
  - `config.method_meta.deviation` — single-turn / 3B / WildGuard 명시
  - `metrics` — ASR / ORR / CR / by_source

---

## 라이선스 / 인용

저자 repo (github.com/AI45Lab/X-Boundary) 라이선스 그대로 (fetch 시 `/content/XBoundary/LICENSE`).

```bibtex
@inproceedings{lu2025xboundary,
  title     = {X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from
               Multi-Turn Jailbreaks without Compromising Usability},
  booktitle = {Findings of EMNLP 2025},
  year      = {2025},
  url       = {https://arxiv.org/abs/2502.09990}
}
```
