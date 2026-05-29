# RepBend @ Llama-3.2-3B — baseline 재구현 (LoRA + activation-bending loss)

> **무엇**: RepBend (Yousefpour et al., *Representation Bending for Large Language
> Model Safety*, **arXiv:2504.01550**, ACL 2025) 의 핵심 학습 loss + LoRA recipe
> 를 우리 base (Llama-3.2-3B-Instruct) 위에 재구현. our v-series adapter 와
> 직접 비교하기 위한 baseline.
>
> **아닌 것**: 저자 repo (github.com/AIM-Intelligence/RepBend) 의 코드 *실행*
> 도 아니고, 저자 공개 ckpt 사용도 아님 — 저자 ckpt 는 Llama-3-**8B** / Mistral-7B
> 만 공개돼 있어 우리 base (3B) 가 없음. 따라서 우리는 저자 train.sh 의 hyperparam
> 그대로, 우리 base 위에서 **새로 학습**.

핵심 정리:
- **paper**: arXiv:2504.01550 (Apr 2025, ACL 2025)
- **official code**: https://github.com/AIM-Intelligence/RepBend
- **base in paper**: Mistral-7B-Instruct-v0.2 / Meta-Llama-3-8B-Instruct
- **base in us**: meta-llama/Llama-3.2-3B-Instruct (project 표준)

---

## 1. Algorithm 요약 (Section 3.2, Algorithm 1)

```
L = α·L_safe − β·L_unsafe + γ·L_cos + ε·L_kl  (+ η·L_safeunsafe; default η=0)
```

| 항 | 정의 | 의미 |
|---|---|---|
| **L_safe** | ‖h_{M'}(safe) − h_M(safe)‖₂ | safe prompt 의 hidden 을 *원래대로* 유지 (학습으로 안 바꿈) |
| **L_unsafe** | ‖h_{M'}(unsafe) − h_M(unsafe)‖₂ | unsafe prompt 의 hidden 을 *멀리* 밀어냄 (부호 −β 라 minimization 이 ↑) |
| **L_cos** | mean(1 − cos(h_{M'}(unsafe))) pairwise | unsafe 끼리 서로 다양화 (collapse 방지) |
| **L_kl** | KL(M'(safe) ‖ M(safe)) | safe prompt 위 일반능력 보존 (logit 보존) |

M = base LLM (`disable_adapter()` 로 호출), M' = M + LoRA adapter (학습 가능).
**LoRA 만 학습** — base weight freeze.

저자 train.py 에서 LoRA `layers_to_transform = [0..max(target_layers)]`,
hidden 비교는 `target_layers = [start_idx .. start_idx+window-1]` 사이 layer 에서만.

---

## 2. 우리 적용 — 저자와의 차이 (의도적 deviation 표)

| 항목 | 저자 (Llama-3-8B) | 우리 (Llama-3.2-3B) | 이유 |
|---|---|---|---|
| base model | Meta-Llama-3-8B-Instruct (32 layer) | meta-llama/Llama-3.2-3B-Instruct (**28 layer**) | project 표준 |
| target layers | start=20, window=11 → **20..30** | start=**18**, window=**10** → **18..27** | 3B 는 layer 0..27 만 존재; 'last ~1/3' 비율 (저자 11/32≈34% / 우리 10/28≈36%) 유지 |
| LoRA r / α / dropout | 16 / 16 / 0.05 | **동일** | — |
| LoRA target_modules | q/k/v/o + gate/up/down_proj (all linear) | **동일** | — |
| loss α / β / γ / ε / η | 0.5 / 0.5 / 0.1 / 0.3 / 0.0 | **동일** | — |
| optimizer / lr | AdamW / 1e-5 / constant | **동일** | — |
| batch / grad_accum | 4 / 4 (effective 16) | **동일** | — |
| max_seq_length | 4096 | **1024** | 3B + Colab GPU 예산 |
| max_steps | 450 | **300** | 우리 데이터 1024 sample 기준 (저자 10k → 우리 1024) |
| num_examples (각 ss/us/uu) | 10000 | **1024** | memory `comparison-baselines-llama31` 의 common.SCALE 정책 (harmful 512 + benign 512) — v-series 와 데이터 예산 정합 |
| train dataset | allenai/wildguardmix wildguardtrain | **동일** | 저자 그대로 — train data 는 변경 X (label noise 보존) |
| loss_mode | `response_all` | **동일** | — |
| alpha_mode | `all` (safe 는 전체 layer) | **동일** | — |
| eval judge | HarmBench classifier / WildGuard | **WildGuard 만** | project memory `no-gpt4o-judge` (GPT-4o 금지) |
| eval data | HarmBench / WildGuardTest / XSTest | **5 dataset eval_idx**: `load_xstest(safe) + load_harmbench + load_advbench + load_or_bench(hard_only) + load_alpaca` 의 `[train_n=64 : train_n+per_source_n=100]` slice | X-Boundary 와 **동일 split** — v-series head-to-head 호환. category={harmful, safe_sensitive, benign} 으로 ASR/ORR/CR 분리 |
| 학습 framework | trl SFTTrainer + accelerate + deepspeed | **plain torch loop, single GPU** | 단순화 (Colab 1xL4 OK). loss 식은 byte-identical. |

---

## 3. 산출 파일

```
experiment/repbend/
├── README.md                    # 이 파일
├── requirements_repbend.txt     # peft, datasets 등
├── smoke_test.py                # 학습 전 sanity (LoRA no-op 검증)
├── train_repbend.py             # LoRA + RepBend loss training
├── eval_repbend.py              # ckpt → eval split 응답 dump (resp_*.json)
└── repbend.ipynb                # end-to-end notebook (smoke + train + eval + WG)
```

---

## 4. Colab 실행 절차 (격리 런타임)

### A. 환경 설치

```bash
# 새 Colab 런타임 (런타임 > 연결 해제 및 새로). peft 설치가 메인 노트북
# 런타임을 오염시킬 수 있어 분리.
cd /content/drive/MyDrive/.../over_refuse
pip -q install -r experiment/repbend/requirements_repbend.txt
export HF_TOKEN='...'   # gated meta-llama/Llama-3.2-3B-Instruct
```

### B. 검증 (학습 전)

```bash
# 1) smoke: LoRA zero-init 시 base bit-identical 확인 + base 응답 sanity
python experiment/repbend/smoke_test.py

# 2) 1-step sign check (학습 부호 검증 — L_unsafe 가 Δ > 0 이어야)
python experiment/repbend/train_repbend.py --smoke
```

### C. 학습 (~30-60 min @ L4)

```bash
python experiment/repbend/train_repbend.py \
    --out-dir experiment/output/repbend_3b/ckpt \
    --num-examples 1024 \
    --max-steps 300
```

학습 끝나면 `experiment/output/repbend_3b/ckpt/` 에 PEFT adapter 저장.
`train_log.json` 에 step 별 loss 분리 기록.

### D. eval (5 dataset eval_idx 응답 생성, ~10-15 min @ H100)

```bash
# 1) smoke (앞 8개 + base 비교 — greedy deviation 검증)
python experiment/repbend/eval_repbend.py \
    --ckpt experiment/output/repbend_3b/ckpt \
    --smoke 8

# 2) full — 5 dataset × per_source_n=100 = 500 sample
python experiment/repbend/eval_repbend.py \
    --ckpt experiment/output/repbend_3b/ckpt \
    --out experiment/output/resp_repbend_3b.json \
    --per-source-n 100 \
    --train-n 64 \
    --compare-base   # base 대비 deviation 측정 같이
```

⚠ `--per-source-n` 과 `--train-n` 은 **X-Boundary 와 동일**해야 head-to-head 비교 가능. v-series 와도 동일 split (`notebooks/01` 의 train_n=64 표준).

### E. WildGuard judge (in-notebook, GPU 권장)

`repbend.ipynb` 마지막 셀 — `src.classifier.classify_refuse` 직접 호출.
WildGuard PROMPT/_parse 절대 수정 X (memory).

---

## 5. 검증 protocol (실제 통과 여부)

| # | 검증 항목 | 어디 (cell / 파일) | 통과 여부 |
|---|---|---|---|
| 1 | smoke (no-op): LoRA zero-init = base bit-identical | `smoke_test.py` step [3/4] | **실행 필요** (Colab GPU) — code 검증 PASS (PEFT W_b init=0 보장) |
| 2 | sign verification: 1-step 후 L_unsafe Δ > 0 (push 부호 OK) | `train_repbend.py --smoke` → `sign_check()` | **실행 필요** (Colab GPU). hard-fail = Δ < −0.1 |
| 3 | greedy deviation: 학습 후 base 와 응답 차이 > 0 | `eval_repbend.py --compare-base` → `deviation` field | **실행 필요** (학습 ckpt 있어야). smoke 모드에선 hard-fail |
| 4 | dataset split disjoint: train(wildguardmix) ∩ eval(wildjb+orbench) = ∅ | `smoke_test.py` step [4/4] + 본 README 표 | PASS (source 자체 disjoint, id prefix 도 다름) |
| 5 | resp JSON schema: `{id, prompt, response, refused, source, category, expected}` (X-Boundary 와 동일) | `eval_repbend.py` → `_validate_schema()` | PASS (코드 단위, dump 직전 assertion) |
| 6 | WildGuard PROMPT 절대 수정 X | `src/classifier.py` 직접 호출만 | PASS (코드 미수정) |
| 7 | step ≠ same-model: 우리는 동일 base 위 LoRA 만 학습/적용, 다른 step 모델 없음 | n/a | PASS |
| 8 | summary cell (notebook 마지막): smoke / sign / ASR / ORR / deviation 재출력 | `repbend.ipynb` 마지막 셀 | PASS (notebook 안에 작성) |

검증 1-3 은 Colab GPU 환경에서 사용자가 실행. code 정적 검증은 통과:
- PEFT LoRA init 표준: `lora_A` kaiming + `lora_B` zero → 초기 `B @ A = 0` 보장.
- `_safe_norm_loss` 가 `‖h_M' − h_M‖` 형태로 양수, 부호 `−β` 라 minimization → L_unsafe ↑ (저자 식과 동일).
- `_validate_schema` 가 dump 직전 assertion (raise 가능).

---

## 6. 알려진 제약 / 외부 모르는 점

1. **저자 코드 미실행** — peft + trl + accelerate + deepspeed 의 자세한 trainer
   wrapper 까지 그대로 재현하진 않았음. 우리는 핵심 loss 만 동일 (`compute_repbend_loss`
   의 `_safe_norm_loss` / `_cosine_loss` / `_kl_loss`). 차이:
   - **저자**: `SFTTrainer` packing=True + DeepSpeed zero1.
   - **우리**: plain DataLoader + AdamW. 단일 GPU. packing 없음.

2. **chat template 변환**: 저자 dataset.py 는 raw `<|start_header_id|>user...`
   토큰을 직접 쓰는데, 우리는 같은 형식. Llama-3.2 도 동일 special token (검증됨).

3. **η·L_safeunsafe**: 저자 default 0. 우리도 0 → 구현은 했지만 호출 X.

4. **wildguardmix label noise**: 저자가 `prompt_harm_label==harmful &
   response_refusal_label==compliance` 로 uu 집단을 정의 — 우리는 그대로.
   raw label noise (WildGuardMix 가 sometimes 잘못 labeled) 가 있을 수 있음 —
   양쪽 baseline 동일하게 영향 받으므로 비교는 fair.

5. **3B layer 수 차이**: 저자 paper Section 5.2 ablation 에서 target_layer 위치
   변경에 ASR 가 민감 (4-6% pt 차이). 우리는 *layer 비율 (last 1/3)* 만 맞추고
   start_idx 를 18 로 — 만약 ablation 가 필요하면 start_idx 를 (16, 17, 18, 19) 로
   sweep 권장.

6. **max_steps 300** 은 저자 450 (8B + 10k sample) 의 *step ratio* 보다 우리가
   더 많이 train (1024 sample × 5 epoch ≈ 320 step). epoch 너무 많으면 over-bend
   → benign 응답 망가짐 (저자 paper 의 trade-off 곡선). step 줄이고 (e.g. 150)
   ablation 도 권장.

7. **decoding deviation**: 저자 eval 은 `do_sample=True` (논문 기준 일부 metric).
   우리는 `do_sample=False` (greedy) — over_refuse harness 의 다른 baseline 과
   동일 디코딩 (memory: greedy 결정론은 harness 불변식).

---

## 7. 출처 / 인용

- Yousefpour, A., et al. (2025). *Representation Bending for Large Language
  Model Safety*. arXiv:2504.01550. ACL 2025.
  https://arxiv.org/abs/2504.01550
- Official code: https://github.com/AIM-Intelligence/RepBend
- HF data (저자 기준): https://huggingface.co/datasets/allenai/wildguardmix
