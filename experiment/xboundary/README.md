# X-Boundary — Llama-3.2-3B 포팅 baseline

**무엇**: X-Boundary (Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn
Jailbreaks without Compromising Usability, arXiv:2502.09990, EMNLP 2025 Findings, Lu et al.)
의 학습 algorithm 을 **저자 official code (github.com/AI45Lab/X-Boundary) 의
`src/lorra_x_boundary.py` 의 loss 와 dataset 그대로** 우리 셋업
(Llama-3.2-3B-Instruct, over_refuse single-turn eval) 에 포팅. **이건 baseline 재현**이지
저자 결과 transfer 가 아님.


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

### C. 학습 (180 step, ~30-60min A100 추정)
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

### D. eval — full (5 dataset × 100 sample)
```bash
!python experiment/xboundary/eval_xboundary.py \
    --adapter-dir experiment/output/xboundary_3b_adapter \
    --xb-data-dir /content/XBoundary/data/train \
    --out experiment/output/resp_xboundary_3b.json \
    --per-source-n 100
# 산출 = WildGuard 라벨링까지 마친 metric (ASR/ORR/CR) + per_sample dump.
```

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
