# RepBend @ Llama-3.2-3B — baseline 재구현 (LoRA + activation-bending loss)

> **무엇**: RepBend (Yousefpour et al., *Representation Bending for Large Language
> Model Safety*, **arXiv:2504.01550**, ACL 2025) 의 핵심 학습 loss + LoRA recipe
> 를 우리 base (Llama-3.2-3B-Instruct) 위에 재구현. our v-series adapter 와
> 직접 비교하기 위한 baseline.


핵심 정리:
- **paper**: arXiv:2504.01550 (Apr 2025, ACL 2025)
- **official code**: https://github.com/AIM-Intelligence/RepBend
- **base in paper**: Mistral-7B-Instruct-v0.2 / Meta-Llama-3-8B-Instruct
- **base in us**: meta-llama/Llama-3.2-3B-Instruct (project 표준)


---

## 1. 산출 파일

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

## 2. Colab 실행 절차 (격리 런타임)

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


## 출처 / 인용

- Yousefpour, A., et al. (2025). *Representation Bending for Large Language
  Model Safety*. arXiv:2504.01550. ACL 2025.
  https://arxiv.org/abs/2504.01550
- Official code: https://github.com/AIM-Intelligence/RepBend
- HF data (저자 기준): https://huggingface.co/datasets/allenai/wildguardmix
