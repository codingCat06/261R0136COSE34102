"""Benchmark loader. HF datasets → 통일 Sample.

핵심:
  - XSTest, OR-Bench  : over-refusal 측정 (위험해 보이지만 답해야 함)
  - HarmBench, AdvBench : 진짜 harmful (refuse 가 정답)
  - Alpaca             : benign reference (control)

각 loader 는 list[Sample] 반환. Sample.expected ∈ {"refuse", "answer"}, .category ∈
{"harmful", "safe_sensitive", "benign"} 의 통제 vocab.

통합 인터페이스:
  load_eval_set(name, n=None) → 위 loader 중 하나 호출.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


def _load_hf(repo: str, *args, **kw):
    """HF datasets lazy import."""
    from datasets import load_dataset
    return load_dataset(repo, *args, **kw)


@dataclass
class Sample:
    id: str
    prompt: str
    expected: str           # "refuse" | "answer"
    category: str           # "harmful" | "safe_sensitive" | "benign"
    benchmark: str
    meta: dict[str, Any] = field(default_factory=dict)


def _take(rows, n: int | None, seed: int) -> list:
    """deterministic prefix sampling — shuffle once with seed, take [:n].

    같은 seed 면 n 작아도 prefix 일관 (cross-experiment test set 어긋나지 않음).
    """
    if n is None or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in idx[:n]]


# ─────────────────────── over-refusal (safe but looks risky) ────────────────────────

def load_xstest(n: int | None = None, *, seed: int = 42, safe: bool = True) -> list[Sample]:
    """XSTest. safe=True: 위험해 보이지만 답해야 함 (over-refusal 측정, 250 sample).
            safe=False: 실제 위험한 prompt (대조군, 200 sample)."""
    # column = prompt, focus, type, note, label. label ∈ {'safe', 'unsafe'} 이 split key.
    # (이전엔 type.startswith("safe") 로 잘못 봤음 — type 에 safe_targets/safe_contexts 만 매치돼서 50개로 잘렸고 unsafe 는 0.)
    ds = _load_hf("walledai/XSTest", split="test")
    target = "safe" if safe else "unsafe"
    rows = [r for r in ds if r.get("label") == target]
    rows = _take(rows, n, seed)
    return [
        Sample(
            id=f"xstest_{i:04d}",
            prompt=r["prompt"],
            expected="answer" if safe else "refuse",
            category="safe_sensitive" if safe else "harmful",
            benchmark="xstest",
            # focus = 표면적 유해 단어 (예: "kill"). 마지막 시각화에서 token 위치 매핑 후 v_harm projection.
            # type = 18가지 prompt 유형 (homonyms / figurative_language / contrast_xxx ...) — sub-category 분석용.
            meta={"type": r.get("type"), "note": r.get("note"), "focus": r.get("focus")},
        )
        for i, r in enumerate(rows)
    ]


def load_or_bench(n: int | None = None, *, seed: int = 42, hard_only: bool = True) -> list[Sample]:
    """OR-Bench. hard_only=True (default): or-bench-hard-1k. False: or-bench-80k.
    모두 over-refusal 후보 (sensitive 해 보이지만 답해야 함)."""
    cfg = "or-bench-hard-1k" if hard_only else "or-bench-80k"
    ds = _load_hf("bench-llm/or-bench", cfg, split="train")
    rows = _take(list(ds), n, seed)
    return [
        Sample(
            id=f"orbench_{i:04d}",
            prompt=r["prompt"],
            expected="answer",
            category="safe_sensitive",
            benchmark="or_bench",
            meta={"category": r.get("category")},
        )
        for i, r in enumerate(rows)
    ]


ORBENCH_HARD_TEST_N = 500   # OR-Bench Hard-1K 中 *평가 전용* 예약 수.
#   build_test_set(ORR test)·load_or_bench_orr_train(학습 holdout 경계)이 *이 한
#   상수* 를 공유 → test/train Hard split 경계 single-source(silent leak 방지).
#   바꾸려면 여기 한 곳만 + test/manifest 둘 다 재생성.


def orbench_hard_split(*, seed: int = 42,
                       n_test: int = ORBENCH_HARD_TEST_N) -> tuple[list[Sample], list[Sample]]:
    """OR-Bench Hard-1K 을 seed 로 1회 shuffle → (test[:n_test], train_holdout[n_test:]).

    test(build_test_set ORR)와 train(manifest or_corr)이 *동일 split* 공유 →
    id-disjoint 코드 보장. shuffle 알고리즘 = 옛 build_test_set._sample 과 동일
    (random.Random(seed)) → 기존 ORR test id 불변(하위호환).
    """
    full = load_or_bench(None, hard_only=True)        # 전량 ~1319 (원본순서·stable id)
    rng = random.Random(seed)
    idx = list(range(len(full)))
    rng.shuffle(idx)
    return [full[i] for i in idx[:n_test]], [full[i] for i in idx[n_test:]]


def load_or_bench_orr_train(n: int | None = None, *, seed: int = 42,
                            n_test_holdout: int = ORBENCH_HARD_TEST_N) -> list[Sample]:
    """over-refuse(or_corr) 학습용 OR-Bench pool — ORR test 와 prompt·id 모두 disjoint.

    구성 = (a) Hard-1K train-holdout([n_test_holdout:]) +
           (b) or-bench-80k 中 Hard-1K *전체* prompt 제외분(= test 포함 제거).
    or-bench-toxic 은 진짜 유해(prompt_harm=True)→ over-refuse 아님 → 미포함.
    n 주면 합집합 seed-shuffle 후 prefix n. id 는 test(orbench_XXXX)와 안 겹치게
    'orbench_orrtrain_*' 새 prefix(leak-check id 비교 명확).
    """
    test, holdout = orbench_hard_split(seed=seed, n_test=n_test_holdout)
    # Hard-1K 전량 prompt = test ∪ holdout (split 재사용 — Hard-1K 추가 load 회피).
    _hard_p = {(s.prompt or "").strip() for s in test} | {
        (s.prompt or "").strip() for s in holdout}
    _k80_all = load_or_bench(None, hard_only=False)
    k80 = [s for s in _k80_all if (s.prompt or "").strip() not in _hard_p]
    _removed = len(_k80_all) - len(k80)                # 80k 中 Hard-1K 와 일치해 제외된 수
    if _removed < 0.8 * len(_hard_p):                  # exact-match 누락(공백/유니코드 변형) 의심
        raise ValueError(
            f"load_or_bench_orr_train: 80k 에서 Hard-1K prompt 일치 {_removed} "
            f"< 0.8×{len(_hard_p)} — exact-match 누락 의심(정규화 변형/Hard-1K⊄80k). "
            f"방치 시 변형 prompt 가 train 에 남아 ORR test 와 누수 → 점검 요.")
    pool = _take(list(holdout) + k80, n, seed)
    return [
        Sample(id=f"orbench_orrtrain_{i:05d}", prompt=s.prompt,
               expected=s.expected, category=s.category,
               benchmark="or_bench", meta=s.meta)
        for i, s in enumerate(pool)
    ]


def load_phtest(n: int | None = None, *, seed: int = 42, harmless_only: bool = True) -> list[Sample]:
    """PHTest (furonghuang-lab/PHTest, MIT). AutoDAN 으로 자동 생성 — lexical hijack 외 paraphrase 다양.
    harmless_only=True (default): 확실히 무해한 prompt 만 (학습 신호 깨끗).
                                  False: harmless + controversial 둘 다.
    Llama-3.1-8B harmless 거절률 ~27% (paper) → 3.2-3B 추정 30-40%.
    """
    ds = _load_hf("furonghuang-lab/PHTest", split="train")
    # column 명 schema 미확정 (paper 와 HF README 표기 차이) — 후보 다 검사
    def _label(r):
        for k in ("Harmlessness", "Harmfulness", "label", "category"):
            if k in r:
                return str(r[k]).lower()
        return ""
    def _prompt(r):
        # PHTest HF 의 실제 column 명 = "Request" (paper 표기와 다름). 다른 fork 가 prompt/instruction 일 수도 있어 fallback 다.
        for k in ("Request", "prompt", "instruction", "Prompt", "text"):
            if k in r:
                return r[k]
        return None
    rows = []
    for r in ds:
        p = _prompt(r)
        if not p:
            continue
        lbl = _label(r)
        if harmless_only and "harmless" not in lbl:
            continue
        rows.append((p, lbl))
    rows = _take(rows, n, seed)
    return [
        Sample(
            id=f"phtest_{i:05d}",
            prompt=p,
            expected="answer",
            category="safe_sensitive",
            benchmark="phtest",
            meta={"label": lbl},
        )
        for i, (p, lbl) in enumerate(rows)
    ]


# ─────────────────────── harmful ────────────────────────

def load_harmbench(n: int | None = None, *, seed: int = 42, subset: str = "standard") -> list[Sample]:
    """HarmBench harmful prompt. subset = 'standard' (default) | 'contextual' | 'copyright'.
    walledai/HarmBench column = [prompt, category]. category = SemanticCategory
    (chemical_biological / cybercrime_intrusion / ...) — original 의 6 column 중 walledai 가
    prompt + SemanticCategory 만 보존."""
    ds = _load_hf("walledai/HarmBench", subset, split="train")
    rows = _take(list(ds), n, seed)
    return [
        Sample(
            id=f"harmbench_{i:04d}",
            prompt=r["prompt"],
            expected="refuse",
            category="harmful",
            benchmark="harmbench",
            meta={"semantic_category": r.get("category"), "subset": subset},
        )
        for i, r in enumerate(rows)
    ]


def load_wildjailbreak(n: int | None = None, *, seed: int = 42,
                       split: str = "adversarial_harmful",
                       config: str = "train") -> list[Sample]:
    """WildJailbreak (allenai/wildjailbreak). jailbreak prompt — category="harmful_jailbreak" (별도).
    split: 'adversarial_harmful' (default — jailbreak + harmful intent) / 'adversarial_benign' /
           'vanilla_harmful' / 'vanilla_benign'  (← HF split 아니라 data_type 컬럼 필터).
    config: 'train' (default, HF train config ~261k — 학습 manifest 용) /
            'eval' (저자 native held-out 평가셋 2,210 = adv_harmful 2,000 + adv_benign 210).
            test set 은 train↔test leak 방지로 'eval' 사용 — 'train' 을 학습·평가가
            같이 쓰면 same-seed _take 로 100% 누수(실측 1000/1000).
    TSV 라 delimiter='\\t' + keep_default_na=False 안 주면 Arrow 가 텍스트를 double 로 잘못 추론해 폭발.
    """
    # config 별 HF split 명이 다를 수 있어(eval config split 명 미확정) split= 고정 X.
    # DatasetDict 면 모든 split 이어붙임 — train config(단일 split)는 기존과 동일
    # 순서·동일 행집합 → _take(seed) 선택 불변(기존 wildjb_ id / manifest cache 호환).
    ds = _load_hf("allenai/wildjailbreak", config,
                  delimiter="\t", keep_default_na=False)
    if hasattr(ds, "keys"):                          # DatasetDict (split= 미지정 시)
        src = []
        for _k in sorted(ds.keys()):
            src.extend(ds[_k])
    else:
        src = list(ds)
    rows = [r for r in src if r.get("data_type") == split]
    if src and not rows:                             # boundary: 스키마/값 mismatch silent 0행 방지
        _seen = sorted({str(r.get("data_type")) for r in src[:200]})
        raise ValueError(
            f"load_wildjailbreak(config={config!r}, split={split!r}): "
            f"data_type=={split!r} 매칭 0행 (src {len(src)}행). 해당 config 의 실제 "
            f"data_type 값 예: {_seen}. eval config 스키마/라벨이 train 과 다를 수 "
            f"있음 — 컬럼·값 확인 필요(방치 시 빈 평가셋이 저장됨).")
    rows = _take(rows, n, seed)
    is_adv = "adversarial" in split
    is_harmful = "harmful" in split
    return [
        Sample(
            id=f"wildjb_{i:05d}",
            prompt=r.get("adversarial") if is_adv else r.get("vanilla"),
            expected="refuse" if is_harmful else "answer",
            # jailbreak 은 표상이 정상 harmful 과 다를 수 있어 plot 에서 별도 색으로 보고 싶음 — 별도 category.
            category="harmful_jailbreak" if is_harmful else "safe_sensitive",
            benchmark="wildjailbreak",
            # adversarial split 의 같은 row 에는 vanilla(같은 의도의 plain 버전)도 있음 → meta 에 보관.
            # 짝(adversarial↔vanilla) 분석은 이 meta["vanilla"] 로. split 을 따로 두 번 호출하면 _take shuffle 순서가 달라 짝이 깨짐.
            meta={"data_type": split, "tactics": r.get("tactics"),
                  "vanilla": (r.get("vanilla") or None) if is_adv else None},
        )
        for i, r in enumerate(rows)
        if (r.get("adversarial") if is_adv else r.get("vanilla"))
    ]


def load_wildjailbreak_benign(n: int | None = None, *, seed: int = 42,
                              include_adversarial: bool = True, include_vanilla: bool = True) -> list[Sample]:
    """WildJailbreak benign side — adversarial_benign + vanilla_benign (allenai/wildjailbreak, ODC-BY gated).
    adversarial_benign: jailbreak-form coded language 인데 의도 무해 (78,706).
    vanilla_benign    : 단순 benign (50,050).
    Llama-2-7B Tulu2Mix baseline RTA WJ(benign) 54.4% (paper) → 3.2-3B 도 30%+ 추정.
    dataset 한 번만 load 한 뒤 두 split 모두 수집 (기존 load_wildjailbreak 와 달리 효율).
    """
    ds = _load_hf("allenai/wildjailbreak", "train", split="train",
                   delimiter="\t", keep_default_na=False)
    wanted = set()
    if include_adversarial: wanted.add("adversarial_benign")
    if include_vanilla:     wanted.add("vanilla_benign")
    rows = []
    for r in ds:
        dt = r.get("data_type")
        if dt not in wanted:
            continue
        text = r.get("adversarial") if dt == "adversarial_benign" else r.get("vanilla")
        if not text:
            continue
        rows.append({"text": text, "data_type": dt, "tactics": r.get("tactics"),
                     "vanilla": r.get("vanilla") if dt == "adversarial_benign" else None})
    rows = _take(rows, n, seed)
    return [
        Sample(
            id=f"wjb_benign_{i:05d}",
            prompt=r["text"],
            expected="answer",
            category="safe_sensitive",
            benchmark="wildjailbreak_benign",
            meta={"data_type": r["data_type"], "tactics": r["tactics"], "vanilla": r["vanilla"]},
        )
        for i, r in enumerate(rows)
    ]


def load_advbench(n: int | None = None, *, seed: int = 42) -> list[Sample]:
    """AdvBench harmful_behaviors."""
    ds = _load_hf("walledai/AdvBench", split="train")
    rows = _take(list(ds), n, seed)
    return [
        Sample(
            id=f"advbench_{i:04d}",
            prompt=r["prompt"],
            expected="refuse",
            category="harmful",
            benchmark="advbench",
            meta={"target": r.get("target")},
        )
        for i, r in enumerate(rows)
    ]


def load_jailbreakbench(n: int | None = None, *, seed: int = 42) -> list[Sample]:
    """JailbreakBench JBB-Behaviors (Chao et al. 2024). 100 harmful behaviors.
    HuggingFace: JailbreakBench/JBB-Behaviors. column: Behavior / Goal / Target / Category / Source.
    paper 의 표준 jailbreak benchmark — 100 harmful 'behaviors' + 100 benign.
    여기선 harmful 만."""
    try:
        ds = _load_hf("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    except Exception:
        # fallback — full split 에서 harmful 만 filter
        ds = _load_hf("JailbreakBench/JBB-Behaviors", "behaviors")
        ds = ds["harmful"] if "harmful" in ds else list(ds["train"])
    rows = _take(list(ds), n, seed)
    out = []
    for i, r in enumerate(rows):
        prompt = r.get("Goal") or r.get("Behavior") or r.get("prompt") or r.get("goal")
        if not prompt: continue
        out.append(Sample(
            id=f"jbb_{i:04d}",
            prompt=prompt,
            expected="refuse",
            category="harmful_jailbreak",
            benchmark="jailbreakbench",
            meta={"behavior": r.get("Behavior"), "target": r.get("Target"),
                  "category_jbb": r.get("Category")},
        ))
    return out


def load_maliciousinstruct(n: int | None = None, *, seed: int = 42) -> list[Sample]:
    """MaliciousInstruct (Huang et al. 2023). 100 강력 harmful prompts.
    HuggingFace mirror: walledai/MaliciousInstruct."""
    ds = _load_hf("walledai/MaliciousInstruct", split="train")
    rows = _take(list(ds), n, seed)
    out = []
    for i, r in enumerate(rows):
        prompt = r.get("prompt") or r.get("instruction") or r.get("Instruction") or r.get("text")
        if not prompt: continue
        out.append(Sample(
            id=f"malicious_{i:04d}",
            prompt=prompt,
            expected="refuse",
            category="harmful",
            benchmark="maliciousinstruct",
            meta={},
        ))
    return out


def load_strongreject(n: int | None = None, *, seed: int = 42) -> list[Sample]:
    """StrongREJECT (Souly et al. 2024). 313 강력 harmful prompts (다양 카테고리).
    HuggingFace: walledai/StrongREJECT."""
    try:
        ds = _load_hf("walledai/StrongREJECT", split="train")
    except Exception:
        ds = _load_hf("csHugging/StrongREJECT", split="train")
    rows = _take(list(ds), n, seed)
    out = []
    for i, r in enumerate(rows):
        prompt = r.get("forbidden_prompt") or r.get("prompt") or r.get("instruction")
        if not prompt: continue
        out.append(Sample(
            id=f"strongreject_{i:04d}",
            prompt=prompt,
            expected="refuse",
            category="harmful",
            benchmark="strongreject",
            meta={"category": r.get("category"), "source": r.get("source")},
        ))
    return out


# ─────────────────────── benign reference ────────────────────────

def load_alpaca(n: int | None = None, *, seed: int = 42) -> list[Sample]:
    """Alpaca instruction (input 비어있는 instruction-only). general benign reference."""
    ds = _load_hf("tatsu-lab/alpaca", split="train")
    rows = [r for r in ds if not (r.get("input") or "").strip()]
    rows = _take(rows, n, seed)
    return [
        Sample(
            id=f"alpaca_{i:04d}",
            prompt=r["instruction"],
            expected="answer",
            category="benign",
            benchmark="alpaca",
            meta={"output": r.get("output")},
        )
        for i, r in enumerate(rows)
    ]


# ─────────────────────── unified ────────────────────────

LOADERS = {
    "xstest": lambda n, **kw: load_xstest(n, **kw),
    "xstest_safe": lambda n, **kw: load_xstest(n, safe=True, **kw),
    "xstest_unsafe": lambda n, **kw: load_xstest(n, safe=False, **kw),
    "or_bench": lambda n, **kw: load_or_bench(n, **kw),
    "or_bench_orr_train": lambda n, **kw: load_or_bench_orr_train(n, **kw),
    "phtest": lambda n, **kw: load_phtest(n, **kw),
    "harmbench": lambda n, **kw: load_harmbench(n, **kw),
    "advbench": lambda n, **kw: load_advbench(n, **kw),
    "wildjailbreak": lambda n, **kw: load_wildjailbreak(n, **kw),
    "wildjailbreak_benign": lambda n, **kw: load_wildjailbreak_benign(n, **kw),
    "alpaca": lambda n, **kw: load_alpaca(n, **kw),
    "jailbreakbench": lambda n, **kw: load_jailbreakbench(n, **kw),
    "maliciousinstruct": lambda n, **kw: load_maliciousinstruct(n, **kw),
    "strongreject": lambda n, **kw: load_strongreject(n, **kw),
}


def load_eval_set(name: str, n: int | None = None, **kw) -> list[Sample]:
    """name 으로 benchmark 로드. 알 수 없는 이름이면 KeyError."""
    return LOADERS[name](n=n, **kw)
