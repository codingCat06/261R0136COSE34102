"""logging / io / seed / secrets / cache helper — 최소 구성.

secrets:
  - 프로젝트 루트 .env → 환경변수 (이미 있으면 그거 우선)
  - Colab fallback (google.colab.userdata)
  - HF_TOKEN 있으면 huggingface_hub 자동 login

cache helper:
  - load_or_compute(path, fn, ...) : 파일 존재시 load, 없으면 fn() 실행 + 저장 + return
  - 사용자가 if 분기로 직접 control 하고 싶으면 raw save_pt / load_pt / save_json / load_json 직접 호출.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


# ─────────────────────── logging ────────────────────────

_NOISY = (
    "httpx", "httpcore", "urllib3",
    "huggingface_hub", "datasets", "transformers",
    "filelock", "accelerate", "torch",
)


def _silence_libraries() -> None:
    """HF 계열 progress bar / verbose log 끔 (노트북 깔끔)."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
    try:
        import datasets as _ds
        _ds.disable_progress_bar()
    except Exception:
        pass
    try:
        import transformers as _tf
        _tf.logging.set_verbosity_error()
    except Exception:
        pass


def setup_logging(level: str = "INFO") -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for n in _NOISY:
        logging.getLogger(n).setLevel(logging.ERROR)
    _silence_libraries()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ─────────────────────── secrets / .env ────────────────────────

_SECRETS_LOADED = False
_SECRET_KEYS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "OPENAI_API_KEY", "GITHUB_TOKEN")


def _find_dotenv() -> Path | None:
    """현재 파일 위치 → 상위 디렉토리 순으로 .env 검색. pyproject.toml 만나면 stop."""
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents]:
        if (d / ".env").exists():
            return d / ".env"
        if (d / "pyproject.toml").exists():
            break
    return None


def _parse_dotenv(path: Path) -> dict[str, str]:
    """간단한 .env parser (외부 의존성 없음)."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if k:
            out[k] = v
    return out


def load_secrets(verbose: bool = False) -> dict[str, bool]:
    """.env / Colab userdata 에서 secrets 로드. idempotent.

    return: {key: 환경변수에 있는지 bool}
    """
    global _SECRETS_LOADED
    if _SECRETS_LOADED:
        return {k: bool(os.getenv(k)) for k in _SECRET_KEYS}

    # 1) .env 파일 → setdefault (이미 환경변수 있으면 그게 우선)
    env_path = _find_dotenv()
    if env_path is not None:
        try:
            for k, v in _parse_dotenv(env_path).items():
                os.environ.setdefault(k, v)
            if verbose:
                logging.info(f".env loaded from {env_path}")
        except Exception as e:
            if verbose:
                logging.warning(f".env parse 실패 ({env_path}): {e}")

    # 2) Colab fallback — userdata
    try:
        from google.colab import userdata  # type: ignore
        for k in _SECRET_KEYS:
            if not os.getenv(k):
                try:
                    val = userdata.get(k)
                    if val:
                        os.environ[k] = val
                except Exception:
                    pass
    except ImportError:
        pass

    # 3) HF_TOKEN / HUGGINGFACE_TOKEN 정규화 + huggingface_hub login 자동 호출
    hf = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf:
        os.environ.setdefault("HF_TOKEN", hf)
        os.environ.setdefault("HUGGINGFACE_TOKEN", hf)
        try:
            from huggingface_hub import login
            login(hf, add_to_git_credential=False)
        except Exception:
            pass  # huggingface_hub 미설치 / login 실패는 silent (model load 시점에 또 시도됨)

    _SECRETS_LOADED = True
    return {k: bool(os.getenv(k)) for k in _SECRET_KEYS}


# 모듈 import 시 자동 로드 (side-effect 이지만 환경변수 주입 / HF login 은 안전)
load_secrets()


# ─────────────────────── seed ────────────────────────

def set_global_seed(seed: int) -> None:
    """python / numpy / torch 한 번에 seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ─────────────────────── io ────────────────────────

def save_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return p


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_pt(path: str | Path, obj: Any) -> Path:
    """torch.save wrapper. parent dir 자동 생성."""
    import torch
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, p)
    return p


def load_pt(path: str | Path) -> Any:
    """torch.load wrapper. weights_only=False (tensor + dict 같이 저장 가능), cpu 로 load."""
    import torch
    return torch.load(Path(path), weights_only=False, map_location="cpu")


# ─────────────────────── cache helper ────────────────────────

# ─────────────────────── GPU memory ────────────────────────

def free_cuda() -> None:
    """gc.collect + torch.cuda.empty_cache. Colab 등 ephemeral 환경에서 cache 누적 회피.

    매 iteration 마다 호출하면 overhead 큼 — *큰 단계 끝* (forward / generate / classify 함수 끝)
    또는 노트북 셀 사이에 명시적 호출 권장.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ─────────────────────── cache helper ────────────────────────

def load_or_compute(path: str | Path, fn: Callable[..., Any], *args, **kwargs) -> Any:
    """path 존재시 load, 없으면 fn(*args, **kwargs) 실행 + 저장 + return.

    suffix 별 dispatch:
      .pt    → save_pt / load_pt    (embedding tensor cache)
      .json  → save_json / load_json
      .jsonl → save_jsonl / load_jsonl

    사용자가 if 분기로 직접 control 하고 싶으면 raw save_pt / load_pt / ... 직접 호출.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if p.exists():
        if suffix == ".pt":
            return load_pt(p)
        if suffix == ".json":
            return load_json(p)
        if suffix == ".jsonl":
            return load_jsonl(p)
        raise ValueError(f"지원 안 하는 suffix: {suffix} ({p})")

    out = fn(*args, **kwargs)

    if suffix == ".pt":
        save_pt(p, out)
    elif suffix == ".json":
        save_json(p, out)
    elif suffix == ".jsonl":
        save_jsonl(p, out)
    else:
        raise ValueError(f"지원 안 하는 suffix: {suffix} ({p})")

    return out
