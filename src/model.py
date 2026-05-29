"""모델 load + 2가지 모드:
  (1) get_embeddings : 특정 layer 의 hidden state 추출 (forward 만, generation X)
  (2) generate       : output decoding 까지 (직접 응답 확인용)

prompt 는 list[str]. batch_size 인자.
chat template 은 default 적용 (Llama-3.2 instruct). raw 분석시 use_chat_template=False.

embedding return 의 spans 에 두 위치 정보 포함 (사용자 가이드: "harm direction은 instruction
내부, refuse direction은 post-instruction 위치"):
  - user_token_span (start, end) : chat template 안의 user prompt token 위치 (instruction 내부)
  - last_token_idx              : 입력 마지막 token (post-instruction = assistant 답변 직전)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch
from tqdm.auto import tqdm

from src.utils import get_logger, free_cuda

logger = get_logger(__name__)


# ─────────────────────── load ────────────────────────

def load_model(
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
    *,
    dtype: str = "bf16",
    device_map: str = "auto",
):
    """model + tokenizer 한번에 load. tokenizer.pad_token / padding_side 자동 set.

    dtype: "bf16" | "fp16" | "fp32".
    return: (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        # Llama 는 pad token 없음 → eos 로 대체
        tokenizer.pad_token = tokenizer.eos_token
    # generation 호환 (Llama 는 우측에 새 토큰 생성). embedding 은 attention_mask 로 valid 영역만 추출 → padding side 무관.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    logger.info(f"loaded {model_id} ({dtype}, device={model.device})")
    return model, tokenizer


# ─────────────────────── chat template ────────────────────────

def apply_chat(tokenizer, prompts: Sequence[str], *, add_generation_prompt: bool = False) -> list[dict]:
    """user message wrap → chat template 적용. user prompt 의 char span 도 같이.

    add_generation_prompt=True : assistant prefix 까지 추가 (post-instruction 위치 잡으려면 True 필요).

    return: list[{"text": str, "user_char_span": (int, int)}]
        text.find(prompt) 로 user prompt 의 char position 잡음. 못 찾으면 (0, len(text)) fallback.
    """
    out = []
    for p in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        s = text.find(p)
        if s < 0:
            # template 처리 중 escape 등 → fallback
            s, e = 0, len(text)
        else:
            e = s + len(p)
        out.append({"text": text, "user_char_span": (s, e)})
    return out


# ─────────────────────── embedding 추출 ────────────────────────

def _decoder_layers(model):
    """Llama 계열 decoder layer list."""
    return model.model.layers


@contextmanager
def _capture(model, layers: Sequence[int]) -> Iterator[dict[int, list[torch.Tensor]]]:
    """forward hook 으로 layer 별 hidden state 캡처. context 끝나면 hook 제거."""
    buf: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            buf[layer_idx].append(h.detach())
            return output
        return hook

    decoder = _decoder_layers(model)
    for l in layers:
        handles.append(decoder[l].register_forward_hook(make_hook(l)))
    try:
        yield buf
    finally:
        for h in handles:
            h.remove()


def get_embeddings(
    model,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    *,
    batch_size: int = 4,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
) -> dict[str, Any]:
    """prompt 별 layer 별 hidden state 추출 (forward 만, padding 제거).

    add_generation_prompt=True (default): chat template 끝에 assistant header 추가 →
    입력 마지막 token = "post-instruction" position (assistant 답변 직전).

    return:
      {
        "embeddings":  {l: list[tensor[T_i, H]]},  # prompt 별 (padding 제거된) hidden, fp32 cpu
        "tokens":      list[list[str]],             # prompt 별 token 문자열
        "input_texts": list[str],                    # chat template 적용 후 텍스트
        "spans":       list[{
            "user_token_span": (int, int),  # user prompt token (start, end) — instruction 내부 분석용
            "last_token_idx":  int,           # 마지막 valid token — post-instruction 위치
        }],
      }
    """
    prompts = list(prompts)
    layers = list(layers)

    if use_chat_template:
        chat = apply_chat(tokenizer, prompts, add_generation_prompt=add_generation_prompt)
        texts = [c["text"] for c in chat]
        user_char_spans = [c["user_char_span"] for c in chat]
    else:
        texts = list(prompts)
        user_char_spans = [(0, len(t)) for t in texts]

    out_emb: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    out_tokens: list[list[str]] = []
    out_spans: list[dict] = []

    bs = max(1, batch_size)
    for i in tqdm(range(0, len(texts), bs), desc="get_embeddings", leave=False):
        chunk_texts = texts[i:i + bs]
        chunk_user_spans = user_char_spans[i:i + bs]

        # fast tokenizer 의 offset_mapping 으로 char → token 변환
        enc = tokenizer(
            chunk_texts, return_tensors="pt", padding=True,
            return_offsets_mapping=True, add_special_tokens=False,
        ).to(model.device)
        attn = enc["attention_mask"]
        offsets = enc.pop("offset_mapping")  # forward 에 안 넘김

        with _capture(model, layers) as buf, torch.inference_mode():
            # model.model (LlamaModel) 직접 호출 — lm_head(hidden→vocab) 스킵.
            # hook 이 decoder layer 에 걸려 hidden 캡처는 동일(결과 bit-identical).
            # 8B 는 vocab 128k 라 lm_head logits 가 거대 → L4 OOM 회피. 3B 도 더 가벼움.
            model.model(**enc, use_cache=False)

        # span / token 계산
        for j in range(len(chunk_texts)):
            valid = attn[j].bool()
            valid_offsets = offsets[j][valid]  # [T_valid, 2]
            cs, ce = chunk_user_spans[j]

            # user span 안에 들어오는 token: char 범위가 [cs, ce] 안에 있고 비어있지 않은 (special token X)
            user_indices = [
                tok_i for tok_i in range(valid_offsets.shape[0])
                if valid_offsets[tok_i, 0].item() < valid_offsets[tok_i, 1].item()  # non-empty (special token 제외)
                and valid_offsets[tok_i, 0].item() >= cs
                and valid_offsets[tok_i, 1].item() <= ce
            ]
            if user_indices:
                user_start, user_end = user_indices[0], user_indices[-1] + 1
            else:
                # fallback: 전체 valid 영역
                user_start, user_end = 0, valid_offsets.shape[0]

            out_spans.append({
                "user_token_span": (user_start, user_end),
                "last_token_idx": valid_offsets.shape[0] - 1,
            })

            valid_ids = enc["input_ids"][j][valid].cpu().tolist()
            out_tokens.append(tokenizer.convert_ids_to_tokens(valid_ids))

        # embedding (padding 제거)
        for l in layers:
            h_full = buf[l][-1]  # [B, T_max, H]
            for j in range(len(chunk_texts)):
                valid = attn[j].bool()
                # bf16 → fp32 cast: downstream (mean / cosine 등) 정밀도
                out_emb[l].append(h_full[j, valid].to(torch.float32).cpu())

    free_cuda()  # batch loop 끝나고 GPU cache 회수 (Colab 누적 회피)
    return {
        "embeddings": out_emb,
        "tokens": out_tokens,
        "input_texts": texts,
        "spans": out_spans,
    }


# ─────────────────────── generation ────────────────────────

def generate(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    temperature: float = 0.0,
    do_sample: bool = False,
    use_chat_template: bool = True,
) -> list[str]:
    """prompt 별 응답 (생성된 부분만, 입력 echo 제외). default greedy (do_sample=False)."""
    prompts = list(prompts)
    if use_chat_template:
        chat = apply_chat(tokenizer, prompts, add_generation_prompt=True)
        texts = [c["text"] for c in chat]
    else:
        texts = list(prompts)

    outs: list[str] = []
    bs = max(1, batch_size)
    for i in tqdm(range(0, len(texts), bs), desc="generate", leave=False):
        chunk = texts[i:i + bs]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = gen[:, prompt_len:]
        outs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    free_cuda()  # generate batch loop 끝나고 GPU cache 회수
    return outs
