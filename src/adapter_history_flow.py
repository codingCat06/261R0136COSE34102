"""v14 history-aware Flow — trajectory 정보 활용.

v13 의 layer-local Flow 의 한계:
- 각 layer 의 delta 가 *그 layer 의 hidden 만* 보고 결정
- jb_blocked vs jb_corr 의 *trajectory 차이* 구분 못함 (둘 다 jailbreak prompt → layer hidden 비슷)

v14 의 history-aware Flow:
- 각 layer 의 delta 가 *현재 layer + 이전 history_n 개 layer 의 hidden* 함께 보고 결정
- trajectory pattern 인식 가능 → 보존 그룹 (jb_blocked, harm_refuse) 자연스럽게 약한 delta

구조:
  HistoryFlow:
    delta(l, h_l, h_history) — h_history = list of [..., H] (이전 layer hidden)
    input = concat(h_l, h_history[-history_n:], t_emb) → MLP → delta

  HistoryAdapter:
    학습 시 — 각 layer 의 hidden_states[l] + hidden_states[l-1..l-history_n] 받음
    inference 시 (steer hook) — _layer_cache 로 이전 layer hidden 저장
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter_flow import _decoder_layers


class HistoryFlow(nn.Module):
    """v_θ(h_l, h_history, l) — 이전 layer hidden 도 input."""

    def __init__(self, hidden: int, layers, rank: int = 256, t_dim: int = 32, history_n: int = 2):
        super().__init__()
        self.hidden = hidden
        self.layers = list(layers)
        self.l_idx = {l: i for i, l in enumerate(self.layers)}
        self.history_n = history_n
        self.t_emb = nn.Embedding(len(self.layers), t_dim)
        # input = h_l + history_n × h_prev + t_emb
        in_dim = hidden * (history_n + 1) + t_dim
        self.W1 = nn.Parameter(torch.randn(rank, in_dim) * in_dim ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(rank))
        self.W2 = nn.Parameter(torch.zeros(hidden, rank))   # init 0 → 학습 전 delta 0

    def delta(self, l: int, h_l: torch.Tensor, h_history: list) -> torch.Tensor:
        """h_l: [..., H]. h_history: list of [..., H] (이전 layer hidden, 최근 → 옛 순서).
        부족하면 zero pad. history_n 만 사용 (가장 최근 N개)."""
        if l not in self.l_idx:
            return torch.zeros_like(h_l)
        # history 정규화: 길이 history_n, 부족하면 앞에 zero (가장 옛쪽)
        hist = list(h_history)
        while len(hist) < self.history_n:
            hist = [torch.zeros_like(h_l)] + hist
        hist = hist[-self.history_n:]   # 가장 최근 N 개

        te = self.t_emb(torch.tensor(self.l_idx[l], device=h_l.device)).to(h_l.dtype)
        te = te.expand(*h_l.shape[:-1], -1)
        x = torch.cat([h_l] + hist + [te], dim=-1)
        return (F.gelu(x @ self.W1.to(x.dtype).T + self.b1.to(x.dtype)) @ self.W2.to(x.dtype).T).to(h_l.dtype)


class HistoryAdapter(nn.Module):
    """v14 adapter — HistoryFlow + Q projection + steer with layer cache.

    학습 시 — 학습 코드가 hidden_states[l-1], hidden_states[l] 같은 식으로 명시 전달.
    inference 시 — steer hook 이 layer 마다 forward output 을 cache, 다음 hook 이 사용.
    """

    def __init__(self, hidden: int, *, act_layers, subspace_basis: dict,
                 v_harm_unit: dict, v_refuse_unit: dict,
                 harm_proj_target: dict | None = None, norm_safe_vrefuse: dict | None = None,
                 rank: int = 256, t_dim: int = 32, history_n: int = 2):
        super().__init__()
        self.act_layers = list(act_layers)
        self.history_n = history_n
        self.flow = HistoryFlow(hidden, self.act_layers, rank, t_dim, history_n)
        for l in self.act_layers:
            self.register_buffer(f"Q_{l}", subspace_basis[l].float())
            if l in v_harm_unit:   self.register_buffer(f"vh_{l}", v_harm_unit[l].float())
            if l in v_refuse_unit: self.register_buffer(f"vr_{l}", v_refuse_unit[l].float())
            if harm_proj_target and l in harm_proj_target:
                self.register_buffer(f"htgt_{l}", torch.as_tensor(float(harm_proj_target[l])))
            if norm_safe_vrefuse and l in norm_safe_vrefuse:
                self.register_buffer(f"rtgt_{l}", torch.as_tensor(float(norm_safe_vrefuse[l])))

    def delta(self, l: int, h: torch.Tensor, h_history: list | None = None) -> torch.Tensor:
        """학습용 — h_history 명시 전달. Q projection 자동."""
        if h_history is None:
            h_history = []
        d = self.flow.delta(l, h, h_history)
        if not hasattr(self, f"Q_{l}"):
            return d
        Q = getattr(self, f"Q_{l}").to(d.dtype)
        return (d @ Q.T) @ Q

    @contextmanager
    def steer(self, model, user_token_spans, last_token_idxs):
        """inference — steer hook 이 layer cache 유지. 각 hook 이 이전 layer hidden 사용."""
        decoder = _decoder_layers(model)
        handles = []
        layer_cache = {}   # {layer_idx: cached_hidden} — modified hidden 저장

        def make_hook(l):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                T = h.shape[1]
                # history 만들기: 이전 act_layers 의 cached hidden
                act_layer_idx = self.act_layers.index(l)
                prev_layers = self.act_layers[max(0, act_layer_idx - self.history_n):act_layer_idx]
                h_history_full = [layer_cache.get(pl) for pl in prev_layers]
                h_history_full = [h_h for h_h in h_history_full if h_h is not None]

                delta = torch.zeros_like(h)
                for b in range(h.shape[0]):
                    s = int(user_token_spans[b][0])
                    e = min(int(user_token_spans[b][1]), T)
                    li = int(last_token_idxs[b])
                    if e > s:
                        # span 의 hidden — history 도 같은 span 추출
                        h_span = h[b, s:e]    # [span, H]
                        h_history_span = [hh[b, s:e] for hh in h_history_full]
                        delta[b, s:e] = self.delta(l, h_span, h_history_span)
                    if li < T:
                        h_last = h[b, li]
                        h_history_last = [hh[b, li] for hh in h_history_full]
                        delta[b, li] = delta[b, li] + self.delta(l, h_last, h_history_last)

                h_new = h + delta
                # 다음 layer 의 hook 이 사용할 cache 저장 (modified hidden)
                layer_cache[l] = h_new
                return (h_new,) + tuple(out[1:]) if isinstance(out, tuple) else h_new
            return hook

        try:
            for l in self.act_layers:
                handles.append(decoder[l].register_forward_hook(make_hook(l)))
            yield
        finally:
            for hd in handles:
                hd.remove()
            layer_cache.clear()
