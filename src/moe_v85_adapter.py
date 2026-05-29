"""V85 one-pass MoE adapter.

The adapter routes and applies the two HistoryFlow experts inside the same
prefill forward.  The gate is history-aware and shared across ACT_LAYERs.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter_flow import _decoder_layers
from .adapter_history_flow import HistoryFlow

__all__ = ["V85Gate", "V85OnePassMoEAdapter"]


def _lookup_layer(mapping: dict | None, layer: int):
    if not mapping:
        return None
    if layer in mapping:
        return mapping[layer]
    key = str(layer)
    if key in mapping:
        return mapping[key]
    return None


class V85Gate(nn.Module):
    """History-aware shared gate."""

    CLASSES_4WAY = ("harmful", "jailbreak", "over_refuse", "benign")

    def __init__(self, hidden: int, layers, *, rank: int = 128, t_dim: int = 16, history_n: int = 2):
        super().__init__()
        self.hidden = int(hidden)
        self.layers = [int(layer) for layer in layers]
        self.layer_index = {layer: i for i, layer in enumerate(self.layers)}
        self.history_n = int(history_n)

        in_dim = self.hidden * (self.history_n + 1) + int(t_dim)
        self.t_emb = nn.Embedding(len(self.layers), int(t_dim))
        self.W1 = nn.Parameter(torch.randn(int(rank), in_dim) * in_dim**-0.5)
        self.b1 = nn.Parameter(torch.zeros(int(rank)))
        self.head_4way = nn.Linear(int(rank), 4)
        self.head_or = nn.Linear(int(rank), 1)

    def _pad_history(self, current: torch.Tensor, history: list[torch.Tensor]) -> list[torch.Tensor]:
        hist = list(history)
        while len(hist) < self.history_n:
            hist.insert(0, torch.zeros_like(current))
        return hist[-self.history_n :]

    def features(self, layer: int, current: torch.Tensor, history: list[torch.Tensor]) -> torch.Tensor:
        li = int(layer)
        if li not in self.layer_index:
            raise KeyError(f"layer {li} is not in gate layers {self.layers}")
        hist = self._pad_history(current, history)
        te = self.t_emb(torch.tensor(self.layer_index[li], device=current.device)).to(current.dtype)
        te = te.expand(*current.shape[:-1], -1)
        x = torch.cat([current] + hist + [te], dim=-1)
        return F.gelu(x @ self.W1.to(x.dtype).T + self.b1.to(x.dtype))

    def logits(self, layer: int, current: torch.Tensor, history: list[torch.Tensor]):
        feat = self.features(layer, current, history)
        return self.head_4way(feat), self.head_or(feat).squeeze(-1)

    def coefficients(self, layer: int, current: torch.Tensor, history: list[torch.Tensor]):
        logits_4way, logits_or = self.logits(layer, current, history)
        p4 = logits_4way.softmax(-1)
        return p4[..., 1], torch.sigmoid(logits_or)


class V85OnePassMoEAdapter(nn.Module):
    """Two-expert MoE with one-pass routing and steering."""

    def __init__(
        self,
        hidden: int,
        *,
        act_layers,
        subspace_basis: dict,
        rank: int = 256,
        t_dim: int = 32,
        history_n: int = 2,
        gate_rank: int = 128,
        gate_t_dim: int = 16,
        abstain_threshold: float = 0.20,
        jb_scale: float = 1.0,
        or_scale: float = 2.0,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.act_layers = [int(layer) for layer in act_layers]
        self.history_n = int(history_n)
        self.abstain_threshold = float(abstain_threshold)
        self.jb_scale = float(jb_scale)
        self.or_scale = float(or_scale)

        self.gate = V85Gate(
            self.hidden,
            self.act_layers,
            rank=gate_rank,
            t_dim=gate_t_dim,
            history_n=self.history_n,
        )
        self.jb_flow = HistoryFlow(self.hidden, self.act_layers, rank, t_dim, self.history_n)
        self.or_flow = HistoryFlow(self.hidden, self.act_layers, rank, t_dim, self.history_n)

        for layer in self.act_layers:
            q = _lookup_layer(subspace_basis, layer)
            if q is None:
                raise KeyError(f"missing subspace_basis for layer {layer}")
            self.register_buffer(f"Q_{layer}", torch.as_tensor(q, dtype=torch.float32))

    def gate_parameters(self) -> Iterable[nn.Parameter]:
        return self.gate.parameters()

    def expert_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.jb_flow.parameters()
        yield from self.or_flow.parameters()

    def set_gate_requires_grad(self, enabled: bool) -> None:
        for param in self.gate.parameters():
            param.requires_grad = bool(enabled)

    def set_expert_requires_grad(self, enabled: bool) -> None:
        for param in self.jb_flow.parameters():
            param.requires_grad = bool(enabled)
        for param in self.or_flow.parameters():
            param.requires_grad = bool(enabled)

    def set_steering_scales(
        self,
        *,
        jb_scale: float | None = None,
        or_scale: float | None = None,
        abstain_threshold: float | None = None,
    ) -> None:
        if jb_scale is not None:
            self.jb_scale = float(jb_scale)
        if or_scale is not None:
            self.or_scale = float(or_scale)
        if abstain_threshold is not None:
            self.abstain_threshold = float(abstain_threshold)

    def _project_q(self, layer: int, delta: torch.Tensor) -> torch.Tensor:
        q = getattr(self, f"Q_{int(layer)}").to(delta.device, delta.dtype)
        return (delta @ q.T) @ q

    def delta(self, expert: str, layer: int, h: torch.Tensor, h_history: list[torch.Tensor] | None = None):
        li = int(layer)
        history = h_history or []
        if expert == "jb":
            out = self.jb_flow.delta(li, h, history)
        elif expert == "or":
            out = self.or_flow.delta(li, h, history)
        else:
            raise ValueError(f"unknown expert {expert!r}")
        return self._project_q(li, out)

    def gate_logits(self, layer: int, current_mean: torch.Tensor, history_means: list[torch.Tensor]):
        return self.gate.logits(int(layer), current_mean, history_means)

    def gate_coefficients(self, layer: int, current_mean: torch.Tensor, history_means: list[torch.Tensor]):
        return self.gate.coefficients(int(layer), current_mean, history_means)

    def _user_mean(self, h: torch.Tensor, user_token_spans, T: int) -> torch.Tensor:
        means = []
        for b in range(h.shape[0]):
            s = int(user_token_spans[b][0])
            e = min(int(user_token_spans[b][1]), T)
            if e > s:
                means.append(h[b, s:e].float().mean(0))
            else:
                means.append(torch.zeros(h.shape[-1], device=h.device, dtype=torch.float32))
        return torch.stack(means)

    def _apply_mask(self, h: torch.Tensor, user_token_spans, last_token_idxs, T: int) -> torch.Tensor:
        mask = torch.zeros(h.shape[0], T, 1, device=h.device, dtype=h.dtype)
        for b in range(h.shape[0]):
            s = int(user_token_spans[b][0])
            e = min(int(user_token_spans[b][1]), T)
            li = max(0, min(int(last_token_idxs[b]), T - 1))
            if e > s:
                mask[b, s:e] = 1
            mask[b, li] = 1
        return mask

    def _threshold(self, p: torch.Tensor) -> torch.Tensor:
        threshold = float(self.abstain_threshold)
        if threshold <= 0:
            return p
        return torch.where(p >= threshold, p, torch.zeros_like(p))

    @contextmanager
    def steer(self, model, user_token_spans, last_token_idxs, *, train_mode: bool = False, capture: bool = False):
        """Register one-pass hooks and yield routing logs plus optional loss records."""
        decoder = _decoder_layers(model)
        handles = []
        layer_cache: dict[int, torch.Tensor] = {}
        state = {
            "routing_log": {"p_jb": {}, "p_jb_scaled": {}, "p_or": {}, "p_or_scaled": {}},
            "records": {},
        }

        def make_hook(layer: int):
            li = int(layer)
            ai = self.act_layers.index(li)

            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                T = h.shape[1]
                if T == 1:
                    return out

                h_in = h.detach()
                grad_context = torch.enable_grad() if train_mode else torch.no_grad()
                with grad_context:
                    prev_layers = self.act_layers[max(0, ai - self.history_n) : ai]
                    h_history = [layer_cache[prev] for prev in prev_layers if prev in layer_cache]

                    current_mean = self._user_mean(h_in, user_token_spans, T)
                    history_means = [self._user_mean(hist, user_token_spans, T) for hist in h_history]

                    logits_4way, logits_or = self.gate_logits(li, current_mean, history_means)
                    p4 = logits_4way.softmax(-1)
                    p_jb_raw = p4[..., 1]
                    p_or_raw = torch.sigmoid(logits_or)
                    p_jb = self._threshold(p_jb_raw) * self.jb_scale
                    p_or = self._threshold(p_or_raw) * self.or_scale

                    d_jb = self.delta("jb", li, h_in, h_history)
                    d_or = self.delta("or", li, h_in, h_history)
                    view_shape = (-1,) + (1,) * (h_in.ndim - 1)
                    delta = p_jb.to(h_in.dtype).view(view_shape) * d_jb
                    delta = delta + p_or.to(h_in.dtype).view(view_shape) * d_or

                    apply_mask = self._apply_mask(h_in, user_token_spans, last_token_idxs, T)
                    h_new = (h_in + delta * apply_mask).to(h.dtype)

                layer_cache[li] = h_new
                state["routing_log"]["p_jb"][li] = p_jb_raw.detach().float().cpu()
                state["routing_log"]["p_jb_scaled"][li] = p_jb.detach().float().cpu()
                state["routing_log"]["p_or"][li] = p_or_raw.detach().float().cpu()
                state["routing_log"]["p_or_scaled"][li] = p_or.detach().float().cpu()

                if capture:
                    state["records"][li] = {
                        "h_in": h_in,
                        "h_corr": h_new,
                        "d_jb": d_jb,
                        "d_or": d_or,
                        "p_jb": p_jb,
                        "p_jb_raw": p_jb_raw,
                        "p_or": p_or,
                        "p_or_raw": p_or_raw,
                        "apply_mask": apply_mask,
                        "gate_logits_4way": logits_4way,
                        "gate_logits_or": logits_or,
                        "current_mean": current_mean,
                        "history_means": history_means,
                    }

                returned = h_new.detach()
                return (returned,) + tuple(out[1:]) if isinstance(out, tuple) else returned

            return hook

        try:
            for layer in self.act_layers:
                handles.append(decoder[int(layer)].register_forward_hook(make_hook(int(layer))))
            yield state
        finally:
            for handle in handles:
                handle.remove()
            layer_cache.clear()
