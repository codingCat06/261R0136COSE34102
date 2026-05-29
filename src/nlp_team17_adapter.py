"""NLP-TEAM 17 one-pass MoE adapter.

NLP-TEAM 17 keeps the trained v85 one-pass architecture and evaluates it with
the final inference-time steering coefficients.
"""
from __future__ import annotations

from contextlib import contextmanager
from inspect import signature

import torch

from .adapter_flow import _decoder_layers
from .moe_v85_adapter import V85Gate, V85OnePassMoEAdapter

__all__ = ["NLPTeam17Adapter", "V85Gate"]


class NLPTeam17Adapter(V85OnePassMoEAdapter):
    """V85 one-pass adapter with NLP-TEAM 17 inference-time scales."""

    def __init__(self, *args, jb_scale: float = 2.0, **kwargs):
        params = signature(V85OnePassMoEAdapter.__init__).parameters
        super_kwargs = dict(kwargs)
        if "jb_scale" in params:
            super_kwargs["jb_scale"] = jb_scale
        or_scale = super_kwargs.get("or_scale", None)
        if "or_scale" not in params:
            or_scale = super_kwargs.pop("or_scale", or_scale)
        super().__init__(*args, **super_kwargs)
        self.jb_scale = float(jb_scale)
        if or_scale is not None:
            self.or_scale = float(or_scale)
        elif not hasattr(self, "or_scale"):
            self.or_scale = 2.0

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

    def _threshold(self, p: torch.Tensor) -> torch.Tensor:
        threshold = float(getattr(self, "abstain_threshold", 0.0))
        if threshold <= 0:
            return p
        return torch.where(p >= threshold, p, torch.zeros_like(p))

    @contextmanager
    def steer(self, model, user_token_spans, last_token_idxs, *, train_mode: bool = False, capture: bool = False):
        """Register one-pass hooks with NLP-TEAM 17 steering scales."""
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
                    p_jb = self._threshold(p_jb_raw) * float(self.jb_scale)
                    p_or = self._threshold(p_or_raw) * float(self.or_scale)

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
