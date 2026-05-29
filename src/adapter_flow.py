"""over-refusal / jailbreak 교정 adapter — 모델 freeze, hidden 에 forward hook 으로 작용.

진단 (notebooks/03):
  jailbreak/over-refuse 는 "harmful 인지 축들"(여러 개) 과 "거절 축" 이 부분적으로만 겹쳐서 (층별 cos ~0.1~0.4), 표면 신호가 그 틈을 잘못 다뤄 일어난다.
  jailbreak = 인지 축들 중 거절과 연결된 부분이 (표면 위장 때문에) 안 켜짐 → 안 거절. over-refuse = 표면 단어가 그 부분을 잘못 켬 → 거절.
  (figure A1: v_harm 한 축으론 jailbreak≈harmful, v_refuse 축으론 큰 차이.)

처방 = adapter 가 그 입력의 hidden 을 — v_harm/v_refuse 관련 subspace *위에서* — "올바른" 위치로 옮긴다 (jailbreak → harmful 쪽, over-refuse → 정상 쪽). 나머지 차원(질문 의미)은 보존.
  명시적 분류기(gate) 없음 — `Flow.v_θ` 가 conditional(hidden 입력)이라 패턴 보고 알아서 (jailbreak 패턴 → harmful 쪽 delta, over-refuse 패턴 → 정상 쪽, harmful/benign → ≈0; 학습으로).
  → 모델이 *알아서* jailbreak 는 거절·over-refuse 는 응답 (축이 정렬되니까) — 'jailbreak 면 거절해' 같은 switching 이 아님.

  - Flow(v_θ) : hidden(H) + layer 임베딩 → 작은 MLP `W2·gelu(W1·[h;t_emb] + b1)` → delta(H). W2 init 0 → 학습 전 delta 0.
                layer 인덱스가 입력 → 한 함수가 모든 작용 layer 공유 (flow-matching — 층 간 일관, query별 연속 경로).
  - 적용(steer) : 작용 layer(예: 8~23, 인지~거절 구간)의 user_token_span 토큰들 + 답변-직전 토큰 hidden 에 —
                v_θ(h,l) 을 그 층의 v_harm⊕v_refuse subspace(orthonormal basis Q) 로 *사영* 한 것만 더함 (h ← h + Q Qᵀ v_θ).
                다른 방향 차단 — 질문 의미·진짜 harmful 표상 보호. prefill 만 수정 → KV-cache 반영, decode token 엔 안 켬.

학습 loss (notebooks/04): layer-local — jailbreak 뚫림 → v_harm subspace 사영을 harmful 수준으로 / over-refuse 거절당함 → v_refuse subspace 사영을 정상 음수 수준으로 /
  harmful·benign·jailbreak 막힘·over-refuse 정상 → delta≈0 (preserve, λ_preserve 최대) / + uncertainty weighting (데이터 불균형).
flow-matching 정공법(모델 forward 로 trajectory 학습)은 — 위 layer-local 근사로 1차 확인 후 (04 의 inject ② 가 'layer-coherent 필요'를 보였으니) 도입.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


def _decoder_layers(model):
    """Llama 계열 decoder layer list."""
    return model.model.layers


class Flow(nn.Module):
    """v_θ(h, l) — hidden + layer 임베딩 → delta. W2 init 0. layer 인덱스가 입력 → 한 함수가 act_layers 전부 공유."""

    def __init__(self, hidden: int, layers, rank: int = 256, t_dim: int = 32):
        super().__init__()
        self.layers = list(layers)
        self.l_idx = {l: i for i, l in enumerate(self.layers)}
        self.t_emb = nn.Embedding(len(self.layers), t_dim)
        self.W1 = nn.Parameter(torch.randn(rank, hidden + t_dim) * (hidden + t_dim) ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(rank))
        self.W2 = nn.Parameter(torch.zeros(hidden, rank))   # init 0 → 학습 전 delta 0

    def delta(self, l: int, h: torch.Tensor) -> torch.Tensor:    # h: [..., hidden] @ layer l → [..., hidden] (반환 dtype = h.dtype)
        if l not in self.l_idx:
            return torch.zeros_like(h)
        te = self.t_emb(torch.tensor(self.l_idx[l], device=h.device)).to(h.dtype)   # [t_dim]
        te = te.expand(*h.shape[:-1], -1)                                           # [..., t_dim]
        x = torch.cat([h, te], dim=-1)                                              # [..., hidden+t_dim]
        return (F.gelu(x @ self.W1.to(x.dtype).T + self.b1.to(x.dtype)) @ self.W2.to(x.dtype).T).to(h.dtype)


class OverRefuseAdapter(nn.Module):
    """flow v_θ + v_harm/v_refuse subspace basis(projection 용) + target(loss/평가용). gate 없음 — v_θ 가 conditional."""

    def __init__(self, hidden: int, *, act_layers, subspace_basis: dict, v_harm_unit: dict, v_refuse_unit: dict,
                 harm_proj_target: dict | None = None, norm_safe_vrefuse: dict | None = None, rank: int = 256, t_dim: int = 32):
        super().__init__()
        self.act_layers = list(act_layers)
        self.flow = Flow(hidden, self.act_layers, rank, t_dim)
        for l in self.act_layers:
            self.register_buffer(f"Q_{l}", subspace_basis[l].float())                # [k, hidden] orthonormal — projection 용
            if l in v_harm_unit:   self.register_buffer(f"vh_{l}", v_harm_unit[l].float())
            if l in v_refuse_unit: self.register_buffer(f"vr_{l}", v_refuse_unit[l].float())
            if harm_proj_target and l in harm_proj_target:   self.register_buffer(f"htgt_{l}", torch.as_tensor(float(harm_proj_target[l])))
            if norm_safe_vrefuse and l in norm_safe_vrefuse: self.register_buffer(f"rtgt_{l}", torch.as_tensor(float(norm_safe_vrefuse[l])))

    def delta(self, l: int, h: torch.Tensor) -> torch.Tensor:
        """v_θ(h,l) 을 그 층 subspace 로 사영 — 다른 방향(질문 의미·진짜 harmful 표상) 변화 차단."""
        d = self.flow.delta(l, h)
        if not hasattr(self, f"Q_{l}"):
            return d
        Q = getattr(self, f"Q_{l}").to(d.dtype)                                      # [k, hidden] orthonormal
        return (d @ Q.T) @ Q

    @contextmanager
    def steer(self, model, user_token_spans, last_token_idxs):
        """모든 입력에 v_θ 적용 (gate 없음 — conditional 이라 harmful/benign 엔 ≈0 학습됨).
        작용 layer 의 user_token_span 토큰들 + 답변-직전 token hidden 에 delta 더함. prefill 만 수정 → KV-cache 반영, decode 엔 안 켬.
        user_token_spans: list[(s,e)] (batch 순), last_token_idxs: list[int]."""
        decoder = _decoder_layers(model)
        handles = []

        def make_hook(l: int):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out         # [B, T, H]
                T = h.shape[1]
                delta = torch.zeros_like(h)
                for b in range(h.shape[0]):
                    s = int(user_token_spans[b][0]); e = min(int(user_token_spans[b][1]), T); li = int(last_token_idxs[b])
                    if e > s:  delta[b, s:e] = self.delta(l, h[b, s:e])
                    if li < T: delta[b, li] = delta[b, li] + self.delta(l, h[b, li])
                h_new = h + delta
                return (h_new,) + tuple(out[1:]) if isinstance(out, tuple) else h_new
            return hook

        try:
            for l in self.act_layers:
                handles.append(decoder[l].register_forward_hook(make_hook(l)))
            yield
        finally:
            for hd in handles:
                hd.remove()


# ─────────────────────── helper ───────────────────────

def orthonormal_basis(vecs: list[torch.Tensor]) -> torch.Tensor:
    """방향 벡터 리스트 → orthonormal basis [k, hidden] (QR). subspace projection 용."""
    M = torch.stack([v.float() / (v.float().norm() + 1e-8) for v in vecs], dim=1)    # [hidden, k]
    Q, _ = torch.linalg.qr(M)
    return Q.T                                                                        # [k, hidden]


def adapter_loss(*, jb_target_loss=None, or_target_loss=None, preserve_delta_norm=None,
                 log_var=None, lam_preserve: float = 5.0):
    """L = uncertainty([jb_loss, or_loss]) + lam_preserve * preserve_delta_norm.
    preserve 는 uncertainty 에서 *제외* — 안전 보루라 자동 가중치로 약화되면 안 됨
    (이전 실험: preserve 가 uncertainty 자동 가중치로 약화 → ep5 L≈1800 폭주).
    preserve 는 norm² 대신 norm — 제곱이 진동 키움(1차 grad 가 안정).
    log_var=[s_jb, s_or] (2개; preserve 자리 제거). 없으면 가중치 1.
    """
    parts = [p for p in (jb_target_loss, or_target_loss, preserve_delta_norm) if p is not None]
    L = parts[0].new_zeros(()) if parts else torch.zeros(())
    def w(i): return torch.exp(-log_var[i]) if log_var is not None else 1.0
    def s(i): return log_var[i] if log_var is not None else 0.0
    if jb_target_loss is not None:        L = L + w(0) * jb_target_loss + s(0)
    if or_target_loss is not None:        L = L + w(1) * or_target_loss + s(1)
    if preserve_delta_norm is not None:   L = L + lam_preserve * preserve_delta_norm
    return L
