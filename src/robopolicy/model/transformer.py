"""Transformer building blocks, written from scratch.

Nothing here is imported from ``torch.nn`` beyond ``Linear``/``LayerNorm``/``Dropout``
and the module machinery — the attention math and the encoder/decoder assembly are
implemented by hand. The design follows DETR / ACT:

- **Post-norm** layers (``pre_norm=False`` by default), matching the official ACT.
- Positional embeddings are added to the **query and key** projections at every
  attention call, never to the values. This is the DETR convention that ACT inherits.

Shapes use ``(seq, batch, dim)`` throughout (like ``nn.Transformer``'s default), which
keeps the attention reshapes simple.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _clone(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


_USE_FUSED_LN = False


def set_fused_layernorm(enabled: bool) -> None:
    """Toggle the custom fused residual+LayerNorm CUDA kernel for the inference path.

    Off by default: training keeps PyTorch's LayerNorm (autograd + already well tuned).
    When on and inputs are eligible (CUDA float32, no grad), post-norm sublayers
    dispatch to the fused kernel. It is output-parity with the PyTorch path
    (kernels/test_parity.py), so it does not change model behavior — see eval step 8.
    """
    global _USE_FUSED_LN
    _USE_FUSED_LN = enabled


def _norm_residual(norm: nn.LayerNorm, residual: Tensor, sublayer_out: Tensor) -> Tensor:
    """Post-norm combiner: ``norm(residual + sublayer_out)``.

    Dispatches to the fused CUDA kernel when enabled and eligible; otherwise the plain
    PyTorch path (identical math). The fused path folds the residual add into the
    LayerNorm reduction, removing a kernel launch and a full global-memory round-trip.
    """
    if (
        _USE_FUSED_LN
        and residual.is_cuda
        and residual.dtype == torch.float32
        and not torch.is_grad_enabled()
    ):
        from ..fused_ops import residual_layernorm

        return residual_layernorm(residual, sublayer_out, norm.weight, norm.bias, norm.eps)
    return norm(residual + sublayer_out)


def sinusoidal_pos_embedding_1d(num_positions: int, dim: int) -> Tensor:
    """Standard 1D sinusoidal position embedding, shape ``(num_positions, dim)``."""
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(num_positions, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SinusoidalPositionEmbedding2D(nn.Module):
    """2D sinusoidal position embedding for image feature maps (DETR-style).

    Given a feature map ``(B, C, H, W)`` produces ``(B, dim, H, W)`` where the first
    half of the channels encode the y position and the second half the x position.
    """

    def __init__(self, dim: int, temperature: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.half_dim = dim // 2
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        device = x.device
        y_embed = torch.arange(1, h + 1, dtype=torch.float32, device=device).unsqueeze(1)  # (H,1)
        x_embed = torch.arange(1, w + 1, dtype=torch.float32, device=device).unsqueeze(0)  # (1,W)
        # normalize to [0, 2pi] as in DETR
        eps = 1e-6
        y_embed = y_embed / (h + eps) * 2 * math.pi
        x_embed = x_embed / (w + eps) * 2 * math.pi

        dim_t = torch.arange(self.half_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.half_dim)

        pos_y = y_embed.unsqueeze(-1) / dim_t  # (H,1,half)
        pos_x = x_embed.unsqueeze(-1) / dim_t  # (1,W,half)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = pos_y.expand(h, w, self.half_dim)
        pos_x = pos_x.expand(h, w, self.half_dim)
        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H,W,dim)
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)  # (B,dim,H,W)
        return pos


class MultiheadAttention(nn.Module):
    """Multi-head attention from scratch.

    Accepts separate ``query``/``key``/``value`` tensors of shape ``(seq, batch, dim)``
    so it serves both self-attention (q=k=v) and cross-attention. Uses
    ``scaled_dot_product_attention`` for the core softmax(QK^T/sqrt(d))V — this is the
    exact op we later profile and replace with a custom fused CUDA kernel.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _shape(self, x: Tensor, seq: int, batch: int) -> Tensor:
        # (seq, batch, dim) -> (batch, n_heads, seq, head_dim)
        return x.view(seq, batch, self.n_heads, self.head_dim).permute(1, 2, 0, 3)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        tgt_len, batch, _ = query.shape
        src_len = key.shape[0]

        q = self._shape(self.q_proj(query), tgt_len, batch)
        k = self._shape(self.k_proj(key), src_len, batch)
        v = self._shape(self.v_proj(value), src_len, batch)

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )  # (batch, n_heads, tgt_len, head_dim)

        attn = attn.permute(2, 0, 1, 3).reshape(tgt_len, batch, self.dim)
        return self.out_proj(attn)


def _add_pos(x: Tensor, pos: Optional[Tensor]) -> Tensor:
    return x if pos is None else x + pos


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "relu",
        pre_norm: bool = False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = MultiheadAttention(dim, n_heads, dropout)

        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation(activation)

    def forward(self, src: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        if self.pre_norm:
            x = self.norm1(src)
            q = k = _add_pos(x, pos)
            src = src + self.dropout1(self.self_attn(q, k, x))
            x = self.norm2(src)
            src = src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
            return src
        # post-norm (DETR / ACT default)
        q = k = _add_pos(src, pos)
        src = _norm_residual(self.norm1, src, self.dropout1(self.self_attn(q, k, src)))
        src = _norm_residual(
            self.norm2, src, self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src)))))
        )
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, layer: TransformerEncoderLayer, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = _clone(layer, num_layers)
        self.norm = norm

    def forward(self, src: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        out = src
        for layer in self.layers:
            out = layer(out, pos=pos)
        if self.norm is not None:
            out = self.norm(out)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "relu",
        pre_norm: bool = False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = MultiheadAttention(dim, n_heads, dropout)
        self.cross_attn = MultiheadAttention(dim, n_heads, dropout)

        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation(activation)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        if self.pre_norm:
            x = self.norm1(tgt)
            q = k = _add_pos(x, query_pos)
            tgt = tgt + self.dropout1(self.self_attn(q, k, x))
            x = self.norm2(tgt)
            tgt = tgt + self.dropout2(
                self.cross_attn(_add_pos(x, query_pos), _add_pos(memory, pos), memory)
            )
            x = self.norm3(tgt)
            tgt = tgt + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x)))))
            return tgt
        # post-norm (DETR / ACT default)
        q = k = _add_pos(tgt, query_pos)
        tgt = _norm_residual(self.norm1, tgt, self.dropout1(self.self_attn(q, k, tgt)))
        tgt = _norm_residual(
            self.norm2,
            tgt,
            self.dropout2(self.cross_attn(_add_pos(tgt, query_pos), _add_pos(memory, pos), memory)),
        )
        tgt = _norm_residual(
            self.norm3, tgt, self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt)))))
        )
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, layer: TransformerDecoderLayer, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = _clone(layer, num_layers)
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, pos=pos, query_pos=query_pos)
        if self.norm is not None:
            out = self.norm(out)
        return out


def _get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"unsupported activation: {name}")
