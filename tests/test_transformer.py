"""Unit tests for the from-scratch transformer blocks — run anywhere (CPU/MPS)."""

from __future__ import annotations

import torch

from robopolicy.model.transformer import (
    MultiheadAttention,
    SinusoidalPositionEmbedding2D,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    sinusoidal_pos_embedding_1d,
)


def test_mha_shape_and_selfattention():
    dim, heads, seq, batch = 64, 8, 10, 4
    mha = MultiheadAttention(dim, heads)
    x = torch.randn(seq, batch, dim)
    out = mha(x, x, x)
    assert out.shape == (seq, batch, dim)


def test_mha_cross_attention_shapes():
    dim, heads = 64, 8
    mha = MultiheadAttention(dim, heads)
    q = torch.randn(5, 2, dim)
    kv = torch.randn(9, 2, dim)
    out = mha(q, kv, kv)
    assert out.shape == (5, 2, dim)


def test_mha_matches_reference_math():
    # single head, no projections-as-identity: compare against manual softmax attention
    torch.manual_seed(0)
    dim, seq, batch = 16, 6, 2
    mha = MultiheadAttention(dim, n_heads=1, dropout=0.0).eval()
    q = torch.randn(seq, batch, dim)
    k = torch.randn(seq, batch, dim)
    v = torch.randn(seq, batch, dim)
    out = mha(q, k, v)

    # replicate: project, scaled dot product, out_proj
    qp = mha.q_proj(q).transpose(0, 1)  # (B,S,dim)
    kp = mha.k_proj(k).transpose(0, 1)
    vp = mha.v_proj(v).transpose(0, 1)
    scores = qp @ kp.transpose(1, 2) / (dim ** 0.5)
    attn = torch.softmax(scores, dim=-1) @ vp
    expected = mha.out_proj(attn).transpose(0, 1)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_encoder_decoder_shapes():
    dim, heads, ff = 64, 8, 128
    enc = TransformerEncoder(TransformerEncoderLayer(dim, heads, ff, 0.0), 3)
    dec = TransformerDecoder(TransformerDecoderLayer(dim, heads, ff, 0.0), 2)
    src = torch.randn(20, 4, dim)
    pos = torch.randn(20, 4, dim)
    memory = enc(src, pos=pos)
    assert memory.shape == src.shape

    query_pos = torch.randn(7, 4, dim)
    tgt = torch.zeros_like(query_pos)
    out = dec(tgt, memory, pos=pos, query_pos=query_pos)
    assert out.shape == (7, 4, dim)


def test_pos_embeddings():
    pe = sinusoidal_pos_embedding_1d(50, 64)
    assert pe.shape == (50, 64)
    assert torch.isfinite(pe).all()

    pe2d = SinusoidalPositionEmbedding2D(64)
    feat = torch.randn(2, 64, 5, 7)
    out = pe2d(feat)
    assert out.shape == (2, 64, 5, 7)
    assert torch.isfinite(out).all()
