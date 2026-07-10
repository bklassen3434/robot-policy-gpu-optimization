"""ACT (Action Chunking Transformer) policy, assembled from scratch.

Faithful to the LeRobot / original ACT design so trained accuracy is comparable:

- A CVAE encoder (used only in training) compresses ``[CLS, state, action_chunk]``
  into a latent ``z``; at inference ``z = 0``.
- A transformer encoder attends over ``[latent, state, image_tokens...]``.
- A transformer decoder with learned query embeddings cross-attends to that memory
  and emits a chunk of ``chunk_size`` actions in one shot.
- Loss = L1(action) + kl_weight * KL(latent).

The transformer internals live in ``transformer.py`` (from scratch). The ResNet-18
image backbone is reused from torchvision (``backbone.py``), matching official ACT.

Inputs/outputs are assumed already normalized (see ``robopolicy.data``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import ResNet18Backbone
from .transformer import (
    SinusoidalPositionEmbedding2D,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    sinusoidal_pos_embedding_1d,
)


@dataclass
class ACTConfig:
    # dims (filled from dataset metadata)
    state_dim: int = 14
    action_dim: int = 14
    n_cameras: int = 1

    # transformer
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    dropout: float = 0.1
    feedforward_activation: str = "relu"
    pre_norm: bool = False

    # CVAE
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0

    # action chunking
    chunk_size: int = 100
    n_action_steps: int = 100

    # vision
    vision_backbone: str = "resnet18"
    pretrained_backbone: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ACTConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim_model

        # ---- CVAE encoder (training only) ----
        if cfg.use_vae:
            self.vae_cls_embed = nn.Embedding(1, dim)
            self.vae_state_proj = nn.Linear(cfg.state_dim, dim)
            self.vae_action_proj = nn.Linear(cfg.action_dim, dim)
            self.vae_latent_out_proj = nn.Linear(dim, cfg.latent_dim * 2)
            vae_layer = TransformerEncoderLayer(
                dim, cfg.n_heads, cfg.dim_feedforward, cfg.dropout,
                cfg.feedforward_activation, cfg.pre_norm,
            )
            self.vae_encoder = TransformerEncoder(vae_layer, cfg.n_vae_encoder_layers)
            # fixed sinusoidal pos for [cls, state, action_1..action_k]
            vae_seq = cfg.chunk_size + 2
            self.register_buffer(
                "vae_pos_embed",
                sinusoidal_pos_embedding_1d(vae_seq, dim).unsqueeze(1),  # (seq,1,dim)
                persistent=False,
            )

        # projection from latent to a token (z=0 at inference)
        self.latent_proj = nn.Linear(cfg.latent_dim, dim)

        # ---- image backbone + projection ----
        self.backbone = ResNet18Backbone(pretrained=cfg.pretrained_backbone)
        self.img_proj = nn.Conv2d(self.backbone.out_channels, dim, kernel_size=1)
        self.pos2d = SinusoidalPositionEmbedding2D(dim)

        # ---- main encoder ----
        self.state_proj = nn.Linear(cfg.state_dim, dim)
        # learned pos embed for the two 1D tokens (latent, state)
        self.encoder_1d_pos = nn.Embedding(2, dim)
        enc_layer = TransformerEncoderLayer(
            dim, cfg.n_heads, cfg.dim_feedforward, cfg.dropout,
            cfg.feedforward_activation, cfg.pre_norm,
        )
        self.encoder = TransformerEncoder(enc_layer, cfg.n_encoder_layers)

        # ---- decoder ----
        self.decoder_query_pos = nn.Embedding(cfg.chunk_size, dim)
        dec_layer = TransformerDecoderLayer(
            dim, cfg.n_heads, cfg.dim_feedforward, cfg.dropout,
            cfg.feedforward_activation, cfg.pre_norm,
        )
        self.decoder = TransformerDecoder(dec_layer, cfg.n_decoder_layers, norm=nn.LayerNorm(dim))

        self.action_head = nn.Linear(dim, cfg.action_dim)

        self._reset_parameters()
        # inference action buffer
        self._action_queue: list[Tensor] = []

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    def _encode_latent(self, state: Tensor, action: Optional[Tensor], pad: Optional[Tensor]):
        """Run the CVAE encoder. Returns (latent, mu, logvar)."""
        cfg = self.cfg
        b = state.shape[0]
        device = state.device
        if not cfg.use_vae or action is None:
            latent = torch.zeros(b, cfg.latent_dim, device=device)
            return latent, None, None

        cls = self.vae_cls_embed.weight.unsqueeze(0).expand(b, 1, -1)  # (B,1,dim)
        state_tok = self.vae_state_proj(state).unsqueeze(1)  # (B,1,dim)
        action_tok = self.vae_action_proj(action)  # (B,chunk,dim)
        vae_in = torch.cat([cls, state_tok, action_tok], dim=1).permute(1, 0, 2)  # (seq,B,dim)

        out = self.vae_encoder(vae_in, pos=self.vae_pos_embed.to(vae_in.dtype))
        cls_out = out[0]  # (B,dim)
        params = self.vae_latent_out_proj(cls_out)
        mu, logvar = params[:, : cfg.latent_dim], params[:, cfg.latent_dim :]
        std = torch.exp(0.5 * logvar)
        latent = mu + std * torch.randn_like(std)
        return latent, mu, logvar

    def _encode_observation(self, latent: Tensor, state: Tensor, images: Tensor):
        """Build encoder memory + its position embeddings. Returns (memory, pos)."""
        b = state.shape[0]
        latent_tok = self.latent_proj(latent).unsqueeze(0)  # (1,B,dim)
        state_tok = self.state_proj(state).unsqueeze(0)  # (1,B,dim)

        img_tokens, img_pos = [], []
        n_cam = images.shape[1]
        for cam in range(n_cam):
            feat = self.backbone(images[:, cam])  # (B,512,h,w)
            feat = self.img_proj(feat)  # (B,dim,h,w)
            pos = self.pos2d(feat).to(feat.dtype)  # (B,dim,h,w)
            img_tokens.append(feat.flatten(2).permute(2, 0, 1))  # (h*w,B,dim)
            img_pos.append(pos.flatten(2).permute(2, 0, 1))

        tokens = torch.cat([latent_tok, state_tok, *img_tokens], dim=0)  # (S,B,dim)
        pos_1d = self.encoder_1d_pos.weight.unsqueeze(1).expand(-1, b, -1)  # (2,B,dim)
        pos = torch.cat([pos_1d, *img_pos], dim=0)
        memory = self.encoder(tokens, pos=pos)
        return memory, pos

    def forward(self, batch: dict) -> tuple[Tensor, tuple[Optional[Tensor], Optional[Tensor]]]:
        """Returns (actions_pred (B,chunk,action_dim), (mu, logvar))."""
        state = batch["observation.state"]
        images = batch["observation.images"]  # (B, n_cam, 3, H, W)
        action = batch.get("action")
        pad = batch.get("action_is_pad")

        latent, mu, logvar = self._encode_latent(state, action, pad)
        memory, pos = self._encode_observation(latent, state, images)

        b = state.shape[0]
        query_pos = self.decoder_query_pos.weight.unsqueeze(1).expand(-1, b, -1)  # (chunk,B,dim)
        tgt = torch.zeros_like(query_pos)
        dec = self.decoder(tgt, memory, pos=pos, query_pos=query_pos)  # (chunk,B,dim)
        actions = self.action_head(dec.permute(1, 0, 2))  # (B,chunk,action_dim)
        return actions, (mu, logvar)

    # ------------------------------------------------------------------
    def compute_loss(self, batch: dict) -> dict:
        """L1 action loss + KL. ``batch`` must contain the target ``action``."""
        target = batch["action"]  # (B,chunk,action_dim)
        pad = batch.get("action_is_pad")
        pred, (mu, logvar) = self(batch)

        l1 = F.l1_loss(pred, target, reduction="none")  # (B,chunk,action_dim)
        if pad is not None:
            keep = (~pad).unsqueeze(-1)  # (B,chunk,1)
            l1 = (l1 * keep).sum() / keep.sum().clamp(min=1) / target.shape[-1]
        else:
            l1 = l1.mean()

        out = {"l1": l1, "loss": l1}
        if mu is not None:
            kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
            out["kld"] = kld
            out["loss"] = l1 + self.cfg.kl_weight * kld
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, observation: dict) -> Tensor:
        """Return a single action ``(B, action_dim)`` for rollout.

        Runs the policy once per ``n_action_steps`` and serves actions from a queue,
        which is the standard ACT open-loop chunk execution.
        """
        if not self._action_queue:
            batch = {
                "observation.state": observation["observation.state"],
                "observation.images": observation["observation.images"],
            }
            actions, _ = self(batch)  # (B,chunk,action_dim), latent defaults to 0
            actions = actions[:, : self.cfg.n_action_steps]  # (B,n,action_dim)
            self._action_queue = [actions[:, i] for i in range(actions.shape[1])]
        return self._action_queue.pop(0)

    def reset(self) -> None:
        self._action_queue = []
