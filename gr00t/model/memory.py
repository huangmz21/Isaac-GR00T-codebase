# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action-effect memory modules for GR00T.

This module intentionally stays independent from data loading and DINO feature
extraction. The supported training path expects precomputed memory features and
fuses encoded memory into backbone hidden tokens before the action head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GR00TMemoryFusionConfig:
    """Configuration for memory fusion."""

    vision_dim: int = 768
    action_dim: int = 32
    action_chunk_len: int = 4
    hidden_dim: int = 1536
    mem_dim: int = 768
    fusion: str = "per_token"
    per_token_retrieval_layers: int = 2
    per_token_retrieval_ffn_mult: int = 4
    per_token_retrieval_fusion: str = "gate"
    gate_bias_init: float = 2.0

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "GR00TMemoryFusionConfig":
        if cfg is None:
            return cls()
        valid_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{key: value for key, value in cfg.items() if key in valid_keys})


class ActionEffectMemoryEncoder(nn.Module):
    """Encode action-effect events into retrieval keys and memory tokens."""

    def __init__(self, vision_dim: int, action_dim: int, action_chunk_len: int, mem_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_chunk_len = int(action_chunk_len)

        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_chunk_len * self.action_dim, mem_dim),
            nn.GELU(),
            nn.Linear(mem_dim, mem_dim),
        )

        self.memory_encoder = nn.Sequential(
            nn.Linear(vision_dim + vision_dim + mem_dim, mem_dim),
            nn.GELU(),
            nn.Linear(mem_dim, mem_dim),
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(vision_dim + mem_dim, mem_dim),
            nn.GELU(),
            nn.Linear(mem_dim, mem_dim),
        )

    def forward(
        self,
        z_pre: torch.Tensor,
        z_post: torch.Tensor,
        mem_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mem_actions.ndim != 4:
            raise ValueError(f"Expected mem_actions [B, N, K, D], got {mem_actions.shape}")
        batch_size, num_events, action_chunk_len, action_dim = mem_actions.shape
        if action_chunk_len != self.action_chunk_len or action_dim != self.action_dim:
            raise ValueError(
                "Memory action shape does not match config: "
                f"got K={action_chunk_len}, D={action_dim}; "
                f"expected K={self.action_chunk_len}, D={self.action_dim}"
            )

        visual_effect = z_post - z_pre
        action_flat = mem_actions.reshape(batch_size, num_events, action_chunk_len * action_dim)
        action_emb = self.action_encoder(action_flat)
        mem_token = self.memory_encoder(torch.cat([z_pre, visual_effect, action_emb], dim=-1))
        mem_key = self.key_encoder(torch.cat([z_pre, action_emb], dim=-1))
        return mem_key, mem_token, visual_effect


class GateFusion(nn.Module):
    """Dimension-wise gate between current hidden tokens and retrieved memory."""

    def __init__(self, dim: int, bias_init: float = 2.0):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.proj.bias, bias_init)

    def forward(
        self,
        current: torch.Tensor,
        retrieved: torch.Tensor,
        has_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.proj(torch.cat([current, retrieved], dim=-1)))
        fused = gate * current + (1.0 - gate) * retrieved
        if has_memory is None:
            return fused

        while has_memory.ndim < fused.ndim:
            has_memory = has_memory.unsqueeze(-1)
        return torch.where(has_memory.to(device=fused.device, dtype=torch.bool), fused, current)


class CrossTransformerBlock(nn.Module):
    """Small cross-attention retrieval block over memory events."""

    def __init__(self, dim: int, ffn_mult: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        scores = torch.einsum("bld,bnd->bln", q, k) / math.sqrt(k.shape[-1])

        has_memory = None
        if key_mask is not None:
            memory_mask = key_mask.unsqueeze(1)
            scores = scores.masked_fill(~memory_mask, -1e4)
            weights = torch.softmax(scores, dim=-1) * memory_mask.to(dtype=scores.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            has_memory = key_mask.any(dim=-1)
        else:
            weights = torch.softmax(scores, dim=-1)

        attn_out = torch.einsum("bln,bnd->bld", weights, v)
        if has_memory is not None:
            attn_out = attn_out * has_memory[:, None, None].to(dtype=attn_out.dtype)

        x = self.attn_norm(query + attn_out)
        out = self.ffn_norm(x + self.ffn(x))
        if has_memory is not None:
            return torch.where(has_memory[:, None, None], out, query)
        return out


class GR00TMemoryFusion(nn.Module):
    """Fuse action-effect memory into GR00T backbone hidden tokens."""

    def __init__(self, config: GR00TMemoryFusionConfig | dict | None = None):
        super().__init__()
        if not isinstance(config, GR00TMemoryFusionConfig):
            config = GR00TMemoryFusionConfig.from_dict(config)
        self.config = config

        if config.fusion not in {"global", "per_token"}:
            raise ValueError(f"memory fusion must be 'global' or 'per_token', got {config.fusion!r}")
        if config.per_token_retrieval_fusion not in {"gate", "add"}:
            raise ValueError(
                "per_token_retrieval_fusion must be 'gate' or 'add', "
                f"got {config.per_token_retrieval_fusion!r}"
            )

        self.memory_encoder = ActionEffectMemoryEncoder(
            vision_dim=config.vision_dim,
            action_dim=config.action_dim,
            action_chunk_len=config.action_chunk_len,
            mem_dim=config.mem_dim,
        )
        self.hidden_to_memory = nn.Linear(config.hidden_dim, config.mem_dim)
        self.memory_to_hidden = nn.Linear(config.mem_dim, config.hidden_dim)

        self.per_token_retrieval_blocks = nn.ModuleList()
        if config.fusion == "per_token":
            self.per_token_retrieval_blocks = nn.ModuleList(
                [
                    CrossTransformerBlock(
                        config.mem_dim, ffn_mult=config.per_token_retrieval_ffn_mult
                    )
                    for _ in range(config.per_token_retrieval_layers)
                ]
            )
        self.per_token_gate_fusion = (
            GateFusion(config.hidden_dim, bias_init=config.gate_bias_init)
            if config.fusion == "per_token" and config.per_token_retrieval_fusion == "gate"
            else None
        )

    def _encode_memory(
        self,
        mem_pre_feat: torch.Tensor,
        mem_post_feat: torch.Tensor,
        mem_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = self.memory_encoder.action_encoder[0].weight.dtype
        mem_pre_feat = mem_pre_feat.to(dtype=dtype)
        mem_post_feat = mem_post_feat.to(dtype=dtype)
        mem_actions = mem_actions.to(dtype=dtype)

        if mem_pre_feat.shape != mem_post_feat.shape:
            raise ValueError(
                f"Pre/post memory feature shape mismatch: {mem_pre_feat.shape} vs {mem_post_feat.shape}"
            )
        if mem_pre_feat.ndim != 3:
            raise ValueError(f"Expected memory features [B, N, D], got {mem_pre_feat.shape}")
        if mem_pre_feat.shape[-1] != self.config.vision_dim:
            raise ValueError(
                f"Memory feature dim {mem_pre_feat.shape[-1]} does not match "
                f"configured vision_dim {self.config.vision_dim}"
            )
        mem_key, mem_token, _ = self.memory_encoder(mem_pre_feat, mem_post_feat, mem_actions)
        return mem_key, mem_token

    def _retrieve_memory_context(
        self,
        hidden: torch.Tensor,
        mem_key: torch.Tensor,
        mem_token: torch.Tensor,
        mem_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        has_memory = mem_valid_mask.any(dim=-1)
        if self.config.fusion == "per_token":
            query = self.hidden_to_memory(hidden.to(mem_key.dtype))
            if len(self.per_token_retrieval_blocks) > 0:
                memory_context = query
                for block in self.per_token_retrieval_blocks:
                    memory_context = block(
                        query=memory_context,
                        key=mem_key,
                        value=mem_token,
                        key_mask=mem_valid_mask,
                    )
            else:
                scores = torch.einsum("bld,bnd->bln", query, mem_key) / math.sqrt(
                    mem_key.shape[-1]
                )
                memory_mask = mem_valid_mask.unsqueeze(1)
                scores = scores.masked_fill(~memory_mask, -1e4)
                weights = torch.softmax(scores, dim=-1) * memory_mask.to(dtype=scores.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                memory_context = torch.einsum("bln,bnd->bld", weights, mem_token)
            memory_context = memory_context * has_memory[:, None, None].to(dtype=memory_context.dtype)
            return memory_context, has_memory

        query = self.hidden_to_memory(hidden.mean(dim=1).to(mem_key.dtype))
        scores = torch.einsum("bd,bnd->bn", query, mem_key) / math.sqrt(mem_key.shape[-1])
        scores = scores.masked_fill(~mem_valid_mask, -1e4)
        weights = torch.softmax(scores, dim=-1) * mem_valid_mask.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        memory_context = torch.sum(weights.unsqueeze(-1) * mem_token, dim=1)
        memory_context = memory_context * has_memory[:, None].to(dtype=memory_context.dtype)
        memory_context = memory_context.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        return memory_context, has_memory

    def forward(
        self,
        hidden: torch.Tensor,
        mem_pre_feat: torch.Tensor,
        mem_post_feat: torch.Tensor,
        mem_actions: torch.Tensor,
        mem_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mem_valid_mask = mem_valid_mask.to(device=hidden.device, dtype=torch.bool)
        mem_key, mem_token = self._encode_memory(
            mem_pre_feat=mem_pre_feat.to(device=hidden.device),
            mem_post_feat=mem_post_feat.to(device=hidden.device),
            mem_actions=mem_actions.to(device=hidden.device),
        )
        memory_context, has_memory = self._retrieve_memory_context(
            hidden=hidden,
            mem_key=mem_key,
            mem_token=mem_token,
            mem_valid_mask=mem_valid_mask,
        )
        retrieved_hidden = self.memory_to_hidden(memory_context)
        retrieved_hidden = retrieved_hidden * has_memory[:, None, None].to(
            dtype=retrieved_hidden.dtype
        )

        if self.per_token_gate_fusion is not None:
            fused = self.per_token_gate_fusion(
                current=hidden.to(dtype=retrieved_hidden.dtype),
                retrieved=retrieved_hidden,
                has_memory=has_memory,
            )
            return fused.to(dtype=hidden.dtype)
        return hidden + retrieved_hidden.to(dtype=hidden.dtype)
