"""
QuaDMix Proxy Model Architecture (~1M non-embedding parameters)

Aligned with RegMix (Liu et al., 2024, ICLR 2025, arXiv:2407.01492)
  tinyllama_1M config:
    n_layer=2, n_head=8, n_embd=256, block_size=2048
    vocab_size=50432 (50277 padded to 64)
    LLaMAMLP (SwiGLU), RMSNorm, RoPE (100%)
    bias=False, intermediate_size=512
    Tied embedding + LM head
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm (used in LLaMA architecture)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) — full 100% rotary with pre-computed cos/sin."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

        # Pre-compute cos/sin for max_seq_len (block_size fixed during training)
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())  # [max_seq_len, dim]
        self.register_buffer("sin_cached", emb.sin())  # [max_seq_len, dim]

    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor):
    cos = cos[:q.shape[-2], :]
    sin = sin[:q.shape[-2], :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    """Standard MHA with RoPE (100% rotary), no bias."""

    def __init__(self, config: "ProxyConfig"):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.rotary = RotaryEmbedding(
            dim=self.head_dim, max_seq_len=config.block_size, base=config.rope_base,
        )

        causal_mask = torch.triu(
            torch.ones(config.block_size, config.block_size), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * math.sqrt(self.head_dim) ** -1
        attn = attn.masked_fill(
            self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class LLaMAMLP(nn.Module):
    """LLaMA-style SwiGLU MLP."""

    def __init__(self, config: "ProxyConfig"):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-RMSNorm transformer decoder block with LLaMA MLP."""

    def __init__(self, config: "ProxyConfig"):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = LLaMAMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class ProxyConfig:
    """
    Matches RegMix tinyllama_1M exactly.
    """

    def __init__(
        self,
        n_layer: int = 2,
        n_head: int = 8,
        n_embd: int = 256,
        vocab_size: int = 50432,
        padding_multiple: int = 64,
        block_size: int = 2048,
        bias: bool = False,
        norm_eps: float = 1e-5,
        rope_base: int = 10000,
        intermediate_size: int = 512,
    ):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.padding_multiple = padding_multiple
        self.block_size = block_size
        self.bias = bias
        self.norm_eps = norm_eps
        self.rope_base = rope_base
        self.intermediate_size = intermediate_size

    @classmethod
    def from_name(cls, name: str, block_size: Optional[int] = None) -> "ProxyConfig":
        """Get config for a named variant, optionally override block_size."""
        variants = {
            "tinyllama_1M": cls(),
            "tinyllama_5M": cls(n_layer=4, n_head=8, n_embd=384, intermediate_size=1024),
            "tinyllama_20M": cls(n_layer=6, n_head=12, n_embd=512, intermediate_size=1536),
        }
        if name not in variants:
            raise ValueError(f"Unknown variant {name}. Options: {list(variants.keys())}")
        config = variants[name]
        if block_size is not None:
            config.block_size = block_size
        return config


class ProxyModel(nn.Module):
    """
    GPT-style decoder matching RegMix tinyllama_1M:
      - 2 layers, 8 heads, 256 hidden, 50432 vocab
      - LLaMAMLP (SwiGLU), RMSNorm, RoPE (100%)
      - bias=False, intermediate_size=512
      - Tied embedding + LM head weights
    """

    def __init__(self, config: ProxyConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.n_embd)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False):
        B, T = input_ids.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )

        x = self.embed(input_ids)  # (B, T, C)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        if return_hidden:
            return x
        logits = self.lm_head(x)
        return logits

    def count_params(self, non_embedding_only: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding_only:
            total -= self.embed.weight.numel()
        return total
