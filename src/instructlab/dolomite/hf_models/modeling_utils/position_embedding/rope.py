# ----------------------------------------------------------------
# Extracted from https://github.com/ibm-granite/dolomite-engine
# ----------------------------------------------------------------
"""Logic is copied from transformers.models.llama.modeling_utils with slight modifications"""

# Standard
from typing import Tuple

# Third Party
import torch


class RoPE(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.mscale = 1

        self.reset_parameters()

    def forward(
        self, seq_len: int, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        cos = self.cos_cached[:seq_len].to(dtype)
        sin = self.sin_cached[:seq_len].to(dtype)

        return cos, sin

    def reset_parameters(self) -> None:
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    @torch.no_grad()
    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer(
            "cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False
        )


def apply_rotary_pos_emb(
    x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos_sin
    x = (x * cos) + (_rotate_half(x) * sin)
    return x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
