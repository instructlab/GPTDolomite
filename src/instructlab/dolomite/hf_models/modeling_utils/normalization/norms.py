# ----------------------------------------------------------------
# Extracted from https://github.com/ibm-granite/dolomite-engine
# ----------------------------------------------------------------

# Standard
import numbers

# Third Party
import torch

# ---------------- LayerNorm ---------------

_LAYERNORM_MODULES = {
    "torch": torch.nn.LayerNorm,
}


def get_layernorm(
    normalized_shape: int,
    eps: float,
    normalization_implementation: str = "torch",
) -> torch.nn.LayerNorm:
    if normalization_implementation in _LAYERNORM_MODULES:
        return _LAYERNORM_MODULES[normalization_implementation](
            normalized_shape=normalized_shape, eps=eps
        )

    raise ValueError(
        f"unexpected `normalization_implementation` {normalization_implementation}"
    )


# --------------- RMS Norm ---------------
# ----------------------------------------------------------------
# Extracted from https://github.com/ibm-granite/dolomite-engine
# ----------------------------------------------------------------


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype

        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)

        return self.weight * input.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


_RMSNORM_MODULES = {"torch": RMSNorm}


def get_rmsnorm(
    normalized_shape: int,
    eps: float,
    normalization_implementation: str = "torch",
) -> torch.nn.LayerNorm:
    if normalization_implementation in _RMSNORM_MODULES:
        return _RMSNORM_MODULES[normalization_implementation](
            normalized_shape=normalized_shape, eps=eps
        )

    raise ValueError(
        f"unexpected `normalization_implementation` {normalization_implementation}"
    )
