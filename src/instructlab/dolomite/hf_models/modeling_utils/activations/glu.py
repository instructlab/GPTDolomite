# ----------------------------------------------------------------
# Extracted from https://github.com/ibm-granite/dolomite-engine
# ----------------------------------------------------------------
# Third Party
import torch

# Local
from .base import get_base_activation

_GLU_BASE_MAPPING = {
    "ceglu": "celu",
    "eglu": "elu",
    "geglu": "gelu",
    "miglu": "mish",
    "mishglu": "mish",
    "preglu": "prelu",
    "reglu": "relu",
    "rreglu": "rrelu",
    "seglu": "selu",
    "swiglu": "swish",
}


class GLUActivation(torch.nn.Module):
    def __init__(self, base_activation: torch.nn.Module) -> None:
        super().__init__()
        self.base_activation = base_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.chunk(2, dim=-1)
        return x[0] * self.base_activation(x[1])


def get_glu_activation(name: str) -> torch.nn.Module:
    # for glu and sigmoid_glu, we directly return the pytorch's GLU
    if name in ["glu", "sigmoid_glu"]:
        activation_function = torch.nn.modules.GLU()
    else:
        if name in _GLU_BASE_MAPPING:
            name = _GLU_BASE_MAPPING[name]
        elif name.endswith("_glu"):
            name = name.rstrip("_glu")
        else:
            raise ValueError("invalid activation function")

        base_activation = get_base_activation(name)
        activation_function = GLUActivation(base_activation)

    return activation_function


def is_glu(name: str) -> bool:
    return name.endswith("glu")
