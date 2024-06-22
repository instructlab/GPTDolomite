# ----------------------------------------------------------------
# Extracted from https://github.com/ibm-granite/dolomite-engine
# ----------------------------------------------------------------
# Standard
from typing import Tuple

# Third Party
import torch

# Local
from ...config import GPTDolomiteConfig
from ...modeling_utils import get_activation_function, is_glu


class MLP(torch.nn.Module):
    def __init__(self, config: GPTDolomiteConfig) -> None:
        super().__init__()

        hidden_size = config.n_embd
        intermediate_size = config.n_inner
        activation_function = config.activation_function
        add_bias = config.add_bias
        residual_dropout = config.resid_pdrop

        self.c_fc = torch.nn.Linear(
            hidden_size,
            2 * intermediate_size if is_glu(activation_function) else intermediate_size,
            bias=add_bias,
        )

        self.act = get_activation_function(activation_function)

        self.c_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=add_bias)

        self.dropout = (
            torch.nn.Identity()
            if residual_dropout == 0
            else torch.nn.Dropout(residual_dropout)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


def interleave_up_gate_tensor_for_mlp(
    up_weight: torch.Tensor, gate_weight: torch.Tensor
) -> torch.Tensor:
    return torch.cat([up_weight, gate_weight])


def split_up_gate_tensor_for_mlp(
    c_fc_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return c_fc_weight.chunk(2)
