# ----------------------------------------------------------------
# Extracted from https://github.com/ibm-granite/dolomite-engine
# ----------------------------------------------------------------
# Third Party
import torch

# Local
from ..enums import GradientCheckpointingMethod
from .block import block_checkpointing

_GRADIENT_CHECKPOINTING_METHODS = {
    GradientCheckpointingMethod.block: block_checkpointing
}


def apply_gradient_checkpointing(
    model: torch.nn.Module,
    gradient_checkpointing_method: GradientCheckpointingMethod,
    **kwargs,
) -> None:
    checkpointing_function = _GRADIENT_CHECKPOINTING_METHODS[
        gradient_checkpointing_method
    ]
    checkpointing_function(model, **kwargs)
