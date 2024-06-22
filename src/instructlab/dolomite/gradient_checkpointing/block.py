# ----------------------------------------------------------------
# Extracted from https://github.com/ibm-granite/dolomite-engine
# ----------------------------------------------------------------
# Standard
from functools import partial
from typing import List, Type

# Third Party
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
import torch


# originaly from wrapper.py
# we will move this logic out
def get_module_class_from_name(
    model: torch.nn.Module, name: str
) -> List[Type[torch.nn.Module]]:
    modules_children = list(model.children())

    if model.__class__.__name__ == name:
        return model.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class


def block_checkpointing(
    model: torch.nn.Module,
    block_name: str,
    checkpoint_every: int = 1,
    use_reentrant: bool = False,
) -> None:
    block_class = get_module_class_from_name(model, block_name)
    block_idx = 0

    def _whether_to_checkpoint(submodule: torch.nn.Module) -> bool:
        nonlocal block_idx

        if isinstance(submodule, block_class):
            block_idx += 1
            if (block_idx - 1) % checkpoint_every == 0:
                return True
        return False

    checkpoint_wrapper_function = checkpoint_wrapper
    if use_reentrant:
        checkpoint_wrapper_function = partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT
        )

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper_function,
        check_fn=_whether_to_checkpoint,
    )
