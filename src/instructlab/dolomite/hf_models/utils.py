# ----------------------------------------------------------------
# Extracted from https://github.com/ibm-granite/dolomite-engine
# ----------------------------------------------------------------
# Standard
from typing import List, Union

# Third Party
import torch


def check_list_type(
    list_of_list: List[List[Union[int, float]]], error_message: str
) -> None:
    if list_of_list is None:
        return

    assert isinstance(list_of_list, list), error_message
    assert isinstance(list_of_list[0], list), error_message


def flatten_and_convert_to_tensors(x: List[int], device: torch.device) -> torch.Tensor:
    y = []
    for sequence in x:
        y.extend(sequence)

    return torch.tensor(y, device=device)
