
from typing import Callable
import torch

def run_rank_n(func: Callable, rank: int = 0, barrier: bool = False) -> Callable:
    """wraps a function to run on a single rank, returns a no-op for other ranks

    Args:
        func (Callable): function to wrap
        rank (int, optional): rank on which function should run. Defaults to 0.
        barrier (bool, optional): whether to synchronize the processes at the end of function execution. Defaults to False.

    Returns:
        Callable: wrapped function
    """

    # wrapper function for the rank to execute on
    def func_rank_n(*args, **kwargs):
        output = func(*args, **kwargs)
        if barrier:
            torch.distributed.barrier()
        return output

    # a dummy method that doesn't do anything
    def func_rank_other(*args, **kwargs):
        if barrier:
            torch.distributed.barrier()

    global_rank = torch.distributed.get_global_rank()

    if global_rank == rank:
        wrapped_func = func_rank_n
    elif global_rank is None:
        # distributed is not initialized
        wrapped_func = func
    else:
        wrapped_func = func_rank_other

    return wrapped_func
