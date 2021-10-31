from typing import Callable, Iterable, Optional, Tuple, Union
import torch 
from torch.optim import Optimizer

from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup
)

STR_TO_SCHEDULER_FUNCTION = {
    'linear': get_linear_schedule_with_warmup,
    # last_epoch=-1
    'cosine': get_cosine_schedule_with_warmup,
    # num_cycles: float = 0.5, The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0 following a half-cosine).
    # last_epoch: int = -1, The index of the last epoch when resuming training.
    'cosine_with_restarts': get_cosine_with_hard_restarts_schedule_with_warmup, 
    # num_cycles: int = 1, The number of hard restarts to use.
    # last_epoch: int = -1
    'polynomial': get_polynomial_decay_schedule_with_warmup, 
    # lr_end=1e-7, The end LR. 
    # power=1.0, Power factor. `power` defaults to 1.0 as in the fairseq implementation
    # last_epoch=-1
    'constant': get_constant_schedule,
    'constant_with_warmup': get_constant_schedule_with_warmup,
}

def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: Optional[int] = 1,
):
    """
    Unified API to get any scheduler from its name.
    Args:
        name (:obj:`str`:
            The name of the scheduler to use.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (:obj:`int`, `optional`):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (:obj:`int`, `optional`):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """

    schedule_func = STR_TO_SCHEDULER_FUNCTION[name]
    if name == 'constant':
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == 'constant_with_warmup':
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")
    if name == 'cosine_with_restarts':
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=num_cycles)
    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)