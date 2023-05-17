import gc

import torch


def to_device(obj: object, device: str) -> None:
    """For a given an object, moves any torch.Tensor contained by this object into the specified device.

    Args:
        obj (Any): A Python object
        device (str): Desired torch device. For example, 'cpu' and 'cuda'.
    """
    for key, val in vars(obj).items():
        if isinstance(val, torch.Tensor):
            setattr(obj, key, val.to(device=device, non_blocking=True))


def collect_and_free_memory() -> None:
    """Run the garbage collector and empty the torch cache. This function should be used with care. Currently, it's only
    used when switching between training, validation and test modes.
    """
    gc.collect()
    torch.cuda.empty_cache()
