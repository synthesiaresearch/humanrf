import torch


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # NOTE: There is torch.nn.functional.binary_cross_entropy function with the same
    # functionality. However, it gives the following error:
    # '...torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast...'
    # This is why we have this implementation instead.
    pred_clamped = torch.clamp(pred, min=0, max=1)
    return -(target * torch.log(pred_clamped + 1e-10) + (1 - target) * torch.log(1 - pred_clamped + 1e-10))
