import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class _truncated_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x)
        ctx.threshold = threshold

        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        x = ctx.saved_tensors[0]
        threshold = ctx.threshold

        return dy * torch.exp(x.clamp(-threshold, threshold)), None


def truncated_exp(inp: torch.Tensor, threshold: float = 15) -> torch.Tensor:
    """
    Differentiable torch.exp function with local gradient clipping enabled during backward pass.

    Args:
        inp (torch.Tensor): Input tensor to be exponentiated.
        threshold (float, optional): The input tensor will be clipped to [-threshold, threshold]
                                     during the backward pass. This is to prevent having vanishing
                                     or exploding gradients. Defaults to 15. This default parameter
                                     was taken from torch-ngp, and it has worked well so far. Unless
                                     you have a good reason, do not change this value!

    Returns:
        torch.Tensor: Exponentiated input tensor. Returns the same value as torch.exp(inp).
    """
    return _truncated_exp.apply(inp, threshold)
