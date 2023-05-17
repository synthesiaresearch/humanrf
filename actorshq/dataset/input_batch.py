from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InputBatch:
    """
    Output of sampling a batch from the data loader. Most of the tensors have either per-sample or per-ray data. Among the
    tensors, there are two exceptions to this:
        1. 'unique_frame_numbers' includes the unique entries in 'frame_numbers'.
        2. 'ray_masks' will most likely have more entries than #rays. This is because ray sampler omits some of
           rays that hit the background or masked out using the light annotations. So, 'ray_masks' has the shape
           of the original batch size, and it indicates which rays are omitted (denoted by ray_masks==False).
    """

    # (#rays × 3): [torch.float]
    ray_origins: torch.Tensor = None
    # (#rays × 3): [torch.float]
    ray_directions: torch.Tensor = None
    # (#rays × 2): [torch.float]
    minmaxes: torch.Tensor = None
    # (#rays × 4): [torch.float]
    rgba: torch.Tensor = None
    # (>=#rays × 1): [torch.bool]
    ray_masks: torch.Tensor = None
    # (#rays × 1): [torch.int32]
    frame_numbers: torch.Tensor = None
    # (K × 1): [torch.int32]
    unique_frame_numbers: torch.Tensor = None
    # (#rays × 1): [torch.int32]
    camera_numbers: torch.Tensor = None
    # (#samples × 1): [torch.float]
    sample_distances: torch.Tensor = None
    # (#samples × 1): [torch.int32]
    ray_indices: torch.Tensor = None
    # int
    width: int = None
    # int
    height: int = None

    @property
    def num_rays(self) -> int:
        return self.ray_origins.shape[0]

    @property
    def num_samples(self) -> int:
        return self.sample_distances.shape[0]
