from dataclasses import dataclass

import torch


@dataclass
class QueryInput:
    is_training: bool
    positions: torch.Tensor
    directions: torch.Tensor = None
    frame_numbers: torch.Tensor = None
    unique_frame_numbers: torch.Tensor = None
    camera_numbers: torch.Tensor = None


@dataclass
class QueryOutput:
    density: torch.Tensor
    geometry_features: torch.Tensor = None
    radiance: torch.Tensor = None
