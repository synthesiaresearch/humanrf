from __future__ import annotations

from dataclasses import dataclass
from typing import List

import nerfacc
import torch

from actorshq.dataset.input_batch import InputBatch
from humanrf.scene_representation.humanrf import HumanRF
from humanrf.scene_representation.query_io import QueryInput


@dataclass
class RenderOutput:
    """
    Contains per-ray properties that are calculated by accumulating volumetric terms along each ray.
    e.g., accumulating per-sample radiance to get the rendered color of a ray.
    """

    # (#rays × 3): [torch.float]
    color: torch.Tensor = None
    # (#rays × 1): [torch.float]
    weights_sum: torch.Tensor = None

    @classmethod
    @torch.no_grad()
    def merge_render_outputs(cls, render_outputs: List[RenderOutput]) -> RenderOutput:
        final_render_output = RenderOutput()
        for key, val in vars(render_outputs[0]).items():
            if val is None:
                setval = None
            if isinstance(val, torch.Tensor):
                setval = torch.cat([getattr(rout, key) for rout in render_outputs], dim=0)
            else:
                raise RuntimeError("Unknown data type in the input_batches!")
            setattr(final_render_output, key, setval)

        return final_render_output


@torch.no_grad()
def prune_samples(
    input_batch: InputBatch,
    scene_representation: HumanRF,
    is_training: bool,
    render_step_size: float = 4e-4,
) -> None:
    """Prunes the samples in place. Pruning means discarding the samples which have very little impact on the final
    results to make training faster. The criteria of impact is assessed via thresholding on the opacity and transmittance
    values such that the samples with very little weights are discarded (see usage of nerfacc.render_visibility).

    Args:
        input_batch (InputBatch):
            The input from the dataloader.
        scene_representation (BaseSceneRepresentation):
            The representation that provides the density() method.
        is_training (bool):
            Whether it's training or not.
        render_step_size (float, optional):
            Raymarching step size. Defaults to 4e-4.
    """
    if is_training:
        input_batch.sample_distances += torch.rand_like(input_batch.sample_distances) * render_step_size

    query_input = QueryInput(
        is_training=is_training,
        positions=input_batch.ray_origins[input_batch.ray_indices]
        + input_batch.sample_distances * input_batch.ray_directions[input_batch.ray_indices],
        frame_numbers=input_batch.frame_numbers[input_batch.ray_indices],
        unique_frame_numbers=input_batch.unique_frame_numbers,
    )
    density = scene_representation.density(query_input).density.unsqueeze(-1)

    visibility_mask = nerfacc.render_visibility(
        alphas=1.0 - torch.exp(-density * render_step_size),
        ray_indices=input_batch.ray_indices,
        early_stop_eps=1e-4,
        alpha_thre=1e-4,
        n_rays=input_batch.num_rays,
    )

    input_batch.sample_distances = input_batch.sample_distances[visibility_mask]
    input_batch.ray_indices = input_batch.ray_indices[visibility_mask]


def render(
    input_batch: InputBatch,
    scene_representation: HumanRF,
    background_rgb: torch.Tensor,
    is_training: bool,
    render_step_size: float = 4e-4,
) -> RenderOutput:
    """First, it computes weights for each sample and perform rendering to produce per-ray outputs using these weights
    (e.g., per-ray color values).

    Args:
        input_batch (InputBatch):
            The input from the dataloader.
        scene_representation (BaseSceneRepresentation):
            The representation that provides the density() and forward() methods.
        background_rgb (torch.Tensor):
            The rendered color is blended with the 'background_rgb' using 'weights_sum'.
        is_training (bool):
            Whether it's training or not.
        render_step_size (float, optional):
            Raymarching step size. Defaults to 4e-4.
    """
    ib = input_batch  # For the sake of readibility.
    samples_ray_directions = ib.ray_directions[ib.ray_indices]

    query_input = QueryInput(
        is_training=is_training,
        positions=ib.ray_origins[ib.ray_indices] + ib.sample_distances * samples_ray_directions,
        directions=samples_ray_directions,
        frame_numbers=ib.frame_numbers[ib.ray_indices],
        unique_frame_numbers=ib.unique_frame_numbers,
        camera_numbers=ib.camera_numbers[ib.ray_indices],
    )
    queried_props = scene_representation(query_input)
    density, radiance = queried_props.density, queried_props.radiance

    weights = nerfacc.render_weight_from_density(
        t_starts=ib.sample_distances,
        t_ends=ib.sample_distances + render_step_size,
        sigmas=density.unsqueeze(-1),
        ray_indices=ib.ray_indices,
        n_rays=ib.num_rays,
    )
    pred_rgb = nerfacc.accumulate_along_rays(
        weights,
        ib.ray_indices,
        values=radiance,
        n_rays=ib.num_rays,
    )
    weights_sum = nerfacc.accumulate_along_rays(
        weights,
        ib.ray_indices,
        values=None,
        n_rays=ib.num_rays,
    )

    # Background composition.
    if background_rgb is not None:
        pred_rgb = pred_rgb + background_rgb * (1.0 - weights_sum)

    return RenderOutput(
        color=pred_rgb,
        weights_sum=weights_sum,
    )
