from __future__ import annotations

from typing import List, Optional

import torch

from actorshq.dataset.input_batch import InputBatch


def merge_input_batches(input_batches: List[InputBatch], max_num_samples: Optional[int] = None) -> InputBatch:
    final_input_batch = InputBatch()
    for key, val in vars(input_batches[0]).items():
        if key != "ray_indices":
            if val is None:
                setval = None
            elif isinstance(val, torch.Tensor):
                setval = torch.cat([getattr(inp_batch, key) for inp_batch in input_batches], dim=0)
            elif isinstance(val, int):
                setval = getattr(input_batches[0], key)
            else:
                raise RuntimeError("Unknown data type in the input_batches!")
            setattr(final_input_batch, key, setval)

    if input_batches[0].ray_indices is not None:
        final_ray_indices = [input_batches[0].ray_indices]
        previous_accumulated_batch_size = 0
        for i in range(1, len(input_batches)):
            previous_accumulated_batch_size += input_batches[i - 1].num_rays
            final_ray_indices.append(input_batches[i].ray_indices + previous_accumulated_batch_size)

        final_input_batch.ray_indices = torch.cat(final_ray_indices, dim=0)

    if max_num_samples is not None:
        num_rays = final_input_batch.num_rays
        num_samples = final_input_batch.num_samples
        if num_samples > max_num_samples:
            final_ray_cutoff = final_input_batch.ray_indices[max_num_samples]

            for key, val in vars(final_input_batch).items():
                if isinstance(val, torch.Tensor):
                    if key == "ray_masks":
                        setval = val[val.cumsum(0) < final_ray_cutoff]
                    elif val.shape[0] == num_rays:
                        setval = val[:final_ray_cutoff]
                    elif val.shape[0] == num_samples:
                        setval = val[final_input_batch.ray_indices < final_ray_cutoff]
                    setattr(final_input_batch, key, setval)

    # Recompute the unique frame numbers
    if final_input_batch.frame_numbers is not None:
        final_input_batch.unique_frame_numbers = torch.unique(
            final_input_batch.frame_numbers, sorted=False, return_inverse=False
        ).view(-1, 1)

    return final_input_batch
