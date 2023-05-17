from typing import List

import numpy as np
from tqdm import tqdm

from actorshq.dataset.volumetric_dataset import VolumetricDataset

PREDEFINED_SEGMENT_SIZES = [6, 12, 25, 50, 100]


class Cluster:
    def __init__(self):
        self.grid = None
        self.frame_numbers = []

    # Equation (2)
    def add_grid(self, grid: np.ndarray) -> None:
        if self.grid is None:
            self.grid = grid
        else:
            self.grid[grid == 255] = 255


# Equation (3)
def get_total_occupancy(grid: np.ndarray):
    return (grid == 255).sum()


def get_segment_size(num_frames: int):
    for idx, segment_size in enumerate(PREDEFINED_SEGMENT_SIZES[:-1]):
        if num_frames < PREDEFINED_SEGMENT_SIZES[idx + 1]:
            return segment_size

    return PREDEFINED_SEGMENT_SIZES[-1]


def get_final_segment_size(num_frames_left: int):
    for segment_size in PREDEFINED_SEGMENT_SIZES:
        if num_frames_left <= segment_size:
            return segment_size


def compute_adaptive_segment_sizes(
    dataset: VolumetricDataset,
    sorted_frame_numbers: List[int],
    expansion_factor_threshold: float = 1.25,
) -> List[int]:
    """HumanRF's adaptive temporal partitioning algorithm. The core idea is to calculate how the number of occupied
    voxels change through time and spawn a new segment when the change is above a certain threshold.

    Args:
        dataset (VolumetricDataset):
            The volumetric dataset used to retrieve occupancy grids.
        sorted_frame_numbers (List[int]):
            The frame numbers for which the partitioning is performed.
        expansion_factor_threshold (float, optional):
            Threshold on the expansion factor. Larger values lead to larger segment sizes on average. Defaults to 1.25.

    Returns:
        List[int]: Partitioned segment sizes.
    """
    min_segment_size = min(PREDEFINED_SEGMENT_SIZES)
    max_segment_size = max(PREDEFINED_SEGMENT_SIZES)

    current_cluster = Cluster()
    segment_sizes = []

    fnum_idx = 0
    total_num_frames = len(sorted_frame_numbers)
    total_num_frames_decided = 0
    pbar = tqdm(
        total=total_num_frames,
        desc=f"Running adaptive temporal partitioning with threshold {expansion_factor_threshold}",
    )
    while fnum_idx < total_num_frames:
        frame_number = sorted_frame_numbers[fnum_idx]

        grid = dataset.get_occupancy_grid(frame_number=frame_number)
        if len(current_cluster.frame_numbers) == 0:
            initial_occupancy = get_total_occupancy(grid)

        # Equation (2)
        current_cluster.add_grid(grid)
        current_cluster.frame_numbers.append(frame_number)

        current_num_frames = len(current_cluster.frame_numbers)
        if current_num_frames >= min_segment_size:
            # Equation (4)
            expansion_factor = get_total_occupancy(current_cluster.grid) / initial_occupancy
            if expansion_factor > expansion_factor_threshold or current_num_frames >= max_segment_size:
                segment_size = get_segment_size(current_num_frames)
                total_num_frames_decided += segment_size
                pbar.update(segment_size)
                current_cluster = Cluster()
                fnum_idx = total_num_frames_decided
                segment_sizes.append(segment_size)
                continue
        fnum_idx += 1

    if total_num_frames_decided < total_num_frames:
        segment_sizes.append(get_final_segment_size(total_num_frames - total_num_frames_decided))
        pbar.update(segment_sizes[-1])

    pbar.close()

    assert sum(segment_sizes) >= total_num_frames
    return segment_sizes
