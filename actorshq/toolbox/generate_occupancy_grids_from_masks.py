#!/usr/bin/env python3
import argparse
import itertools
import multiprocessing
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm

import actorshq.toolbox.occupancy_grid_generation_native as occupancy_grid_generation_native
from actorshq.dataset.volumetric_dataset import VolumetricDataset


def generate_occupancy_grid_from_masks(data_folder: Path, grid_resolution: int, camera_coverage_threshold: int) -> None:
    """Generates the occupancy grids by carving the empty space using masks. Specifically, each 3D point in the
    occupancy grid is projected onto source cameras provided in the 'data_folder' dataset. If the number of
    projected pixels that correspond to foreground is greater than the provided threshold ('camera_coverage_threshold'),
    the 3D point is assumed to be occupied.

    Args:
        data_folder (Path):
            Source dataset folder for which the occupancy grids are generated.
        grid_resolution (int):
            Resolution of the generated occupancy grids.
        camera_coverage_threshold (int):
            Larger values lead to more conservative occupancy grids. Also, see the description above.
    """
    # This function assumes the scene resides in [-0.5, 0.5] in each canonical axis.
    dataset = VolumetricDataset(data_folder)

    aabb = dataset.get_aabb()
    scene_offset = -aabb.mean(0)
    scene_scale = 1 / np.max(aabb[1] - aabb[0])
    cameras = dataset.get_scaled_cameras(
        scene_offset=scene_offset,
        scene_scale=scene_scale,
    )

    available_camera_numbers, available_frame_numbers = dataset.get_available_cameras_and_frames()
    available_cameras = [cameras[i] for i in available_camera_numbers]
    width = max(available_cameras[0].width, available_cameras[0].height)
    height = min(available_cameras[0].width, available_cameras[0].height)
    num_cameras = len(available_cameras)
    masks = np.empty((num_cameras, width * height), dtype=np.uint8)
    landscape_modes = torch.tensor(
        [cam.width > cam.height for cam in available_cameras],
        device="cuda",
        dtype=torch.bool,
    )

    projection_matrices = (
        torch.from_numpy(
            np.stack([cam.projection_matrix_world2pixel() for cam in available_cameras], axis=0).astype(np.float32)
        )
        .permute(0, 2, 1)
        .to(device="cuda")
        .contiguous()
    )

    # The dilation mask is used to add a margin around the masks rendered from the mesh.
    # This margin ensures that we don't cross the surface while doing ray marching to skip
    # empty space.
    dilation_mask_size = max(width, height) // 128
    dilation_mask = np.ones((dilation_mask_size, dilation_mask_size), np.uint8)
    print(f"Started generating occupancy grids for {str(data_folder)}")
    for frame_number in tqdm.tqdm(
        available_frame_numbers,
        desc=f"Generating occupancy grids with resolution {grid_resolution} and camera coverage threshold {camera_coverage_threshold}/{len(available_cameras)}",
    ):

        def load_mask(buffer_index: int, camera_number: int, frame_number: int) -> np.array:
            mask = dataset.get_mask(camera_number, frame_number, False)
            mask = cv2.dilate(mask.astype(np.uint8), dilation_mask, iterations=1)
            masks[buffer_index] = mask.reshape(-1)

        with ThreadPool(multiprocessing.cpu_count()) as pool:
            pool.starmap(
                load_mask, zip(range(len(available_cameras)), available_camera_numbers, itertools.repeat(frame_number))
            )

        grid = (
            occupancy_grid_generation_native.generate_from_masks(
                torch.from_numpy(masks).to(device="cuda").contiguous(),
                projection_matrices,
                landscape_modes,
                camera_coverage_threshold,
                grid_resolution,
                width,
                height,
            )
            .cpu()
            .numpy()
        )
        output_path = dataset.filepaths.get_occupancy_grid_path(frame_number)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output_path), occupancy_grid=grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=Path, required=True)
    parser.add_argument("--grid_resolution", type=int, required=True)
    parser.add_argument("--camera_coverage_threshold", type=int, required=True)
    args = parser.parse_args()

    generate_occupancy_grid_from_masks(
        data_folder=Path(args.data_folder),
        grid_resolution=args.grid_resolution,
        camera_coverage_threshold=args.camera_coverage_threshold,
    )


if __name__ == "__main__":
    main()
