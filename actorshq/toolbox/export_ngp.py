#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from actorshq.dataset.camera_data import CameraData
from actorshq.dataset.volumetric_dataset import VolumetricDataset


def export_as_ngp(
    cameras: List[CameraData],
    output_folder: Path,
    image_folder: Path,
    scene_offset: np.array,
    scene_scale: float,
) -> None:
    frames = []

    to_ngp_camera = R.from_euler("x", [180], degrees=True).as_matrix()
    to_ngp_world = R.from_euler("xz", [90, 90], degrees=True).as_matrix()
    image_paths = sorted(list(image_folder.glob("*")))
    for camera_idx, camera in enumerate(cameras):
        matrix = np.eye(4)
        matrix[:3, :3] = to_ngp_world @ camera.rotation_matrix_cam2world() @ to_ngp_camera
        matrix[:3, 3] = to_ngp_world @ ((camera.translation + scene_offset) * scene_scale)

        # For camera_angle_x, camera_angle_y see:
        # https://github.com/NVlabs/instant-ngp/blob/1dc8eb6318e47407c8296b0d9549c602280f39be/scripts/colmap2nerf.py#L216
        frames.append(
            {
                "file_path": str(os.path.relpath(image_paths[camera_idx], output_folder)),
                "camera_name": camera.name,
                "transform_matrix": [list(v) for v in list(matrix)],
            }
        )

        output = {
            "cx": camera.cx_pixel,
            "cy": camera.cy_pixel,
            "w": camera.width,
            "h": camera.height,
            "aabb_scale": 1,
            "frames": frames,
            "fl_x": camera.fx_pixel,
            "fl_y": camera.fy_pixel,
            "camera_angle_x": 2.0 * math.atan2(0.5 * camera.width, camera.fx_pixel),
            "camera_angle_y": 2.0 * math.atan2(0.5 * camera.height, camera.fy_pixel),
            "p1": 0.0,  # These are optional
            "p2": 0.0,  # These are optional
            "k1": 0.0,  # These are optional
            "k2": 0.0,  # These are optional
        }

        output_json_path = output_folder / f"transforms{camera_idx:03d}.json"
        output_json_path.write_text(json.dumps(output, indent=2), encoding="UTF-8")
        frames = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=Path, required=True)
    parser.add_argument("--frame_number", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = VolumetricDataset(args.data_folder)

    # The NGP scales the loaded scene by 0.33, hence we scale the scene to ~95% * 3.
    # see: https://github.com/NVlabs/instant-ngp/blob/1dc8eb6318e47407c8296b0d9549c602280f39be/include/neural-graphics-primitives/nerf_loader.h#L28
    # and: https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md#existing-datasets
    aabb = dataset.get_aabb([args.frame_number])
    scene_scale = 0.95 * (3 / np.max(aabb[1] - aabb[0]))
    scene_offset = -aabb.mean(0)

    available_camera_numbers, available_frame_numbers = dataset.get_available_cameras_and_frames()
    if args.frame_number not in available_frame_numbers:
        raise RuntimeError("Requested frame number does not exist in the dataset!")

    image_folder = args.output_dir / "images"
    image_folder.mkdir(parents=True, exist_ok=True)
    for camera_number in available_camera_numbers:
        rgb = dataset.get_rgb(camera_number, args.frame_number)
        mask = dataset.get_mask(camera_number, args.frame_number)
        rgb *= mask
        frame_png = np.concatenate((rgb, mask), axis=-1)
        cv2.imwrite(str(image_folder / f"{dataset.cameras[camera_number].name}.png"), frame_png * 255)

    export_as_ngp(
        cameras=[dataset.cameras[i] for i in available_camera_numbers],
        output_folder=args.output_dir,
        image_folder=image_folder,
        scene_offset=scene_offset,
        scene_scale=scene_scale,
    )


if __name__ == "__main__":
    main()
