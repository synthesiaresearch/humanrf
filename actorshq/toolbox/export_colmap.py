#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

from scipy.spatial.transform import Rotation

from actorshq.dataset.camera_data import CameraData, read_calibration_csv


def export_as_colmap(cameras: List[CameraData], output_folder: Path) -> None:
    camera_lines = ""
    image_lines = ""
    for camera_id, camera in enumerate(cameras):
        world_to_camera = Rotation.from_rotvec(-camera.rotation_axisangle)
        quat = world_to_camera.as_quat()
        tvec = -world_to_camera.as_matrix() @ camera.translation

        fx, fy, cx, cy = camera.fx_pixel, camera.fy_pixel, camera.cx_pixel, camera.cy_pixel
        camera_lines += f"{camera_id} PINHOLE {camera.width} {camera.height} {fx} {fy} {cx} {cy}\n"

        x, y, z, w = tuple(quat)
        tx, ty, tz = tuple(tvec)
        image_lines += f"{camera_id} {w} {x} {y} {z} {tx} {ty} {tz} {camera_id} {camera.name}\n\n"

    # Write intrinsics to cameras.txt
    with open(output_folder / "cameras.txt", "w") as f:
        f.write(camera_lines)

    # Write extrinsics to images.txt
    with open(output_folder / "images.txt", "w") as f:
        f.write(image_lines)

    # For completeness, write an empty points3D.txt file.
    with open(output_folder / "points3D.txt", "w") as f:
        f.write("# Empty file...\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    cameras = read_calibration_csv(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_as_colmap(cameras, args.output_dir)


if __name__ == "__main__":
    main()
