#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from tqdm import tqdm

from actorshq.dataset.volumetric_dataset import VolumetricDataset, VolumetricDatasetFilepaths


def read_yaml(file_path: Path):
    with open(file_path, "r", encoding="UTF-8") as file:
        return yaml.safe_load(file)


def download_lazy(
    source_file: Path,
    target_file: Path,
    verbose: bool = True,
):
    if not target_file.exists():
        response = requests.get(source_file)
        response.raise_for_status()
        if verbose:
            print("Downloading", os.path.basename(urlparse(source_file).path))
        with open(target_file, 'wb') as f:
            f.write(response.content)


def load_and_extract_views(
    file_link: Path,
    target_folder: Path,
    frame_index: int,
):
    tar_name = os.path.basename(urlparse(file_link).path)
    if "rgb" in tar_name:
        type_str = "rgb"
        extension = "jpg"
    else:
        type_str = "mask"
        extension = "png"

    download_lazy(
        file_link,
        target_folder / tar_name,
        verbose=False,
    )
    subprocess.run(
        [
            *("tar", "-xf"),
            os.fspath(target_folder / tar_name),
            *("-C", str(target_folder)),
        ]
    )
    for camera_number in range(1, VolumetricDataset.NUM_CAMERAS + 1):
        cam_name = f"Cam{camera_number:03d}"
        filename = f"{cam_name}_{type_str}{frame_index:06d}.{extension}"
        os.rename(
            target_folder / filename,
            target_folder / cam_name / filename,
        )
    (target_folder / tar_name).unlink()


def download_dataset(
    dataset_file: Path,
    dataset_target: Path,
    actor: str,
    sequence: str,
    scale: int,
    frame_start: int = 0,
    frame_stop: int = 0,
    include_rgb: bool = True,
    include_mask: bool = True,
    include_mesh: bool = False,
    include_lightannotations: bool = True,
) -> Path:
    """Downloads the dataset into the specified folder.

    Args:
        dataset_source (Path):
            The source dataset folder.
        dataset_target (Path):
            The target dataset folder.
        actor (str):
            The actor name.
        sequence (str):
            The sequence name.
        scale (int):
            The downscale factor of the dataset.
        frame_start (int, optional):
            Downloads starting from this frame (inclusive). Defaults to 0.
        frame_stop (int, optional):
            Downloads until this frame. Setting to 0 downloads until the last frame. Defaults to 0.
        include_rgb (bool, optional):
            Whether to download the RGB images. Defaults to True.
        include_mask (bool, optional):
            Whether to download the mask images. Defaults to True.
        include_mesh (bool, optional):
            Whether to download the meshes. Defaults to False.
        include_lightannotations (bool, optional):
            Whether to download the light source annotations. Defaults to True.

    Returns:
        Path: The path to the downloaded dataset.
    """

    if (actor, sequence) in [("Actor03", "Sequence2"), ("Actor07", "Sequence2")]:
        raise RuntimeError(f"{actor}{sequence} is not publicly available!")

    scale_name = f"{scale}x"
    local_sequence_folder = dataset_target / actor / sequence
    local_scale_folder = local_sequence_folder / scale_name
    local_scale_folder.mkdir(exist_ok=True, parents=True)
    dataset_paths = VolumetricDatasetFilepaths(local_scale_folder)
    print("Reading links ...")
    links = read_yaml(dataset_file)

    download_lazy(links[actor][sequence]["scene"], dataset_paths.metadata_path)
    sequence_num_frames = json.loads(dataset_paths.metadata_path.read_text())["num_frames"]

    if frame_stop == 0:
        frame_stop = sequence_num_frames

    for camera_number in range(1, VolumetricDataset.NUM_CAMERAS + 1):
        if include_rgb:
            (local_scale_folder / "rgbs" / f"Cam{camera_number:03d}").mkdir(exist_ok=True, parents=True)
        if include_mask:
            (local_scale_folder / "masks" / f"Cam{camera_number:03d}").mkdir(exist_ok=True, parents=True)

    info_desc = []
    if include_rgb:
        info_desc.append("RGB")
    if include_mask:
        info_desc.append("mask")
    for frame_index in tqdm(range(frame_start, frame_stop), desc=f"Downloading {' and '.join(info_desc)} files ..."):
        if include_rgb:
            # We assume that all views for a frame exist, if camera 1 exists.
            if not dataset_paths.get_rgb_path("Cam001", frame_index).exists():
                load_and_extract_views(
                    links[actor][sequence][scale_name]["rgbs"][f"rgbs_{frame_index:06d}"],
                    local_scale_folder / "rgbs",
                    frame_index,
                )
        if include_mask:
            # We assume that all views for a frame exist, if camera 1 exists.
            if not dataset_paths.get_mask_path("Cam001", frame_index).exists():
                load_and_extract_views(
                    links[actor][sequence][scale_name]["masks"][f"masks_{frame_index:06d}"],
                    local_scale_folder / "masks",
                    frame_index,
                )

    download_lazy(links[actor][sequence][scale_name]["calibration"], dataset_paths.calibration_path)

    # Extract occupancy_grids
    if not dataset_paths.get_occupancy_grid_path(0).exists():
        local_occupancy_tar = local_sequence_folder / "occupancy_grids.tar.gz"
        download_lazy(
            links[actor][sequence]["occupancy_grids"],
            local_occupancy_tar,
        )
        subprocess.run(
            [
                *("tar", "-xzf"),
                os.fspath(local_occupancy_tar),
                "-C",
                str(local_sequence_folder),
            ]
        )
        local_occupancy_tar.unlink()

    if include_mesh:
        download_lazy(
            links[actor][sequence]["meshes"],
            local_sequence_folder / "meshes.abc.xz",
        )
        subprocess.run(
            [
                "xz",
                "-d",
                os.fspath(local_sequence_folder / "meshes.abc.xz"),
            ]
        )

    if include_lightannotations:
        download_lazy(
            links[actor][sequence][scale_name]["light_annotations"],
            dataset_paths.get_light_annotations_path(),
        )

    download_lazy(
        links[actor][sequence]["aabbs"],
        local_sequence_folder / dataset_paths.aabbs_path,
    )

    return local_scale_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--actor", choices=[f"Actor{i:02d}" for i in range(1, 9)], required=True)
    parser.add_argument("--sequence", choices=["Sequence1", "Sequence2"], required=True)
    parser.add_argument("--scale", type=int, choices=[1, 2, 4], default=4, help="Scale to download")
    parser.add_argument("--frame_start", type=int, default=0, help="First frame to download (inclusive)")
    parser.add_argument(
        "--frame_stop", type=int, default=0, help="Last frame to download (exclusive). 0 means all frames."
    )
    parser.add_argument(
        "--include",
        default=["rgb", "mask"],
        choices=["mesh", "rgb", "mask"],
        nargs="*",
        help="Define which data to download",
    )
    args = parser.parse_args()

    download_dataset(
        args.dataset_file,
        args.target,
        args.actor,
        args.sequence,
        args.scale,
        args.frame_start,
        args.frame_stop,
        include_rgb="rgb" in args.include,
        include_mask="mask" in args.include,
        include_mesh="mesh" in args.include,
    )


if __name__ == "__main__":
    main()
