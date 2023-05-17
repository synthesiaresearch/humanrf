import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from actorshq.dataset.aabb_data import AabbData, write_aabbs_csv
from actorshq.dataset.camera_data import CameraData, write_calibration_csv
from actorshq.dataset.volumetric_dataset import VolumetricDataset, VolumetricDatasetFilepaths
from actorshq.toolbox.generate_occupancy_grids_from_masks import generate_occupancy_grid_from_masks

# https://github.com/HaiminLuo/Artemis#dynamic-furry-animals-dfa-dataset
# We use the following parts from DFA to import to ActorsHQ format.
# cat
# ├──  img
# │     └── run                     - Motion name.
# │       └── %d                    - The frame number, start from 0.
# │         └──img_%04d.jpg   		  - RGB images for each view. view number start from 0.
# │         └──img_%04d_alpha.png   - Alpha mattes for corresponding RGB image.
# │     └── ...
# |
# ├──  CamPose.inf				    - Camera extrinsics. In each row, the 3x4 [R T] matrix is displayed in columns, with the third column followed by columns 1, 2, and 4, where R*X^{camera}+T=X^{world}.
# │
# └──  Intrinsic.inf				- Camera intrinsics. The format of each intrinsics is: "idx \n fx 0 cx \n 0 fy cy \n 0 0 1 \n \n" (idx starts from 0)


def import_dfa(dfa_dataset_folder: Path, motion_type: str, output_folder: Path):
    """Converts the input DFA dataset into ActorsHQ format.

    Args:
        dfa_dataset_folder (Path): The input DFA dataset folder, e.g., /path/to/cat
        motion_type (str): Indicates the motion type performed by the animal, e.g., walkprogressive_noz and run.
        output_folder (Path): Output ActorsHQ dataset folder.
    """
    dfa_dataset_path = Path(dfa_dataset_folder)
    images_path = dfa_dataset_path / "img" / motion_type

    output_dataset_filepaths = VolumetricDatasetFilepaths(output_folder)

    frame_dirs = sorted(images_path.glob("*"))
    frame_numbers = [int(fdir.stem) for fdir in frame_dirs]

    # Convert RGB and alpha images into ActorsHQ format.
    for frame_number, fdir in tqdm(
        zip(frame_numbers, frame_dirs),
        total=len(frame_numbers),
        desc="Converting RGB and alpha images into ActorsHQ format",
    ):
        for camera_number in range(36):
            rgb_path = fdir / f"img_{camera_number:04d}.png"
            mask_path = fdir / f"img_{camera_number:04d}_alpha.png"

            camera_name = f"Cam{camera_number:03d}"

            dst_rgb_path = output_dataset_filepaths.get_rgb_path(camera_name, frame_number)
            dst_mask_path = output_dataset_filepaths.get_mask_path(camera_name, frame_number)

            dst_rgb_path.parent.mkdir(parents=True, exist_ok=True)
            dst_mask_path.parent.mkdir(parents=True, exist_ok=True)

            assert rgb_path.exists(), f"An RGB image is missing: {str(rgb_path)}"
            assert mask_path.exists(), f"A mask image is missing: {str(mask_path)}"
            cv2.imwrite(str(dst_rgb_path), cv2.imread(str(rgb_path)))
            shutil.copy(mask_path, dst_mask_path)

    # Convert camera intrinsics and extrinsics into ActorsHq format.
    intrinsics_path = dfa_dataset_path / "Intrinsic.inf"
    extrinsics_path = dfa_dataset_path / "CamPose.inf"

    cameras = []
    with open(intrinsics_path, "r") as intrinsics_file:
        for camera_number in range(36):
            assert camera_number == int(intrinsics_file.readline().strip(" \n"))
            fx, _, cx = [float(ele) for ele in intrinsics_file.readline().strip(" \n").split(" ")]
            _, fy, cy = [float(ele) for ele in intrinsics_file.readline().strip(" \n").split(" ")]
            intrinsics_file.readline()
            intrinsics_file.readline()

            cameras.append(
                CameraData(
                    name=f"Cam{camera_number:03d}",
                    width=1920,
                    height=1080,
                    rotation_axisangle=None,
                    translation=None,
                    focal_length=np.array([fx / 1920, fy / 1080]),
                    principal_point=np.array([cx / 1920, cy / 1080]),
                )
            )

    with open(extrinsics_path, "r") as extrinsics_file:
        for camera, line in zip(cameras, extrinsics_file):
            extrinsics = np.array([float(ele) for ele in line.strip(" \n").split(" ")])
            camera_to_world = np.zeros((3, 3))
            camera_to_world[:, 2] = extrinsics[0:3]
            camera_to_world[:, 0] = extrinsics[3:6]
            camera_to_world[:, 1] = extrinsics[6:9]

            camera.rotation_axisangle = Rotation.from_matrix(camera_to_world).as_rotvec()
            camera.translation = extrinsics[-3:]

    write_calibration_csv(cameras, output_dataset_filepaths.calibration_path)
    print("Calibration file is written.")

    # Write initial aabbs.csv
    frame_numbers = sorted(frame_numbers)

    # Our initial assumption: we know that the scene is always contained in [-1.5, 1.5].
    bound = 1.5
    initial_aabb = np.array([[-bound, -bound, -bound], [bound, bound, bound]])
    write_aabbs_csv([AabbData(fnum, initial_aabb) for fnum in frame_numbers], output_dataset_filepaths.aabbs_path)
    print("Initial aabb.csv is written.")

    # Compute initial occupancy grids based on the initial aabbs.
    grid_resolution = 256
    generate_occupancy_grid_from_masks(
        data_folder=output_folder,
        grid_resolution=grid_resolution,
        camera_coverage_threshold=36,
    )
    print("Initial occupancy grids are generated.")

    # Now from the initial occupancy grids, calculate tighter aabbs.
    aabbs = []
    output_dataset = VolumetricDataset(output_folder)
    xx, yy, zz = np.meshgrid(
        np.linspace(start=-bound, stop=bound, num=grid_resolution),
        np.linspace(start=-bound, stop=bound, num=grid_resolution),
        np.linspace(start=-bound, stop=bound, num=grid_resolution),
        indexing="ij",
    )
    coords = np.stack((zz, yy, xx), axis=-1)
    for frame_number in tqdm(frame_numbers, desc="Refining the aabbs based on the initial occupancy grids"):
        grid = output_dataset.get_occupancy_grid(frame_number)
        occupied = coords[grid > 0]
        aabb = np.stack((occupied.min(0), occupied.max(0)), axis=0)
        assert (np.abs(aabb) < bound).all()
        aabbs.append(AabbData(frame_number=frame_number, aabb=aabb))
    write_aabbs_csv(aabbs, output_dataset_filepaths.aabbs_path)
    print("Final aabb.csv is written.")

    # Generate the final occupancy grids.
    generate_occupancy_grid_from_masks(
        data_folder=output_folder,
        grid_resolution=grid_resolution,
        camera_coverage_threshold=36,
    )
    print("Final occupancy grids are generated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dfa_dataset_folder", type=Path, required=True, help="The input DFA dataset folder, e.g., /path/to/cat"
    )
    parser.add_argument(
        "--motion_type",
        type=str,
        required=True,
        help="Indicates the motion type performed by the animal, e.g., walkprogressive_noz and run.",
    )
    parser.add_argument("--output_folder", type=Path, required=True, help="Output ActorsHQ dataset folder.")
    args = parser.parse_args()

    import_dfa(
        args.dfa_dataset_folder,
        args.motion_type,
        args.output_folder,
    )


if __name__ == "__main__":
    main()
