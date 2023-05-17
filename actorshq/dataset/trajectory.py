import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

from actorshq.dataset.camera_data import read_calibration_csv, write_calibration_csv
from actorshq.dataset.data_loader import DataLoader
from actorshq.dataset.generate_camera_trajectory import generate_camera_trajectory
from actorshq.dataset.volumetric_dataset import VolumetricDataset, VolumetricDatasetFilepaths


def get_trajectory_dataloader_from_calibration(
    calibration_path: Path,
    base_data_folder: Path,
    device: str,
    dataloader_output_mode: DataLoader.OutputMode,
    space_pruning_mode: DataLoader.SpacePruningMode,
    batch_size: int,
    frame_numbers: Tuple[int, ...],
) -> DataLoader:
    """Creates the trajectory from the calibration path provided as an argument.

    The trajectory is as follows if [# cameras] > len(frame_numbers) -- e.g. for 5 cameras and 3 frames:
    (Cam1, Frame1), (Cam2, Frame2), (Cam3, Frame3), (Cam4, Frame2), (Cam5, Frame1)

    The trajectory is as follows if [# cameras] < len(frame_numbers) -- e.g. for 3 cameras and 5 frames:
    (Cam1, Frame1), (Cam2, Frame2), (Cam3, Frame3), (Cam2, Frame4), (Cam1, Frame5)

    Args:
        calibration_path (Path):
            Path to the calibration csv that contains trajectory cameras.
        base_data_folder (Path):
            The dataset that provides per-frame properties such as aabbs and occupancy grids to be used by the
            newly-created trajectory.
        device (str):
            PyTorch device, e.g., 'cuda'.
        dataloader_output_mode (DataLoader.OutputMode):
            Can be one of RAYS or RAYS_AND_SAMPLES (see description of `DataLoader.OutputMode`)
        space_pruning_mode (DataLoader.SpacePruningMode):
            Can be one of AABB or OCCUPANCY_GRID. (see description of `DataLoader.SpacePruningMode` enum)
        batch_size (int):
            Indicates the number of rays to be sampled for each iteration.
        frame_numbers (Tuple[int, ...]):
            Frame numbers (0-indexed) to be used from the dataset.

    Returns:
        DataLoader: A data loader for the trajectory through the cameras in the input calibration file.
    """
    test_data_folder = base_data_folder.parent / "test"
    if test_data_folder.exists():
        shutil.rmtree(test_data_folder)

    test_data_folder.mkdir()
    new_dataset_fp = VolumetricDatasetFilepaths(test_data_folder)
    shutil.copy(calibration_path, new_dataset_fp.calibration_path)

    new_cameras = read_calibration_csv(new_dataset_fp.calibration_path)
    trajectory_num_cameras = len(new_cameras)
    assert trajectory_num_cameras > 0

    render_sequence = []
    total_num_frames = len(frame_numbers)
    total_length = max(total_num_frames, trajectory_num_cameras)
    for num in range(total_length):
        camera_number = num % trajectory_num_cameras
        if (num // trajectory_num_cameras) % 2 == 1:
            camera_number = trajectory_num_cameras - 1 - camera_number

        frame_idx = num % total_num_frames
        if (num // total_num_frames) % 2 == 1:
            frame_idx = total_num_frames - 1 - frame_idx

        render_sequence.append((camera_number, frame_numbers[frame_idx]))

    return DataLoader(
        dataset=VolumetricDataset(new_dataset_fp.folder, crop_center_square=False),
        device=device,
        mode=DataLoader.Mode.TEST,
        dataloader_output_mode=dataloader_output_mode,
        space_pruning_mode=space_pruning_mode,
        batch_size=batch_size,
        camera_numbers=list(range(trajectory_num_cameras)),
        frame_numbers=frame_numbers,
        max_buffer_size=1,
        render_sequence=render_sequence,
    )


def get_trajectory_dataloader_from_keycams(
    trajectory: Tuple[int, ...],
    base_data_folder: Path,
    device: str,
    dataloader_output_mode: DataLoader.OutputMode,
    space_pruning_mode: DataLoader.SpacePruningMode,
    batch_size: int,
    frame_numbers: Tuple[int, ...],
    trajectory_num_cameras: int,
) -> DataLoader:
    """Creates a data loader for the trajectory through specified camera indices ('trajectory').

    The trajectory is as follows if trajectory_num_cameras > len(frame_numbers) -- e.g. for 5 cameras and 3 frames:
    (Cam1, Frame1), (Cam2, Frame2), (Cam3, Frame3), (Cam4, Frame2), (Cam5, Frame1)

    The trajectory is as follows if trajectory_num_cameras < len(frame_numbers) -- e.g. for 3 cameras and 5 frames:
    (Cam1, Frame1), (Cam2, Frame2), (Cam3, Frame3), (Cam2, Frame4), (Cam1, Frame5)

    Args:
        trajectory (Tuple[int, ...]):
            Camera indices that specifies the 0-indexed 'key' cameras such that a trajectory through these cameras is
            created.
        base_data_folder (Path):
            The dataset that provides cameras to create the trajectory.
        device (str):
            PyTorch device, e.g., 'cuda'.
        dataloader_output_mode (DataLoader.OutputMode):
            Can be one of RAYS or RAYS_AND_SAMPLES (see description of `DataLoader.OutputMode`)
        space_pruning_mode (DataLoader.SpacePruningMode):
            Can be one of AABB or OCCUPANCY_GRID. (see description of `DataLoader.SpacePruningMode` enum)
        batch_size (int):
            Indicates the number of rays to be sampled for each iteration.
        frame_numbers (Tuple[int, ...]):
            Frame numbers (0-indexed) to be used from the dataset.
        trajectory_num_cameras (int):
            Not used when there is only one camera specified in trajectory. Specifies the number of cameras to generate
            for one complete trajectory. Note that if the number of frames available via frame_numbers is more than
            [trajectory_num_cameras], trajectory is repeated enough to show all the frames.

    Returns:
        DataLoader: A data loader for the trajectory through specified cameras.
    """
    if len(trajectory) == 1:
        return DataLoader(
            dataset=VolumetricDataset(base_data_folder, crop_center_square=False),
            device=device,
            mode=DataLoader.Mode.TEST,
            dataloader_output_mode=dataloader_output_mode,
            space_pruning_mode=space_pruning_mode,
            batch_size=batch_size,
            camera_numbers=trajectory,
            frame_numbers=frame_numbers,
            max_buffer_size=1,
        )
    else:
        cameras = read_calibration_csv(VolumetricDatasetFilepaths(base_data_folder).calibration_path)
        trajectory_cameras = generate_camera_trajectory(
            key_cameras=[cameras[i] for i in trajectory],
            intrinsics_camera=cameras[trajectory[1]],
            num_frames=trajectory_num_cameras,
        )
        with TemporaryDirectory() as tmpdir:
            tmp_calibration_path = Path(tmpdir) / "calibration.csv"
            write_calibration_csv(trajectory_cameras, tmp_calibration_path)

            return get_trajectory_dataloader_from_calibration(
                calibration_path=tmp_calibration_path,
                base_data_folder=base_data_folder,
                device=device,
                dataloader_output_mode=dataloader_output_mode,
                space_pruning_mode=space_pruning_mode,
                batch_size=batch_size,
                frame_numbers=frame_numbers,
            )
