from typing import List

import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from actorshq.dataset.camera_data import CameraData


def generate_camera_trajectory(
    key_cameras: List[CameraData],
    intrinsics_camera: CameraData,
    num_frames: int,
) -> List[CameraData]:
    """Generates a smooth trajectory from the 'key_cameras'.

    For N cameras, there are N-1 intervals to fill in with cameras by interpolating between the key camera poses.
    To generate the poses between key cameras, we use spherical interpolation of key rotations, and spline interpolation
    of key translations.

    Args:
        key_cameras (List[CameraData]): Cameras to create the trajectory.
        intrinsics_camera (CameraData): The intrinsics of this camera will be used for the entire trajectory.
        num_frames (int): Indicates the number of cameras to be generated for the whole trajectory.
    Returns:
        List[CameraData]: Generated cameras along the smooth trajectory.
    """

    key_rotations = np.stack([camera.rotation_matrix_cam2world().T for camera in key_cameras], axis=0)
    key_translations = np.stack(
        [-rot @ camera.translation for camera, rot in zip(key_cameras, key_rotations)],
        axis=0,
    )
    key_cameras_positions = np.stack([camera.translation for camera in key_cameras], axis=0)
    key_rotations = R.from_matrix(key_rotations)

    interval_lengths = np.linalg.norm(key_cameras_positions[1:] - key_cameras_positions[0:-1], axis=1)
    interval_lengths /= interval_lengths.sum()
    key_times = np.cumsum([0] + list(interval_lengths))
    slerp_func = Slerp(key_times, key_rotations)
    interp_spline_func = interpolate.make_interp_spline(key_times, key_translations, k=2)

    time_samples = np.linspace(1e-5, 1 - 1e-5, num_frames)
    interpolated_rotations = slerp_func(time_samples).as_matrix().astype(np.float32)
    interpolated_translations = interp_spline_func(time_samples).astype(np.float32)

    trajectory_cameras = []
    num_decimals = int(np.log10(num_frames)) + 1
    for idx, (rotation, translation) in enumerate(zip(interpolated_rotations, interpolated_translations)):
        camera = CameraData(
            name=f"Cam{idx + 1}".zfill(num_decimals),
            width=intrinsics_camera.width,
            height=intrinsics_camera.height,
            rotation_axisangle=R.from_matrix(rotation.T).as_rotvec(),
            translation=-rotation.T @ translation,
            focal_length=intrinsics_camera.focal_length.copy(),
            principal_point=intrinsics_camera.principal_point.copy(),
        )
        trajectory_cameras.append(camera)

    return trajectory_cameras
