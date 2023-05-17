from __future__ import annotations

import copy
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from actorshq.dataset.aabb_data import read_aabbs_csv
from actorshq.dataset.camera_data import CameraData, read_calibration_csv


class VolumetricDatasetFilepaths:
    CAMERA_IDENTIFIER = "{camera_name}"
    FRAME_IDENTIFIER = "{frame_number:06d}"
    FRAME_IDENTIFIER_MODULO = "%06d"

    RGB_PATTERN = f"rgbs/{CAMERA_IDENTIFIER}/{CAMERA_IDENTIFIER}_rgb{FRAME_IDENTIFIER}.jpg"
    MASK_PATTERN = f"masks/{CAMERA_IDENTIFIER}/{CAMERA_IDENTIFIER}_mask{FRAME_IDENTIFIER}.png"
    OCCUPANCY_GRID_PATTERN = f"occupancy_grids/occupancy_grid{FRAME_IDENTIFIER}.npz"
    CALIBRATION_CSV = "calibration.csv"
    AABBS_CSV = "aabbs.csv"
    MESH_FILE = "meshes.abc.xz"
    BLEND_FILE = "scene.blend"
    LIGHT_ANNOTATIONS_CSV = "light_annotations.csv"
    METADATA_JSON = "scene.json"

    def __init__(self, data_folder: Path) -> None:
        self.folder = data_folder
        self.calibration_path: Path = data_folder / VolumetricDatasetFilepaths.CALIBRATION_CSV
        self.aabbs_path: Path = data_folder.parent / VolumetricDatasetFilepaths.AABBS_CSV
        self.metadata_path: Path = data_folder.parent / VolumetricDatasetFilepaths.METADATA_JSON

    def _get_pattern(self, pattern: str) -> str:
        return str(
            self.folder
            / pattern.replace(
                VolumetricDatasetFilepaths.FRAME_IDENTIFIER,
                VolumetricDatasetFilepaths.FRAME_IDENTIFIER_MODULO,
            )
        )

    def get_rgb_path(self, camera_name: str, frame_number: int) -> Path:
        return self.folder / VolumetricDatasetFilepaths.RGB_PATTERN.format(
            camera_name=camera_name,
            frame_number=frame_number,
        )

    def get_mask_path(self, camera_name: str, frame_number: int) -> Path:
        return self.folder / VolumetricDatasetFilepaths.MASK_PATTERN.format(
            camera_name=camera_name,
            frame_number=frame_number,
        )

    def get_occupancy_grid_path(self, frame_number: int) -> Path:
        return self.folder.parent / VolumetricDatasetFilepaths.OCCUPANCY_GRID_PATTERN.format(
            frame_number=frame_number,
        )

    def get_light_annotations_path(self) -> Path:
        return self.folder / VolumetricDatasetFilepaths.LIGHT_ANNOTATIONS_CSV

    def get_metadata_path(self) -> Path:
        return self.folder.parent / VolumetricDatasetFilepaths.METADATA_JSON

    def get_rgb_pattern(self) -> str:
        return self._get_pattern(pattern=VolumetricDatasetFilepaths.RGB_PATTERN)

    def get_mask_pattern(self) -> str:
        return self._get_pattern(pattern=VolumetricDatasetFilepaths.MASK_PATTERN)


class VolumetricDataset:

    NUM_CAMERAS = 160

    def __init__(
        self,
        data_folder: Path,
        crop_center_square: bool = False,
    ) -> None:
        """
        Args:
            data_folder (Path): Path to the dataset.
            crop_center_square (bool): Whether to crop the center square from each image. This may introduce several benefits:
                                       1-There would be just a single resolution if this feature is used.
                                       2-Faster training because we get rid of some regions that are repeated in other images.
                                       Defaults to False.
        """
        self.filepaths = VolumetricDatasetFilepaths(data_folder=data_folder)
        self.cameras = read_calibration_csv(self.filepaths.calibration_path)
        self.aabbs = read_aabbs_csv(self.filepaths.aabbs_path)
        if crop_center_square:
            self.crop_offsets = self._crop_cameras()
        else:
            self.crop_offsets = None

        self._cname2camera = {camera.name: camera for camera in self.cameras}
        self._cname2cnum = {camera.name: cnum for cnum, camera in enumerate(self.cameras)}
        self._fnum2aabb = {aabb.frame_number: aabb for aabb in self.aabbs}

    def get_available_cameras_and_frames(self) -> Tuple[List[int], List[int]]:
        """Queries for which cameras and frames the RGB images are actually available, and returns them as tuple
        of lists.

        Returns:
            Tuple[List[int], List[int]]: Tuple of available camera numbers and frame numbers.
        """
        available_cameras = [
            camera_number
            for camera_number, camera in enumerate(self.cameras)
            if len(list(Path(self.filepaths.get_rgb_pattern().format(camera_name=camera.name)).parent.glob("*"))) > 0
        ]
        available_frames = [
            frame_number
            for frame_number in self._fnum2aabb
            if self.filepaths.get_rgb_path(self.cameras[available_cameras[0]].name, frame_number).exists()
        ]
        return available_cameras, available_frames

    def get_scaled_cameras(self, scene_offset: np.ndarray, scene_scale: float) -> List[CameraData]:
        """We often need a canonical space to perform rendering. Therefore, it is sometimes desirable to modify camera
        positions (via translation and scaling) to fit them in a certain space (e.g., unit cube).

        Args:
            scene_offset (np.ndarray): Camera positions are first translated with this offset.
            scene_scale (float): Cameras are scaled with this value after being translated.

        Returns:
            List[CameraData]: Scaled cameras.
        """
        cameras = copy.deepcopy(self.cameras)
        for camera in cameras:
            camera.translation = (camera.translation + scene_offset) * scene_scale

        return cameras

    def get_aabb(self, frame_numbers: Optional[List[int]] = None) -> np.ndarray:
        """Calculates the union bounding box over the provided frame numbers. If the argument is set to None,
        it returns the union over all the available frames.
        """
        if frame_numbers is None:
            all_aabbs = np.stack([aabb.aabb for aabb in self.aabbs], axis=0)
        else:
            all_aabbs = np.stack([self._fnum2aabb[i].aabb for i in frame_numbers], axis=0)
        return np.stack((all_aabbs[:, 0].min(0), all_aabbs[:, 1].max(0)), axis=0)

    def get_occupancy_grid(self, frame_number: int) -> np.ndarray:
        """Retrieves the occupancy grid for the specified frame and returns it as a numpy array."""
        return np.load(self.filepaths.get_occupancy_grid_path(frame_number))["occupancy_grid"]

    def get_rgb(self, camera_number: int, frame_number: int, normalize: bool = True) -> np.ndarray:
        """Retrieves the RGB image specified by the camera and frame numbers, and returns it as a numpy array.
        Image is normalized to range [0, 1] by default (normalize==True)
        """
        if self.crop_offsets is not None:
            crop_x, crop_y = self.crop_offsets[camera_number]
        else:
            crop_x, crop_y = 0, 0

        camera = self.cameras[camera_number]
        rgb_path = str(self.filepaths.get_rgb_path(camera.name, frame_number))
        rgb = cv2.imread(rgb_path)

        if normalize:
            rgb = rgb / np.float32(255)

        return rgb[crop_y : crop_y + camera.height, crop_x : crop_x + camera.width]

    def get_mask(self, camera_number: int, frame_number: int, normalize: bool = True) -> np.ndarray:
        """Retrieves the mask image specified by the camera and frame numbers, and returns it as a numpy array.
        Image is normalized to range [0, 1] by default (normalize==True)
        """
        if self.crop_offsets is not None:
            crop_x, crop_y = self.crop_offsets[camera_number]
        else:
            crop_x, crop_y = 0, 0

        camera = self.cameras[camera_number]
        mask_path = str(self.filepaths.get_mask_path(camera.name, frame_number))
        mask = cv2.imread(mask_path)
        if len(mask.shape) == 2:
            mask = mask[..., None]
        elif len(mask.shape) == 3:
            mask = mask[..., 0:1]

        if normalize:
            mask = mask / np.float32(255)

        return mask[crop_y : crop_y + camera.height, crop_x : crop_x + camera.width]

    def get_light_annotations(self) -> Dict:
        """Load light annotations from the light_annotations.csv file.

        Returns:
            Dict: A dictionary with camera number as key and a list of light annotations as value.
                  Each annotation provides a tuple of (x, y, r) where x and y are the coordinates of the center of the
                  light and r is the radius of the light.
        """
        with open(self.filepaths.get_light_annotations_path()) as csv_file:
            lights_csv_reader = csv.DictReader(csv_file)
            light_annotations = defaultdict(list)
            for row in lights_csv_reader:
                camera = self._cname2camera[row["camera"]]
                camera_number = self._cname2cnum[camera.name]

                if self.crop_offsets is not None:
                    crop_x, crop_y = self.crop_offsets[camera_number]
                else:
                    crop_x, crop_y = 0, 0

                light_annotations[camera_number].append(
                    (
                        round(float(row["x"]) - crop_x),
                        round(float(row["y"]) - crop_y),
                        round(float(row["r"])),
                    )
                )
            return light_annotations

    def _crop_cameras(self) -> List[Tuple[int, int]]:
        """Adjusts self.cameras to represent the cropped version of their original images. The new resolution for
        each image would be (min_len, min_len) where min_len=min(width, height).

        Returns:
            List[Tuple[int, int]]: List of coordinates indicating the top-left corners of each crop.
        """
        crop_offsets = []
        for camera in self.cameras:
            offset = np.abs(camera.height - camera.width) // 2
            if camera.width < camera.height:
                offset_h = offset
                offset_w = 0
                new_width = new_height = camera.width
            else:
                offset_h = 0
                offset_w = offset
                new_width = new_height = camera.height

            crop_offsets.append((offset_w, offset_h))
            camera.principal_point[0] -= offset_w / camera.width
            camera.principal_point[1] -= offset_h / camera.height

            scaling_w = camera.width / new_width
            scaling_h = camera.height / new_height
            camera.focal_length[0] *= scaling_w
            camera.focal_length[1] *= scaling_h
            camera.principal_point[0] *= scaling_w
            camera.principal_point[1] *= scaling_h

            camera.width = new_width
            camera.height = new_height

        return crop_offsets
