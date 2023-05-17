from __future__ import annotations

import itertools
import multiprocessing
import threading
import time
from enum import Enum
from multiprocessing.pool import ThreadPool
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch

import actorshq.dataset.occupancy_grid_native as occupancy_grid_native
import actorshq.dataset.ray_sampler_native as ray_sampler_native
from actorshq.dataset.input_batch import InputBatch
from actorshq.dataset.volumetric_dataset import VolumetricDataset


class DataLoader:
    class Mode(Enum):
        # Mode.TRAINING generates random samples (not suitable to render complete images) and provides supervision data
        # (e.g., rgb and mask)
        TRAINING = 0

        # Mode.VALIDATION is not random, allows to generate full images and also provides supervision data for
        # evaluation purposes.
        VALIDATION = 1

        # Mode.TEST is like Mode.VALIDATION but without any supervision data.
        TEST = 2

    class OutputMode(Enum):
        # OutputMode.RAYS samples rays and provides other per-ray information for neural rendering.
        RAYS = 0

        # OutputMode.RAYS_AND_SAMPLES first samples rays. Then, it samples points along each ray to be used
        # during volumetric rendering. These positions are tested and filtered against the occupancy grid of the scene
        # if SpacePruningMode.OCCUPANCY_GRID is chosen.
        RAYS_AND_SAMPLES = 1

    class SpacePruningMode(Enum):
        # In SpacePruningMode.AABB, the entire volumetric rendering process will be performed inside
        # the axis-aligned bounding box (AABB) of a particular time frame. This also means, the entry and exit point
        # of a ray through the aabb is considered as start and end distances while performing raymarching.
        AABB = 0

        # In SpacePruningMode.OCCUPANCY_GRID, the entire volumetric rendering process will be performed inside
        # the occupancy grid of a particular time frame. This also means, the entry and exit point
        # of a ray through the occupancy grid is considered as start and end distances while performing raymarching.
        OCCUPANCY_GRID = 1

    def __init__(
        self,
        dataset: VolumetricDataset,
        device: str,
        mode: DataLoader.Mode,
        dataloader_output_mode: DataLoader.OutputMode,
        space_pruning_mode: DataLoader.SpacePruningMode,
        batch_size: int,
        camera_numbers: Tuple[int],
        frame_numbers: Tuple[int],
        max_buffer_size: int,
        max_num_frames_per_batch: Optional[int] = None,
        use_mask: Optional[bool] = None,
        filter_light_bloom: Optional[bool] = None,
        render_sequence: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """Considering the terabytes of data we need to deal with, preparing batches in a precomputation stage and saving
        it to the filesystem would be an overkill in terms of storage requirements.

        The idea behind this dataloader is to bypass the reading/writing large chunks of data by sampling batch of rays
        on the fly from as many images as possible across different cameras and time frames. To do this, we define pool
        of images, and randomly sample from this pool continuously in the main thread while another thread is working
        in the background to replace the images in the pool (also called buffer in the code). By replacing and sampling
        at the same time, we use only a modest amount of GPU & CPU memory to accommodate the pool.

        Also, ray_sampler_native makes use of occupancy grids (that are initialized from masks) and aabbs to skip empty space.
        This speeds up the rendering significantly, and increases the effective capacity of the model because the empty
        space does not have to be modeled.

        Args:
            dataset (VolumetricDataset):
                                Dataset, that implements the minimal functionality for training.
            device (str):
                                PyTorch device, e.g., 'cuda'.
            mode (DataLoader.Mode):
                                Can be one of TRAINING, VALIDATION and TEST (see description of `Mode` enum)
            dataloader_output_mode (DataLoader.OutputMode):
                                Can be one of RAYS or RAYS_AND_SAMPLES (see description of `OutputMode`)
            space_pruning_mode (DataLoader.SpacePruningMode):
                                Can be one of AABB or OCCUPANCY_GRID. (see description of `SpacePruningMode` enum)
            batch_size (int):
                                Indicates the number of rays to be sampled for each iteration.
            camera_numbers (Tuple[int]):
                                Camera numbers (0-indexed) to be used in the dataset.
            frame_numbers (Tuple[int]):
                                Frame numbers (0-indexed) to be used in the dataset.
            max_buffer_size (int):
                                Upper limit on the size of the pool of images from which the rays are sampled.
            max_num_frames_per_batch (Optional[int], optional): Defaults to None.
                                Only for TRAINING.
                                Indicates the number of distinct frames that could exist in the pool from which the
                                rays are sampled. It only guarantees the upper bound for values >= 2. The value also
                                indicates the maximum number of occupancy grid textures that will reside in the CUDA
                                memory at any time instance.
            use_mask (Optional[bool], optional): Defaults to None.
                                Only for TRAINING and VALIDATION.
                                Whether to use mask information in the dataset. If True, then the RGB images will be
                                set to black for training wherever the mask is zero. If False, then the alpha channel
                                of the ground truth RGBA will be set to 1 on the full image.
            filter_light_bloom (Optional[bool], optional): Defaults to None.
                                Only for TRAINING and VALIDATION.
                                Whether to filter out light bloom regions.
            render_sequence (Optional[List[Tuple[int, int]]], optional): Defaults to None.
                                Only for VALIDATION and TEST.
                                Indicates sequence of (camera, frame) pairs that want to be rendered. Indices are from
                                camera_numbers and frame_numbers lists.
        """
        super().__init__()

        self.device = device
        self.mode = mode
        self.batch_size = batch_size
        # Check camera numbers for duplicates.
        self.camera_numbers = camera_numbers
        if len(set(self.camera_numbers)) != len(self.camera_numbers):
            raise RuntimeError("Provided camera numbers cannot have duplicates.")
        self.frame_numbers = frame_numbers
        # Check frame numbers for duplicates.
        if len(set(self.frame_numbers)) != len(self.frame_numbers):
            raise RuntimeError("Provided frame numbers cannot have duplicates.")

        # Function to check if optional arguments that are valid to a subset of modes are used properly or not.
        def _check_and_get_arg(arg: Any, arg_name: str, valid_modes: List[DataLoader.Mode], non_valid_default: Any):
            if self.mode in valid_modes:
                if arg is None:
                    raise RuntimeError(f"'{arg_name}' has to be given for {str(self.mode)}")
                return arg
            else:
                if arg is not None:
                    raise RuntimeError(f"'{arg_name}' cannot be used for {str(self.mode)}")
                return non_valid_default

        # Check if 'max_num_frames_per_batch' is properly used.
        self.max_num_frames_per_batch = _check_and_get_arg(
            max_num_frames_per_batch, "max_num_frames_per_batch", [DataLoader.Mode.TRAINING], None
        )
        if self.mode == DataLoader.Mode.TRAINING:
            if len(self.frame_numbers) > 1 and self.max_num_frames_per_batch < 2:
                raise RuntimeError("'max_num_frames_per_batch >= 2' has to be met.")
            self.max_num_frames_per_batch = min(self.max_num_frames_per_batch, len(self.frame_numbers))
        # Check if 'use_mask' is properly used.
        self.use_mask = _check_and_get_arg(
            use_mask, "use_mask", [DataLoader.Mode.TRAINING, DataLoader.Mode.VALIDATION], False
        )
        # Check if 'filter_light_bloom' is properly used.
        self.filter_light_bloom = _check_and_get_arg(
            filter_light_bloom, "filter_light_bloom", [DataLoader.Mode.TRAINING, DataLoader.Mode.VALIDATION], False
        )
        # Check if 'render_sequence' is properly used.
        self.render_sequence = _check_and_get_arg(
            render_sequence, "render_sequence", [DataLoader.Mode.VALIDATION, DataLoader.Mode.TEST], None
        )
        # Set total number of (camera, frame) pairs to be processed.
        if self.mode == DataLoader.Mode.TRAINING:
            self.num_camera_frame_pairs = len(camera_numbers) * len(frame_numbers)
        else:
            self.num_camera_frame_pairs = len(self.render_sequence)

        self.space_pruning_mode = space_pruning_mode
        om_str = "rays" if dataloader_output_mode == DataLoader.OutputMode.RAYS else "samples"
        sp_str = "aabb" if space_pruning_mode == DataLoader.SpacePruningMode.AABB else "occupancy"
        self.ray_sampler_func = getattr(ray_sampler_native, f"get_{om_str}_{sp_str}_minmax")

        self.dataset = dataset

        # Here, the axis-aligned bounding box (AABB) is the tightest bounding box that covers the space spanned by all
        # the frames. In other words, the logical union of the occupancy grids of all the frames in this dataset is
        # contained in this AABB.
        self.aabb = self.dataset.get_aabb()
        self.scene_offset = -self.aabb.mean(0)
        self.scene_scale = 1 / np.max(self.aabb[1] - self.aabb[0])

        # Translate and scale the camera positions so that the scene that is encapsulated by the AABB resides
        # in [-0.5, 0.5]
        self.cameras = self.dataset.get_scaled_cameras(
            scene_offset=self.scene_offset,
            scene_scale=self.scene_scale,
        )
        # We use inverse projection matrices to compute a ray direction from a pixel (x, y).
        # e.g., raydir = inverse(projection_matrix)[:3, :3] × (x, y, 1).
        self.all_inverse_krs = (
            torch.from_numpy(
                np.stack(
                    [np.linalg.inv(cam.projection_matrix_world2pixel()) for cam in self.cameras],
                    axis=0,
                )
            )[..., :3, :3]
            .transpose(-1, -2)
            .float()
            .cuda()
            .contiguous()
        )  # Shape: (N, 3, 3)
        # Important: the ".transpose(-1, -2)" is performed here just because the glm library used on the CUDA side
        # stores matrices in column-major order.

        self.all_camera_origins = (
            torch.from_numpy(np.stack([cam.translation for cam in self.cameras], axis=0)).float().cuda().contiguous()
        )  # Shape: (N, 3)

        # Apply the same scaling to the AABB so that the ray-aabb intersection gives accurate results.
        self.aabb = (self.aabb + self.scene_offset) * self.scene_scale
        self.aabb = torch.from_numpy(self.aabb).to(device).float().contiguous()

        unique_num_pixels = list(set(self.cameras[cn].width * self.cameras[cn].height for cn in self.camera_numbers))
        if len(unique_num_pixels) != 1:
            raise RuntimeError("Each camera should have the same number of pixels!")
        self.num_pixels_per_camera = unique_num_pixels[0]
        self.num_batches_per_full_image = int(np.ceil(self.num_pixels_per_camera / self.batch_size))

        # Count how many different image resolutions exist in the dataset.
        unique_resolutions = list(set((self.cameras[cn].width, self.cameras[cn].height) for cn in self.camera_numbers))
        if len(unique_resolutions) > 2 or (
            len(unique_resolutions) == 2
            and not (
                unique_resolutions[0][0] == unique_resolutions[1][1]
                and unique_resolutions[0][1] == unique_resolutions[1][0]
            )
        ):
            raise RuntimeError(
                "Currently, we only support one image resolution with landspace or portrait mode!"
                " (effectively two different resolutions)"
            )

        width = max(unique_resolutions[0][0], unique_resolutions[0][1])
        height = min(unique_resolutions[0][0], unique_resolutions[0][1])
        self.resolution = width, height

        self.light_annotations = None
        if filter_light_bloom:
            self.light_annotations = self.dataset.get_light_annotations()
            person_border_size = round((80 / 4088) * width)
            self.light_annotations_border_filter = np.ones((person_border_size, person_border_size), np.uint8)

        # Shrink the buffer if it's larger than the number of images available in the dataset.
        self.buffer_size = min(max_buffer_size, self.num_camera_frame_pairs)
        if self.mode == DataLoader.Mode.TRAINING:
            if self.max_num_frames_per_batch > 1:
                # Shrink the buffer if it's unnecessarily large.
                max_reasonable_buffer_size = len(camera_numbers) * (self.max_num_frames_per_batch - 1)
                self.buffer_size = min(self.buffer_size, max_reasonable_buffer_size)
            self.occupancy_grids_buffer_size = min(self.buffer_size, self.max_num_frames_per_batch)
        else:
            self.occupancy_grids_buffer_size = min(self.buffer_size, len(self.frame_numbers))

        # All the tensors below with 'self.buffer_size' as one of their dimensions, stores data corresponding
        # to entries in the buffer. For example, self.pixel_colors_cpu has per-entry rgba data and
        # self.frame_numbers_cuda has per-entry frame number information.
        self.pixel_colors_cpu = torch.empty(
            size=(self.buffer_size, self.num_pixels_per_camera, 4),
            device="cpu",
            dtype=torch.uint8,
        )

        self.light_mask_cpu = torch.empty(
            size=(self.buffer_size, self.num_pixels_per_camera, 1),
            device="cpu",
            dtype=torch.bool,
        )

        if self.mode == DataLoader.Mode.TRAINING:
            self.frame_numbers_cuda = torch.full(
                size=(self.buffer_size,),
                fill_value=-1,
                device=device,
                dtype=torch.int32,
            )
            self.camera_numbers_cuda = torch.full(
                size=(self.buffer_size,),
                fill_value=-1,
                device=device,
                dtype=torch.int32,
            )

        self.landscape_mode_cuda = torch.empty(
            size=(self.buffer_size,),
            device=device,
            dtype=torch.bool,
        )

        self.inverse_krs_cuda = torch.empty(
            size=(self.buffer_size, 3, 3),
            device=device,
            dtype=torch.float,
        )

        self.camera_origins_cuda = torch.empty(
            size=(self.buffer_size, 3),
            device=device,
            dtype=torch.float,
        )

        self.grid_texture_objects_cuda = torch.empty(
            size=(self.buffer_size,),
            device=device,
            dtype=torch.int64,
        )

        self.occupancy_grid_resolution = 0
        if space_pruning_mode == DataLoader.SpacePruningMode.OCCUPANCY_GRID:
            self.occupancy_grid_resolution = self.dataset.get_occupancy_grid(frame_number=0).shape[0]
            self.cuda_grid_texture = occupancy_grid_native.OccupanyGrid(
                self.occupancy_grid_resolution,
                self.occupancy_grids_buffer_size,
            )
            # Keeps track of the frame numbers that are currently in the buffer and which 'grid_texture_object' index they
            # correspond to.
            self.frame_to_grid_texture = {}

            # Used to enforce the integrity of the loading and caching behavior of the occupancy grids.
            self.grid_texture_lock = threading.Lock()

        # Used to make sure sampling from the dataset and replacing the data will not interfere.
        self.data_lock = threading.Lock()
        # Used to pause or continue the work of replacer thread.
        self.replacer_event = threading.Event()
        self.replacer_event.clear()
        # If the buffer size is not enough accomodate all the camera-frame pairs.
        self.run_replacer_thread = self.buffer_size < self.num_camera_frame_pairs

        if self.run_replacer_thread and self.mode != DataLoader.Mode.TRAINING:
            self.empty_slots_sem = threading.Semaphore(self.buffer_size)
            self.available_slots_sem = threading.Semaphore(0)

        self.camera_frame_pairs = self._camera_frame_pair_generator()
        preload_pairs = [next(self.camera_frame_pairs) for _ in range(self.buffer_size)]
        thread_pool_size = min(multiprocessing.cpu_count(), self.buffer_size)
        start = time.time()
        with ThreadPool(thread_pool_size) as pool:
            pool.starmap(
                self._load_and_copy_camera_frame_data,
                zip(
                    preload_pairs,
                    range(self.buffer_size),
                    itertools.repeat(None),
                ),
            )
        print(f"Images are loaded in {time.time() - start} seconds by a pool of {thread_pool_size} processes.")
        self.pair_load_index = self.buffer_size

        if self.run_replacer_thread:
            threading.Thread(target=self._replace_next_buffer_entry, daemon=True).start()

    def _camera_frame_pair_generator(self):
        if self.mode != DataLoader.Mode.TRAINING:
            for pair in itertools.cycle(self.render_sequence):
                yield pair
        else:
            # 'num_cams_per_frame_in_buffer' is the number of images that are loaded into the buffer at the same time
            # per frame.
            if self.max_num_frames_per_batch > 1:
                num_cams_per_frame_in_buffer = int(np.ceil(self.buffer_size / (self.max_num_frames_per_batch - 1)))
            elif self.max_num_frames_per_batch == 1:
                assert len(self.frame_numbers) == 1
                num_cams_per_frame_in_buffer = len(self.camera_numbers)
            assert num_cams_per_frame_in_buffer <= len(self.camera_numbers)

            camera_numbers_per_frame = {
                frame_number: {
                    "next_yield_index": 0,
                    "camera_numbers": list(self.camera_numbers),
                }
                for frame_number in self.frame_numbers
            }
            frame_numbers = list(self.frame_numbers)
            while True:
                # In-place shuffling:
                # The generator repeatedly iterates through all `frame_numbers`. For each frame, it chooses
                # `num_cams_per_frame_in_buffer` cameras. Not all cameras will be loaded per frame iteration, so the
                # `next_yield_index` variable remembers for every frame which idx in the list of per frame cameras
                # was last yielded. Once all camera numbers have been yielded for a certain frame, the list of camera
                # numbers gets shuffled.
                np.random.shuffle(frame_numbers)
                for frame_number in frame_numbers:
                    frame_iter_info = camera_numbers_per_frame[frame_number]
                    for _ in range(num_cams_per_frame_in_buffer):
                        next_yield_index = frame_iter_info["next_yield_index"]
                        camera_numbers = frame_iter_info["camera_numbers"]
                        if next_yield_index == 0:
                            np.random.shuffle(camera_numbers)
                        yield camera_numbers[next_yield_index], frame_number
                        frame_iter_info["next_yield_index"] = (next_yield_index + 1) % len(camera_numbers)

    def _replace_next_buffer_entry(self):
        for camera_frame_pair in self.camera_frame_pairs:
            self.replacer_event.wait()

            # Update the frame_to_texture mapping before loading the data:
            # Clean out any frame_numbers that are currently not present in the buffer. There are certain number of
            # cameras that could be loaded into the buffer per frame (see 'self._camera_frame_pair_generator'). After
            # loading for a frame is finished, the data loader moves onto the next frame number. So, whenever the
            # frame_number changes, the frame_number that was processed 'max_num_frames_per_batch' frames ago will be
            # cleaned out of 'self.frame_to_grid_texture' so that the new frame can be added.
            if (
                self.space_pruning_mode == DataLoader.SpacePruningMode.OCCUPANCY_GRID
                and self.mode == DataLoader.Mode.TRAINING
            ):
                with self.grid_texture_lock:
                    pop_list = [fnum for fnum in self.frame_to_grid_texture if fnum not in self.frame_numbers_cuda]
                    for fnum in pop_list:
                        self.frame_to_grid_texture.pop(fnum)

                assert len(self.frame_to_grid_texture) <= self.occupancy_grids_buffer_size

            self._load_and_copy_camera_frame_data(
                camera_frame_pair=camera_frame_pair,
                buffer_index=self.pair_load_index % self.buffer_size,
                data_lock=self.data_lock,
            )
            self.pair_load_index += 1

    def _load_and_copy_camera_frame_data(
        self,
        camera_frame_pair: Tuple[int, int],
        buffer_index: int,
        data_lock: Optional[threading.Lock],
    ) -> None:
        camera_number, frame_number = camera_frame_pair
        camera = self.cameras[camera_number]

        if self.mode != DataLoader.Mode.TEST:
            # Using [2, 1, 0] converts BGR to RGB.
            rgb = self.dataset.get_rgb(camera_number, frame_number)[..., [2, 1, 0]]
            if self.use_mask:
                mask = self.dataset.get_mask(camera_number, frame_number)
                rgb *= mask
            else:
                mask = np.ones_like(rgb[..., 0:1])
            rgba = np.concatenate((rgb, mask), axis=-1)
            rgba = (rgba * np.float32(255)).astype(np.uint8).reshape(-1, 4)
            rgba = torch.from_numpy(rgba)

        # compute light_mask
        if self.light_annotations is not None:
            light_coords = self.light_annotations[camera_number]
            person_border = mask - cv2.erode(mask, self.light_annotations_border_filter)[..., np.newaxis]
            light_mask = np.zeros_like(rgb[..., 0:1], dtype=np.uint8)
            for c in light_coords:
                light_mask = cv2.circle(light_mask, (c[0], c[1]), c[2], (255), -1)
            light_mask = torch.from_numpy((person_border > 0) & (light_mask > 0)).reshape(-1, 1)

        if self.space_pruning_mode == DataLoader.SpacePruningMode.OCCUPANCY_GRID:
            if frame_number not in self.frame_to_grid_texture:
                grid_data_gpu = torch.from_numpy(self.dataset.get_occupancy_grid(frame_number)).to(device=self.device)

        # Following code implements producer-consumer scheme for non-training mode.
        # For training mode, we do not care about this because we randomly sample rays and images are not consumed as
        # whole. We only use the replace lock to make sure the ray sampler and data replacer do not work at the same
        # time. However, non-training mode sample rays of an entire image, and once an image is sampled as whole,
        # the next image needs to be loaded into the buffer. This requires the producer-consumer scheme.

        # If non-training mode, reduce the number of empty slots.
        if self.run_replacer_thread and self.mode != DataLoader.Mode.TRAINING:
            self.empty_slots_sem.acquire()

        # Acquire the lock before the critical region.
        # Note that locks hit the performance. So, if you don't have to use them, (e.g., during initialization),
        # where you know that it will not be consumed but only be uploaded, just set them to None.
        if data_lock is not None:
            data_lock.acquire()

        # Unfortunately, this part HAS TO be inside the critical region. The reason is that ray sampling kernel launch
        # and copying to texture memory via "self.cuda_grid_texture.add_grid" SHOULD NOT happen at the same time.
        # On the bright side, this has a performance penalty only when a new grid has to be copied thanks to the
        # caching mechanism introduced by self.frame_to_grid_texture.
        if self.space_pruning_mode == DataLoader.SpacePruningMode.OCCUPANCY_GRID:
            with self.grid_texture_lock:
                if frame_number in self.frame_to_grid_texture:
                    grid_texture_object = self.frame_to_grid_texture[frame_number]
                else:
                    grid_texture_object = self.cuda_grid_texture.add_grid(grid_data_gpu)

                    # We don't have to cache the grid textures if we're not on the training mode.
                    # Cost of switching to another grid texture is relatively negligible for non-training mode.
                    if self.mode == DataLoader.Mode.TRAINING:
                        self.frame_to_grid_texture[frame_number] = grid_texture_object

        try:
            if self.mode != DataLoader.Mode.TEST:
                self.pixel_colors_cpu[buffer_index].copy_(rgba, non_blocking=False)
                if self.light_annotations is not None:
                    self.light_mask_cpu[buffer_index].copy_(light_mask, non_blocking=False)
                if self.mode == DataLoader.Mode.TRAINING:
                    self.frame_numbers_cuda[buffer_index] = frame_number
                    self.camera_numbers_cuda[buffer_index] = camera_number
            self.landscape_mode_cuda[buffer_index] = camera.width > camera.height
            self.inverse_krs_cuda[buffer_index].copy_(self.all_inverse_krs[camera_number], non_blocking=False)
            self.camera_origins_cuda[buffer_index].copy_(self.all_camera_origins[camera_number], non_blocking=False)
            if self.space_pruning_mode == DataLoader.SpacePruningMode.OCCUPANCY_GRID:
                self.grid_texture_objects_cuda[buffer_index] = grid_texture_object
        finally:
            # Release the lock.
            if data_lock is not None:
                data_lock.release()

        # If non-training mode, increase the number of available slots.
        if self.run_replacer_thread and self.mode != DataLoader.Mode.TRAINING:
            for _ in range(self.num_batches_per_full_image):
                self.available_slots_sem.release()

    def __len__(self):
        if self.mode == DataLoader.Mode.TRAINING:
            raise NotImplementedError("Size of the training data loader is not defined.")
        else:
            return self.num_camera_frame_pairs * self.num_pixels_per_camera

    def __iter__(self):
        self.iternum = 0
        # Set the event so that replacer thread can start working.
        self.continue_replacing()
        return self

    def pause_replacing(self):
        self.replacer_event.clear()

    def continue_replacing(self):
        self.replacer_event.set()

    def __next__(self) -> InputBatch:
        if self.mode in [DataLoader.Mode.VALIDATION, DataLoader.Mode.TEST]:
            if self.iternum >= len(self):
                # Clear the event to put replacer thread on hold.
                self.pause_replacing()
                raise StopIteration

        width, height = self.resolution
        if self.mode == DataLoader.Mode.TRAINING:
            ray_indices = torch.randint(
                0,
                self.buffer_size * self.num_pixels_per_camera,
                size=(self.batch_size,),
                dtype=torch.int64,
                device=self.device,
            )

            with self.data_lock:
                (
                    ray_origins,
                    ray_directions,
                    rgba,
                    frame_numbers,
                    camera_numbers,
                    minmaxes,
                    ray_masks,
                    distance_per_sample,
                    relative_ray_indices_per_sample,
                ) = self.ray_sampler_func(
                    self.pixel_colors_cpu.view(-1, 4),
                    self.light_mask_cpu.view(-1),
                    self.frame_numbers_cuda,
                    self.camera_numbers_cuda,
                    self.grid_texture_objects_cuda,
                    self.landscape_mode_cuda,
                    ray_indices,
                    self.inverse_krs_cuda,
                    self.camera_origins_cuda,
                    self.aabb,
                    self.occupancy_grid_resolution,
                    width,
                    height,
                    4e-4,
                    self.filter_light_bloom,
                )
        else:
            # Common for testing and validation.
            ray_index_start = self.iternum % self.num_pixels_per_camera
            ray_index_end = min(ray_index_start + self.batch_size, self.num_pixels_per_camera)
            ray_indices = torch.arange(ray_index_start, ray_index_end, dtype=torch.int64, device=self.device)

            image_num = self.iternum // self.num_pixels_per_camera
            camera_number, frame_number = self.render_sequence[image_num]
            buffer_index = image_num % self.buffer_size

            frame_numbers_cuda = torch.tensor([frame_number], dtype=torch.int32, device=self.device)
            camera_numbers_cuda = torch.tensor([camera_number], dtype=torch.int32, device=self.device)
            landscape_mode_cuda = torch.tensor([True], dtype=torch.bool, device=self.device)

            if self.run_replacer_thread:
                # Decrease the number of available slots.
                self.available_slots_sem.acquire()

            with self.data_lock:
                if not self.landscape_mode_cuda[buffer_index]:
                    height, width = self.resolution

                (
                    ray_origins,
                    ray_directions,
                    rgba,
                    frame_numbers,
                    camera_numbers,
                    minmaxes,
                    ray_masks,
                    distance_per_sample,
                    relative_ray_indices_per_sample,
                ) = self.ray_sampler_func(
                    self.pixel_colors_cpu[buffer_index],
                    self.light_mask_cpu[buffer_index].view(-1),
                    frame_numbers_cuda,
                    camera_numbers_cuda,
                    self.grid_texture_objects_cuda[buffer_index : buffer_index + 1],
                    landscape_mode_cuda,
                    ray_indices,
                    self.inverse_krs_cuda[buffer_index : buffer_index + 1],
                    self.camera_origins_cuda[buffer_index : buffer_index + 1],
                    self.aabb,
                    self.occupancy_grid_resolution,
                    width,
                    height,
                    4e-4,
                    self.filter_light_bloom,
                )

            if self.run_replacer_thread:
                if self.available_slots_sem._value % self.num_batches_per_full_image == 0:
                    # Increase the number of empty slots.
                    self.empty_slots_sem.release()

        self.iternum += ray_indices.numel()

        return InputBatch(
            ray_origins=ray_origins.view(-1, 3),  # [torch.float]
            ray_directions=ray_directions.view(-1, 3),  # [torch.float]
            # Minimum and maximum distances to be raymarched along each ray (as in `distance*ray_dir`).
            minmaxes=minmaxes.view(-1, 2),  # [torch.float]
            # RGBA value corresponding to the ray (i.e., ground truth color and alpha information).
            rgba=None if self.mode == DataLoader.Mode.TEST else rgba.view(-1, 4),  # [torch.float]
            # Indicates which rays are masked out (where ray_masks=False) during space pruning.
            # Important during VALIDATION and TEST to produce the full images with the background.
            ray_masks=ray_masks.view(-1, 1),  # [torch.bool]
            # 0-indexed frame indices.
            frame_numbers=frame_numbers.view(-1, 1),  # [torch.int32]
            # 0-indexed camera indices.
            camera_numbers=camera_numbers.view(-1, 1),  # [torch.int32]
            # Unique values in frame_numbers.
            unique_frame_numbers=torch.unique(frame_numbers, sorted=False, return_inverse=False).view(
                -1, 1
            ),  # [torch.int32]
            # When OutputMode.RAYS_AND_SAMPLES is selected, this tensor provides at which distance each sample is
            # positioned for the respective ray.
            # If OutputMode.RAYS is selected, this tensor is empty.
            sample_distances=distance_per_sample.view(-1, 1),  # [torch.float]
            # When OutputMode.RAYS_AND_SAMPLES is selected, it indicates which ray each sample belongs to.
            # If OutputMode.RAYS is selected, this tensor is empty.
            ray_indices=relative_ray_indices_per_sample.view(-1).long(),  # [torch.long]
            width=width,
            height=height,
        )
