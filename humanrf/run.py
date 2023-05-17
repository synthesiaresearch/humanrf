#!/usr/bin/env python3
import os
import random

import numpy as np
import torch
import yaml

import actorshq.evaluation.presets as presets
from actorshq.dataset.data_loader import DataLoader
from actorshq.dataset.trajectory import (
    get_trajectory_dataloader_from_calibration,
    get_trajectory_dataloader_from_keycams,
)
from actorshq.dataset.volumetric_dataset import VolumetricDataset
from actorshq.evaluation.evaluate import evaluate
from humanrf.adaptive_temporal_partitioning import compute_adaptive_segment_sizes
from humanrf.args.run_args import parse_args
from humanrf.scene_representation.humanrf import HumanRF
from humanrf.trainer import Trainer
from humanrf.utils.memory import collect_and_free_memory

if __name__ == "__main__":
    config = parse_args()

    # Set the seed for each possible source of random numbers.
    random.seed(config.random_seed)
    os.environ["PYTHONHASHSEED"] = str(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)

    frame_numbers = config.dataset.frame_numbers
    frame_range = (min(frame_numbers), max(frame_numbers) + 1)

    workspace = config.workspace
    workspace.mkdir(parents=True, exist_ok=True)

    with open(workspace / "config.yaml", "w") as f:
        yaml.dump(config, f)

    data_folder = config.dataset.path / config.dataset.actor / config.dataset.sequence / f"{config.dataset.scale}x"

    if config.model.temporal_partitioning == "none":
        segment_sizes = [len(frame_numbers)]
    elif config.model.temporal_partitioning == "adaptive":
        segment_sizes = compute_adaptive_segment_sizes(
            dataset=VolumetricDataset(data_folder),
            sorted_frame_numbers=frame_numbers,
            expansion_factor_threshold=config.model.expansion_factor_threshold,
        )
    elif config.model.temporal_partitioning == "fixed":
        fixed_size = config.model.fixed_segment_size
        segment_sizes = [fixed_size for _ in range(int(np.ceil(len(frame_numbers) / fixed_size)))]
    else:
        raise NotImplementedError("Unknown temporal partitioning type!")

    inputs = {
        "sorted_frame_numbers": tuple(sorted(frame_numbers)),
        "segment_sizes": tuple(segment_sizes),
        **vars(config.model),
    }
    model = HumanRF(**inputs)

    if config.train:
        training_data_loader = DataLoader(
            dataset=VolumetricDataset(data_folder, config.dataset.crop_center_square),
            device=config.device,
            mode=DataLoader.Mode.TRAINING,
            dataloader_output_mode=DataLoader.OutputMode.RAYS_AND_SAMPLES,
            space_pruning_mode=DataLoader.SpacePruningMode.OCCUPANCY_GRID,
            batch_size=config.training.rays_initial_batch_size,
            camera_numbers=presets.camera_configs[config.training.camera_preset],
            frame_numbers=frame_numbers,
            max_buffer_size=config.dataset.max_buffer_size,
            max_num_frames_per_batch=config.dataset.max_num_frames_per_batch,
            use_mask=True,
            filter_light_bloom=config.dataset.filter_light_bloom,
        )
        render_sequence_validation = presets.get_render_sequence(
            coverage=config.validation.coverage,
            camera_preset=config.validation.camera_preset,
            frame_numbers=frame_numbers,
            repeat_cameras=config.validation.repeat_cameras,
        )
        validation_data_loader = DataLoader(
            dataset=VolumetricDataset(data_folder, config.dataset.crop_center_square),
            device=config.device,
            mode=DataLoader.Mode.VALIDATION,
            dataloader_output_mode=DataLoader.OutputMode.RAYS_AND_SAMPLES,
            space_pruning_mode=DataLoader.SpacePruningMode.OCCUPANCY_GRID,
            batch_size=config.validation.rays_batch_size,
            camera_numbers=presets.camera_configs[config.validation.camera_preset],
            frame_numbers=frame_numbers,
            max_buffer_size=1,
            use_mask=True,
            filter_light_bloom=config.dataset.filter_light_bloom,
            render_sequence=render_sequence_validation,
        )

        optimizer = torch.optim.Adam(model.get_params(config.training.lr), betas=(0.9, 0.99), eps=1e-15)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: config.training.lr_decay ** min(step / config.training.max_steps, 1)
        )

        trainer = Trainer(
            config=config,
            workspace=workspace,
            checkpoint=config.training.checkpoint,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        trainer.train(training_data_loader, validation_data_loader, max_steps=config.training.max_steps)
        collect_and_free_memory()

    results_folder = workspace / "results"

    if config.test.trajectory_via_keycams is not None:
        trajectory_data_loader = get_trajectory_dataloader_from_keycams(
            trajectory=config.test.trajectory_via_keycams,
            base_data_folder=data_folder,
            device=config.device,
            dataloader_output_mode=DataLoader.OutputMode.RAYS_AND_SAMPLES,
            space_pruning_mode=DataLoader.SpacePruningMode.OCCUPANCY_GRID,
            batch_size=config.test.rays_batch_size,
            frame_numbers=frame_numbers,
            trajectory_num_cameras=config.test.trajectory_num_cameras,
        )

        trainer = Trainer(
            config=config,
            workspace=workspace,
            checkpoint=config.test.checkpoint,
            model=model,
            optimizer=None,
            lr_scheduler=None,
        )
        trainer.test(trajectory_data_loader, results_folder / "test_keycams", True)
        collect_and_free_memory()

    if config.test.trajectory_via_calibration_file is not None:
        trajectory_data_loader = get_trajectory_dataloader_from_calibration(
            calibration_path=config.test.trajectory_via_calibration_file,
            base_data_folder=data_folder,
            device=config.device,
            dataloader_output_mode=DataLoader.OutputMode.RAYS_AND_SAMPLES,
            space_pruning_mode=DataLoader.SpacePruningMode.OCCUPANCY_GRID,
            batch_size=config.test.rays_batch_size,
            frame_numbers=frame_numbers,
        )

        trainer = Trainer(
            config=config,
            workspace=workspace,
            checkpoint=config.test.checkpoint,
            model=model,
            optimizer=None,
            lr_scheduler=None,
        )
        trainer.test(trajectory_data_loader, results_folder / "test_calibration_file", True)
        collect_and_free_memory()

    if config.evaluate:
        if config.evaluation.frame_numbers is not None:
            frame_numbers = config.evaluation.frame_numbers

        render_sequence_evaluation = presets.get_render_sequence(
            coverage=config.evaluation.coverage,
            camera_preset=config.evaluation.camera_preset,
            frame_numbers=frame_numbers,
        )
        evaluation_data_loader = DataLoader(
            dataset=VolumetricDataset(data_folder, crop_center_square=False),
            device=config.device,
            mode=DataLoader.Mode.TEST,
            dataloader_output_mode=DataLoader.OutputMode.RAYS_AND_SAMPLES,
            space_pruning_mode=DataLoader.SpacePruningMode.OCCUPANCY_GRID,
            batch_size=config.test.rays_batch_size,
            camera_numbers=presets.camera_configs[config.evaluation.camera_preset],
            frame_numbers=frame_numbers,
            max_buffer_size=1,
            render_sequence=render_sequence_evaluation,
        )

        trainer = Trainer(
            config=config,
            workspace=workspace,
            checkpoint=config.test.checkpoint,
            model=model,
            optimizer=None,
            lr_scheduler=None,
        )
        trainer.test(evaluation_data_loader, results_folder / "test_frames", False)

        evaluate(
            results_directory=results_folder,
            output_directory=results_folder,
            coverage=config.evaluation.coverage,
            camera_preset=config.evaluation.camera_preset,
            frame_numbers=frame_numbers,
            data_folder=data_folder,
            result_suffix=".png",
        )
