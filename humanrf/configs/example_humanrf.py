import actorshq.evaluation.presets as presets

frame_configs = presets.frame_configs["siggraph_interval_1"]

config = [
    # fmt: off
    "--train", "true",
    "--evaluate", "true",
    "--test.trajectory_via_keycams", "34", "126", "90",

    "--model.log2_hashmap_size", "19",
    "--model.n_features_per_level", "2",
    "--model.n_levels", "16",
    "--model.coarsest_resolution", "32",
    "--model.finest_resolution", "2048",

    "--model.temporal_partitioning", "adaptive",
    "--model.expansion_factor_threshold", "1.25",
    "--model.camera_embedding_dim", "2",  # This is set to "0" for the numerical comparisons in the paper.

    "--training.max_steps", "50_001",
    "--training.scaler_growth_interval", "100_000",
    "--training.samples_max_batch_size", "640_000",
    "--validation.repeat_cameras", "2",
    "--validation.every_n_steps", "2_500",

    "--training.camera_preset", "siggraph_train",
    "--validation.camera_preset", "siggraph_train_validation",
    "--evaluation.camera_preset", "siggraph_test",
    "--evaluation.coverage", "siggraph_test",

    "--dataset.actor", "Actor01",
    "--dataset.sequence", "Sequence1",
    "--dataset.scale", "4",
    "--dataset.crop_center_square", "true",
    "--dataset.filter_light_bloom", "false",  # Set "true" to avoid light bleeding into the actor.
    "--dataset.frame_numbers", *[str(i) for i in range(*frame_configs)],
    # fmt: on
]
