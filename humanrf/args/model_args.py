from dataclasses import dataclass

from simple_parsing import field


@dataclass
class _shallow_mlp_args:
    # sigma MLP outputs [geometry_feature_dim] + 1 dimensional features where the last [geometry_feature_dim] dimensional
    # part is fed into the color MLP to produce RGB color values and the first dimension is used to regress density.
    geometry_feature_dim: int = 15
    # number of neurons to use in the sigma and color MLPs.
    n_neurons: int = 64
    # number of hidden layers to be used in the sigma MLP.
    n_hidden_layers_density: int = 1
    # number of hidden layers to be used in the color MLP.
    n_hidden_layers_color: int = 2
    # degrees of spherical harmonics encoding for viewing directions, where resulting encodings are [sh_degree]^2
    # dimensional.
    sh_degree: int = 4


@dataclass
class _decomposition4d_args:
    # there are four 3D multi-scale feature grids in the 4D feature grid decomposition, and each will have a hash map
    # size of 2^[log2_hashmap_size].
    log2_hashmap_size: int = 19
    # can be 1,2,4 or 8.
    # the final feature dimension will be equal to [n_features_per_level] * [n_levels].
    n_features_per_level: int = 2
    # the number of levels in the 3D multi-scale feature grids.
    n_levels: int = 16
    # resolution of the coarsest level in the 3D multi-scale feature grids.
    coarsest_resolution: int = 32
    # resolution of the finest level in the 3D multi-scale feature grids.
    finest_resolution: int = 2048


@dataclass
class _model_args(_shallow_mlp_args, _decomposition4d_args):
    # strategy to partition the temporal domain.
    temporal_partitioning: str = field(default="adaptive", choices=["adaptive", "fixed", "none"])
    # determines the frequency of spawning new segments if "adaptive" is selected for temporal_partitioning.
    expansion_factor_threshold: float = 1.25
    # determines the fixed size of the temporal segments if "fixed" is selected for temporal_partitioning.
    fixed_segment_size: int = 12
    # density output is scaled with this value, values around 100 improves the convergence rate.
    density_scale: int = 100
    # camera embeddings dimensionality.
    # in some cases, camera embeddings help with the brightness inconsistencies across different cameras.
    camera_embedding_dim: int = 0
