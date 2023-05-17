import numpy as np
import tinycudann as tcnn
import torch

import humanrf.scene_representation.tensor_composition_native as tensor_composition_native


class _tensor_composition_autograd_module(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_features, xyt_features, yzt_features, xzt_features, xyzt_vectors, xyzt_coordinates):
        ctx.save_for_backward(xyz_features, xyt_features, yzt_features, xzt_features, xyzt_vectors, xyzt_coordinates)

        output_features = tensor_composition_native.compose_tensors_forward(
            xyz_features, xyt_features, yzt_features, xzt_features, xyzt_vectors, xyzt_coordinates
        )

        return output_features

    @staticmethod
    def backward(ctx, d_output_features):
        xyz_features, xyt_features, yzt_features, xzt_features, xyzt_vectors, xyzt_coordinates = ctx.saved_tensors

        (
            d_xyz_features,
            d_xyt_features,
            d_yzt_features,
            d_xzt_features,
            d_xyzt_vectors,
        ) = tensor_composition_native.compose_tensors_backward(
            xyz_features,
            xyt_features,
            yzt_features,
            xzt_features,
            xyzt_vectors,
            xyzt_coordinates,
            d_output_features.contiguous(),
        )

        return d_xyz_features, d_xyt_features, d_yzt_features, d_xzt_features, d_xyzt_vectors, None


class Decomposition4D(torch.nn.Module):
    def __init__(
        self,
        ngp_n_levels: int = 16,
        ngp_n_features_per_level: int = 2,
        ngp_log2_hashmap_size: int = 19,
        ngp_base_resolution: int = 32,
        ngp_finest_resolution: int = 2048,
        vectors_finest_resolution: int = 2048,
    ):
        """HumanRF's feature grid representation that uses 4D decomposition via four 3D multi-scale hash grids,
        and four 1D dense grids.

        Args:
            ngp_n_levels (int, optional):
                The number of levels in the 3D multi-scale feature grids. Defaults to 16.
            ngp_n_features_per_level (int, optional):
                Can be 1,2,4 or 8. The final feature dimension will be equal to [ngp_n_features_per_level] * [ngp_n_levels].
                Defaults to 2.
            ngp_log2_hashmap_size (int, optional):
                Each 3D hash grid in the decomposition will have a hash map size of 2^[log2_hashmap_size].
                Defaults to 19.
            ngp_base_resolution (int, optional):
               Resolution of the coarsest level in the 3D multi-scale feature grids. Defaults to 32.
            ngp_finest_resolution (int, optional):
                Resolution of the finest level in the 3D multi-scale feature grids. Defaults to 2048.
            vectors_finest_resolution (int, optional):
                Resolution of the 1D dense grids. Defaults to 2048.
        """
        super().__init__()

        per_level_scale = np.exp(np.log(ngp_finest_resolution / ngp_base_resolution) / (ngp_n_levels - 1))

        feature_size = ngp_n_levels * ngp_n_features_per_level
        self.vectors = torch.nn.Parameter(
            torch.randn((4, vectors_finest_resolution, feature_size), dtype=torch.float) * 0.1
        )
        self.xyz_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": ngp_n_levels,
                "n_features_per_level": ngp_n_features_per_level,
                "log2_hashmap_size": ngp_log2_hashmap_size,
                "base_resolution": ngp_base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        self.xyt_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": ngp_n_levels,
                "n_features_per_level": ngp_n_features_per_level,
                "log2_hashmap_size": ngp_log2_hashmap_size,
                "base_resolution": ngp_base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        self.yzt_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": ngp_n_levels,
                "n_features_per_level": ngp_n_features_per_level,
                "log2_hashmap_size": ngp_log2_hashmap_size,
                "base_resolution": ngp_base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        self.xzt_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": ngp_n_levels,
                "n_features_per_level": ngp_n_features_per_level,
                "log2_hashmap_size": ngp_log2_hashmap_size,
                "base_resolution": ngp_base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

    def forward(self, xyz, times):
        xyzt = torch.cat((xyz, times), axis=-1)
        xyz_features = self.xyz_encoding(xyz)
        xyt_features = self.xyt_encoding(xyzt[..., [0, 1, 3]])
        yzt_features = self.yzt_encoding(xyzt[..., [1, 2, 3]])
        xzt_features = self.xzt_encoding(xyzt[..., [0, 2, 3]])

        result = _tensor_composition_autograd_module.apply(
            xyz_features, xyt_features, yzt_features, xzt_features, self.vectors, xyzt
        )

        return result
