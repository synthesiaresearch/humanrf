from typing import List, Tuple

import numpy as np
import tinycudann as tcnn
import torch

from humanrf.adaptive_temporal_partitioning import PREDEFINED_SEGMENT_SIZES
from humanrf.scene_representation.decomposition4d import Decomposition4D
from humanrf.scene_representation.query_io import QueryInput, QueryOutput
from humanrf.utils.activation import truncated_exp


class HumanRF(torch.nn.Module):
    def __init__(
        self,
        density_scale: float,
        sorted_frame_numbers: Tuple[int, ...],
        n_features_per_level: int,
        log2_hashmap_size: int,
        n_levels: int,
        coarsest_resolution: int,
        finest_resolution: int,
        geometry_feature_dim: int,
        n_neurons: int,
        n_hidden_layers_density: int,
        n_hidden_layers_color: int,
        sh_degree: int,
        segment_sizes: Tuple[int, ...],
        camera_embedding_dim: int,
        **kwargs
    ):
        """HumanRF model that takes adaptively partitioned sequences as input, and represent each segment by a feature
        grid that utilizes essentially a 4D decomposition.

        Args:
            density_scale (float):
                Density output is scaled with this value, values around 100 improves the convergence rate.
            sorted_frame_numbers (Tuple[int, ...]):
                Frame numbers sorted in ascending order.
            n_features_per_level (int):
                Can be 1,2,4 or 8. The final feature dimension will be equal to [n_features_per_level] * [n_levels].
            log2_hashmap_size (int):
                There are four 3D multi-scale feature grids in the 4D feature grid decomposition, and each will have a
                hash map size of 2^[log2_hashmap_size].
            n_levels (int):
                The number of levels in the 3D multi-scale feature grids.
            coarsest_resolution (int):
                Resolution of the coarsest level in the 3D multi-scale feature grids.
            finest_resolution (int):
                Resolution of the finest level in the 3D multi-scale feature grids.
            geometry_feature_dim (int):
                Sigma MLP outputs [geometry_feature_dim] + 1 dimensional features where the last [geometry_feature_dim]
                dimensional part is fed into the color MLP to produce RGB color values and the first dimension is used
                to regress density.
            n_neurons (int):
                Number of neurons to use in the sigma and color MLPs.
            n_hidden_layers_density (int):
                Number of hidden layers to be used in the sigma MLP.
            n_hidden_layers_color (int):
                Number of hidden layers to be used in the color MLP.
            sh_degree (int):
                Degrees of spherical harmonics encoding for viewing directions, where resulting encodings are
                [sh_degree]^2 dimensional.
            segment_sizes (Tuple[int, ...]):
                Output of adaptive temporal partitioning that provides per-segment sizes.
            camera_embedding_dim (int):
                In some cases, camera embeddings help with the brightness inconsistencies across different cameras.
                Setting to 2 has worked well for ActorsHQ dataset.
        """
        super().__init__()

        self.density_scale = density_scale
        self.num_frames = len(sorted_frame_numbers)
        self.num_segments = len(segment_sizes)
        self.camera_embedding_dim = camera_embedding_dim
        if camera_embedding_dim > 0:
            self.camera_embeddings = torch.nn.Embedding(160, camera_embedding_dim)

        segment_end_frame_index = np.cumsum(segment_sizes, dtype=np.int32)
        segment_end_frame_index[-1] = min(segment_end_frame_index[-1], self.num_frames)
        segment_start_frame_index = np.concatenate((np.zeros(1, dtype=np.int32), segment_end_frame_index[:-1]))
        segments_to_frame_numbers = {
            segment_number: [
                sorted_frame_numbers[j]
                for j in range(segment_start_frame_index[segment_number], segment_end_frame_index[segment_number])
            ]
            for segment_number in range(self.num_segments)
        }
        frame_numbers_to_segment_numbers = np.full((sorted_frame_numbers[-1] + 1), fill_value=-1, dtype=np.int32)
        frame_numbers_to_normalized_local_frame_numbers = np.full(
            (sorted_frame_numbers[-1] + 1), fill_value=-1, dtype=np.float32
        )
        for segment_number in segments_to_frame_numbers:
            segment_num_frames = len(segments_to_frame_numbers[segment_number])
            for local_frame_number, frame_number in enumerate(segments_to_frame_numbers[segment_number]):
                frame_numbers_to_segment_numbers[frame_number] = segment_number
                frame_numbers_to_normalized_local_frame_numbers[frame_number] = local_frame_number / segment_num_frames

        self.register_buffer("frame_numbers_to_segment_numbers", torch.from_numpy(frame_numbers_to_segment_numbers))
        self.register_buffer(
            "frame_numbers_to_normalized_local_frame_numbers",
            torch.from_numpy(frame_numbers_to_normalized_local_frame_numbers),
        )

        self.feature_grids = torch.nn.ModuleList()
        for segment_size in segment_sizes:
            segment_log2_hashmap_size = int(
                np.round(np.log2((segment_size / max(PREDEFINED_SEGMENT_SIZES) * (2**log2_hashmap_size))))
            )

            self.feature_grids.append(
                Decomposition4D(
                    ngp_n_levels=n_levels,
                    ngp_n_features_per_level=n_features_per_level,
                    ngp_log2_hashmap_size=segment_log2_hashmap_size,
                    ngp_base_resolution=coarsest_resolution,
                    ngp_finest_resolution=finest_resolution,
                    vectors_finest_resolution=finest_resolution,
                )
            )

        self.total_feature_dim = n_levels * n_features_per_level
        self.sigma_net = tcnn.Network(
            n_input_dims=self.total_feature_dim,
            n_output_dims=1 + geometry_feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": n_hidden_layers_density,
            },
        )

        self.color_net = tcnn.NetworkWithInputEncoding(
            n_input_dims=3 + geometry_feature_dim + camera_embedding_dim,
            n_output_dims=3,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": sh_degree,
                    },
                    {"otype": "Identity"},
                ],
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": n_neurons,
                "n_hidden_layers": n_hidden_layers_color,
            },
        )

    def density(self, query_input: QueryInput) -> QueryOutput:
        frame_numbers = query_input.frame_numbers.squeeze(1).long()
        unique_frame_numbers = query_input.unique_frame_numbers.squeeze(1).long()

        unique_segments = np.unique(self.frame_numbers_to_segment_numbers[unique_frame_numbers].cpu().numpy()).tolist()
        segments = self.frame_numbers_to_segment_numbers[frame_numbers]

        num_samples = query_input.positions.shape[0]
        features = torch.empty(
            (num_samples, self.total_feature_dim), dtype=torch.half, device=query_input.positions.device
        )
        for segment_number in range(self.num_segments):
            if segment_number in unique_segments:
                self.feature_grids[segment_number].to(device="cuda", non_blocking=True)
                segment_mask = segments == segment_number
                features[segment_mask] = self.feature_grids[segment_number](
                    # query_input.positions has value range [-0.5, 0.5].
                    query_input.positions[segment_mask] + 0.5,
                    self.frame_numbers_to_normalized_local_frame_numbers[frame_numbers[segment_mask]].unsqueeze(-1),
                )
            else:
                self.feature_grids[segment_number].to(device="cpu", non_blocking=True)

        h = self.sigma_net(features)

        return QueryOutput(
            density=truncated_exp(h[..., 0]) * self.density_scale,
            geometry_features=h[..., 1:],
        )

    def forward(self, query_input: QueryInput) -> QueryOutput:
        output = self.density(query_input)

        # query_input.directions has value range [-1, 1]. To query the color_net this is transformed to [0, 1].
        color_net_input = [(query_input.directions + 1) * 0.5, output.geometry_features]
        if self.camera_embedding_dim > 0:
            if query_input.is_training:
                color_net_input.append(self.camera_embeddings(query_input.camera_numbers.squeeze(1).long()))
            else:
                # Using zeros during validation&test time.
                color_net_input.append(
                    torch.zeros(
                        (query_input.directions.shape[0], self.camera_embedding_dim),
                        dtype=query_input.directions.dtype,
                        device=query_input.directions.device,
                    )
                )

        output.radiance = self.color_net(torch.cat(color_net_input, dim=-1))

        return output

    def get_params(self, lr):
        params = [
            {'params': self.feature_grids.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]

        if self.camera_embedding_dim > 0:
            params.append({'params': self.camera_embeddings.parameters(), 'lr': lr})

        return params
