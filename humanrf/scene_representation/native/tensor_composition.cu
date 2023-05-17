#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <utils.cuh>


__global__ void compose_tensors_forward_kernel(
    const torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> xyz_features,  // (num_samples, feature_dim)
    const torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> xyt_features,  // (num_samples, feature_dim)
    const torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> yzt_features,  // (num_samples, feature_dim)
    const torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> xzt_features,  // (num_samples, feature_dim)
    const torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> xyzt_vectors,  // (4, finest_resolution, feature_dim)
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> xyzt_coordinates,  // (num_samples, 4)
    const int num_samples,
    const int feature_dim,
    const int finest_resolution,
    torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> output_features  // (num_samples, feature_dim)
)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_samples * feature_dim)
    {
        return;
    }
    const int feature_index = index % feature_dim;
    const int sample_index = index / feature_dim;

    float sampled_vectors[4];
    for (int i = 0; i < 4; ++i)
    {
        // Following routine corresponds to align_corners=True in PyTorch's grid_sample.
        // TensoRF does the same.
        // Cuda's texture fetch linear mode does the same: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#linear-filtering
        // xyzt_coordinates are assumed to be in [0, 1].
        auto coord = xyzt_coordinates[sample_index][i] * finest_resolution - 0.5f;
        auto coord_floor = floorf(coord);
        auto coord_frac = coord - coord_floor;

        int coord0 = fmaxf(coord_floor, 0.0f);
        int coord1 = fminf(coord_floor + 1, finest_resolution - 1);
        auto val0 = xyzt_vectors[i][coord0][feature_index];
        auto val1 = xyzt_vectors[i][coord1][feature_index];
        sampled_vectors[i] = val0 + coord_frac * (val1 - val0);
    }

    // Fetch vectors and multiply with corresponding part to get the result
    auto result = __half2float(xyz_features[sample_index][feature_index]) * sampled_vectors[3] +
                  __half2float(xyt_features[sample_index][feature_index]) * sampled_vectors[2] +
                  __half2float(yzt_features[sample_index][feature_index]) * sampled_vectors[0] +
                  __half2float(xzt_features[sample_index][feature_index]) * sampled_vectors[1];

    output_features[sample_index][feature_index] = __float2half(result);
}

__global__ void compose_tensors_backward_kernel(
    const torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> xyz_features,  // (num_samples, feature_dim)
    const torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> xyt_features,  // (num_samples, feature_dim)
    const torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> yzt_features,  // (num_samples, feature_dim)
    const torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> xzt_features,  // (num_samples, feature_dim)
    const torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> xyzt_vectors,  // (4, finest_resolution, feature_dim)
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> xyzt_coordinates,  // (num_samples, 4)
    const torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> d_output_features,  // (num_samples, feature_dim)
    const int num_samples,
    const int feature_dim,
    const int finest_resolution,
    torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> d_xyz_features,  // (num_samples, feature_dim)
    torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> d_xyt_features,  // (num_samples, feature_dim)
    torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> d_yzt_features,  // (num_samples, feature_dim)
    torch::PackedTensorAccessor<at::Half, 2, torch::RestrictPtrTraits, size_t> d_xzt_features,  // (num_samples, feature_dim)
    torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> d_xyzt_vectors  // (4, finest_resolution, feature_dim)
)
{
    // Memory coalescing
    // If the feature size is 64, 64*2=128bytes makes four global memory read per warp.
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_samples * feature_dim)
    {
        return;
    }
    const int feature_index = index % feature_dim;
    const int sample_index = index / feature_dim;

    float features[] = {
        __half2float(yzt_features[sample_index][feature_index]),
        __half2float(xzt_features[sample_index][feature_index]),
        __half2float(xyt_features[sample_index][feature_index]),
        __half2float(xyz_features[sample_index][feature_index])
    };
    auto d_output = __half2float(d_output_features[sample_index][feature_index]);
    float sampled_vectors[4];
    for (int i = 0; i < 4; ++i)
    {
        // Following routine corresponds to align_corners=True in PyTorch's grid_sample.
        // TensoRF does the same.
        // Cuda's texture fetch linear mode does the same: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#linear-filtering
        // xyzt_coordinates are assumed to be in [0, 1].
        auto coord = xyzt_coordinates[sample_index][i] * finest_resolution - 0.5f;
        auto coord_floor = floorf(coord);
        auto coord_frac = coord - coord_floor;

        int coord0 = fmaxf(coord_floor, 0.0f);
        int coord1 = fminf(coord_floor + 1.0f, finest_resolution - 1);
        auto val0 = xyzt_vectors[i][coord0][feature_index];
        auto val1 = xyzt_vectors[i][coord1][feature_index];
        sampled_vectors[i] = val0 + coord_frac * (val1 - val0);

        auto dval = features[i] * d_output;
        atomicAdd(&d_xyzt_vectors[i][coord0][feature_index], dval * (1 - coord_frac));
        atomicAdd(&d_xyzt_vectors[i][coord1][feature_index], dval * coord_frac);
    }

    d_xyz_features[sample_index][feature_index] = __float2half(sampled_vectors[3] * d_output);
    d_xyt_features[sample_index][feature_index] = __float2half(sampled_vectors[2] * d_output);
    d_yzt_features[sample_index][feature_index] = __float2half(sampled_vectors[0] * d_output);
    d_xzt_features[sample_index][feature_index] = __float2half(sampled_vectors[1] * d_output);
}

torch::Tensor compose_tensors_forward(
    torch::Tensor xyz_features,
    torch::Tensor xyt_features,
    torch::Tensor yzt_features,
    torch::Tensor xzt_features,
    torch::Tensor xyzt_vectors,
    torch::Tensor xyzt_coordinates
)
{
    const auto num_samples = xyz_features.size(0);
    const auto feature_dim = xyz_features.size(1);
    const auto finest_resolution = xyzt_vectors.size(1);

    CHECK_CONTIGUITY_AND_DEVICE(xyz_features, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(xyt_features, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(yzt_features, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(xzt_features, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(xyzt_vectors, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(xyzt_coordinates, torch::kCUDA);

    auto output_features = torch::empty_like(xyz_features);

    auto block_size = calculate_optimal_block_size(compose_tensors_forward_kernel);
    dim3 threads(block_size, 1, 1);
    dim3 blocks((num_samples * feature_dim) / threads.x + 1, 1, 1);

    compose_tensors_forward_kernel<<<blocks, threads>>>(
        xyz_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        xyt_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        yzt_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        xzt_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        xyzt_vectors.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
        xyzt_coordinates.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        num_samples,
        feature_dim,
        finest_resolution,
        output_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>()
    );

    return output_features;
}

std::vector<torch::Tensor> compose_tensors_backward(
    torch::Tensor xyz_features,
    torch::Tensor xyt_features,
    torch::Tensor yzt_features,
    torch::Tensor xzt_features,
    torch::Tensor xyzt_vectors,
    torch::Tensor xyzt_coordinates,
    torch::Tensor d_output_features
)
{
    const auto num_samples = xyz_features.size(0);
    const auto feature_dim = xyz_features.size(1);
    const auto finest_resolution = xyzt_vectors.size(1);

    CHECK_CONTIGUITY_AND_DEVICE(xyz_features, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(xyt_features, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(yzt_features, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(xzt_features, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(xyzt_vectors, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(xyzt_coordinates, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(d_output_features, torch::kCUDA);

    auto d_xyz_features = torch::empty_like(xyz_features);
    auto d_xyt_features = torch::empty_like(xyt_features);
    auto d_yzt_features = torch::empty_like(yzt_features);
    auto d_xzt_features = torch::empty_like(xzt_features);
    auto d_xyzt_vectors = torch::zeros_like(xyzt_vectors);

    auto block_size = calculate_optimal_block_size(compose_tensors_backward_kernel);
    dim3 threads(block_size, 1, 1);
    dim3 blocks((num_samples * feature_dim) / threads.x + 1, 1, 1);

    compose_tensors_backward_kernel<<<blocks, threads>>>(
        xyz_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        xyt_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        yzt_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        xzt_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        xyzt_vectors.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
        xyzt_coordinates.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        d_output_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        num_samples,
        feature_dim,
        finest_resolution,
        d_xyz_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        d_xyt_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        d_yzt_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        d_xzt_features.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
        d_xyzt_vectors.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>()
    );

    return {
        d_xyz_features,
        d_xyt_features,
        d_yzt_features,
        d_xzt_features,
        d_xyzt_vectors
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compose_tensors_forward", &compose_tensors_forward, "compose_tensors_forward");
    m.def("compose_tensors_backward", &compose_tensors_backward, "compose_tensors_backward");
}
