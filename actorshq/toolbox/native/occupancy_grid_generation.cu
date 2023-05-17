#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <glm/glm.hpp>
#include <utils.cuh>


// Very important: glm stores matrices in column-major format!
constexpr int kMaxNumCameras = 160;
__constant__ glm::mat4 kProjectionMatrices[kMaxNumCameras];
__constant__ bool kLandscapeModes[kMaxNumCameras];


__global__ void generate_from_masks_kernel(
    const torch::PackedTensorAccessor<uint8_t, 2, torch::RestrictPtrTraits, size_t> masks,  // (num_cameras, H*W)
    const int camera_coverage_threshold,
    const int num_cameras,
    const int grid_resolution,
    const int width,
    const int height,
    torch::PackedTensorAccessor<uint8_t, 3, torch::RestrictPtrTraits, size_t> occupancy_grid  // (D, H, W)
)
{
    const int voxel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_index >= grid_resolution * grid_resolution * grid_resolution)
    {
        return;
    }

    const int grid_x = voxel_index % grid_resolution;
    const int grid_y = (voxel_index / grid_resolution) % grid_resolution;
    const int grid_z = (voxel_index / grid_resolution) / grid_resolution;

    // Scene resides in [-0.5, 0.5]
    glm::vec4 voxel_coord(glm::vec3(grid_x, grid_y, grid_z) / static_cast<float>(grid_resolution - 1) - 0.5f, 1.0f);

    int num_covered_cameras = 0;
    bool in_convex_hull = false;
    for (int camera_number = 0; camera_number < num_cameras; ++camera_number)
    {
        const auto landscape_mode = kLandscapeModes[camera_number];
        const auto camera_width = landscape_mode ? width : height;
        const auto camera_height = landscape_mode ? height : width;

        auto projected_point = kProjectionMatrices[camera_number] * voxel_coord;
        const int pixel_x = projected_point.x / projected_point.z;
        const int pixel_y = projected_point.y / projected_point.z;

        if (pixel_x >= 0 && pixel_x < camera_width && pixel_y >= 0 && pixel_y < camera_height)
        {
            const int pixel_x1 = min(pixel_x + 1, camera_width - 1);
            const int pixel_y1 = min(pixel_y + 1, camera_height - 1);
            if (
                masks[camera_number][pixel_x + pixel_y * camera_width] == 0 &&
                masks[camera_number][pixel_x1 + pixel_y * camera_width] == 0 &&
                masks[camera_number][pixel_x + pixel_y1 * camera_width] == 0 &&
                masks[camera_number][pixel_x1 + pixel_y1 * camera_width] == 0
            )
            {
                const int num_rest_cameras = num_cameras - camera_number - 1;
                if (num_covered_cameras + num_rest_cameras < camera_coverage_threshold)
                {
                    break;
                }
            }
            else
            {
                ++num_covered_cameras;
                in_convex_hull = num_covered_cameras >= camera_coverage_threshold;
                if (in_convex_hull)
                {
                    break;
                }
            }
        }
    }

    occupancy_grid[grid_z][grid_y][grid_x] = in_convex_hull ? 255 : 0;
}

torch::Tensor generate_from_masks(
    torch::Tensor masks,
    torch::Tensor projection_matrices,
    torch::Tensor landscape_modes,
    const int camera_coverage_threshold,
    const int grid_resolution,
    const int width,
    const int height
)
{
    if (masks.size(1) != width * height)
    {
        throw std::runtime_error("The number mask entries per camera has to be equal to width*height!");
    }
    CHECK_CONTIGUITY_AND_DEVICE(masks, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(projection_matrices, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(landscape_modes, torch::kCUDA);
    const int num_cameras = projection_matrices.size(0);
    cudaMemcpyToSymbol(kProjectionMatrices, projection_matrices.data_ptr<float>(), num_cameras * sizeof(glm::mat4));
    cudaMemcpyToSymbol(kLandscapeModes, landscape_modes.data_ptr<bool>(), num_cameras * 1 * sizeof(bool));

    auto occupancy_grid = torch::empty({grid_resolution, grid_resolution, grid_resolution}, masks.options().dtype(torch::kUInt8));

    auto block_size = calculate_optimal_block_size(generate_from_masks_kernel);
    dim3 threads(block_size, 1, 1);
    dim3 blocks(grid_resolution * grid_resolution * grid_resolution / threads.x + 1, 1, 1);
    generate_from_masks_kernel<<<blocks, threads>>>(
        masks.packed_accessor<uint8_t, 2, torch::RestrictPtrTraits, size_t>(),
        camera_coverage_threshold,
        num_cameras,
        grid_resolution,
        width,
        height,
        occupancy_grid.packed_accessor<uint8_t, 3, torch::RestrictPtrTraits, size_t>()
    );

    return occupancy_grid;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generate_from_masks", &generate_from_masks, "generate_from_masks");
}
