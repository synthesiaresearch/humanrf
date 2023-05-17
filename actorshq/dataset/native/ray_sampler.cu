#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <glm/glm.hpp>
#include <utils.cuh>
#include <string>

__constant__ glm::vec3 kAabb[2];

__device__ glm::vec2 compute_aabb_minmax(
    const glm::vec3& ray_origin,
    const glm::vec3& ray_direction
)
{
    auto inv_ray_direction = 1.0f / ray_direction;
    auto t0 = (kAabb[0] - ray_origin) * inv_ray_direction;
    auto t1 = (kAabb[1] - ray_origin) * inv_ray_direction;

    auto min = glm::min(t0, t1);
    auto max = glm::max(t0, t1);
    auto tmin = glm::max(min.x, glm::max(min.y, min.z));
    auto tmax = glm::min(max.x, glm::min(max.y, max.z));

    return glm::vec2(tmin, tmax);
}

__device__ glm::vec2 compute_occupancy_minmax(
    const glm::vec3& ray_origin,
    const glm::vec3& ray_direction,
    const cudaTextureObject_t grid_texture_object,
    const float step_size
)
{
    const auto minmax = compute_aabb_minmax(ray_origin, ray_direction);
    auto tmin = minmax[0];
    while (tmin < minmax[1])
    {
        const auto current_point = ray_origin + ray_direction * tmin + 0.5f; // With +0.5f convert from [-0.5, 0.5] to [0, 1]
        if (tex3D<float>(grid_texture_object, current_point.x, current_point.y, current_point.z) > 0)
        {
            break;
        }
        tmin += step_size;
    }

    if (tmin < minmax[1])
    {
        // Refine tmin
        auto refine_step_size = -step_size * 0.5f;
        for (int i = 0; i < 5; ++i)
        {
            tmin += refine_step_size;
            const auto current_point = ray_origin + ray_direction * tmin + 0.5f; // With +0.5f convert from [-0.5, 0.5] to [0, 1]
            if (tex3D<float>(grid_texture_object, current_point.x, current_point.y, current_point.z) > 0)
            {
                refine_step_size = -fabsf(refine_step_size) * 0.5;
            }
            else
            {
                refine_step_size = fabsf(refine_step_size) * 0.5;
            }
        }
    }

    auto tmax = minmax[1];
    while (tmax > tmin)
    {
        const auto current_point = ray_origin + ray_direction * tmax + 0.5f; // With +0.5f convert from [-0.5, 0.5] to [0, 1]
        if (tex3D<float>(grid_texture_object, current_point.x, current_point.y, current_point.z) > 0)
        {
            break;
        }
        tmax -= step_size;
    }

    return glm::vec2(tmin, tmax);
}

template <bool kOccupancyMinmax>
__global__ void compute_minmax_kernel(
    const torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> inverse_krs,  // (num_images, 3, 3)
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> camera_origins,  // (num_images, 3)
    const torch::PackedTensorAccessor<bool, 1, torch::RestrictPtrTraits, size_t> landscape_modes,  // (num_images)
    const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> ray_indices,  // (num_rays)
    const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> grid_texture_objects,  // (num_images)
    const int num_rays,
    const int grid_resolution,
    int width,
    int height,
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> sampled_ray_directions,  // (num_rays, 3)
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> sampled_minmaxes,  // (num_rays, 2)
    torch::PackedTensorAccessor<bool, 1, torch::RestrictPtrTraits, size_t> ray_mask  // (num_rays)
)
{
    const int ray_num = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_num >= num_rays)
    {
        return;
    }

    const auto sampled_ray_index = ray_indices[ray_num];
    const int image_number = sampled_ray_index / (width * height);

    // Width and height are fixed for landscape mode. If it's otherwise, swap them.
    if (!landscape_modes[image_number])
    {
        auto temp = width;
        width = height;
        height = temp;
    }

    const float pixel_x = sampled_ray_index % width + 0.5f;
    const float pixel_y = (sampled_ray_index / width) % height + 0.5f;

    const auto ray_origin = *reinterpret_cast<const glm::vec3*>(&camera_origins[image_number][0]);
    const auto ray_direction = glm::normalize(
        *reinterpret_cast<const glm::mat3*>(&inverse_krs[image_number][0][0]) * glm::vec3(pixel_x, pixel_y, 1.0f)
    );

    glm::vec2 minmax;
    if constexpr (kOccupancyMinmax)
    {
        minmax = compute_occupancy_minmax(
            ray_origin,
            ray_direction,
            grid_texture_objects[image_number],
            0.5f / grid_resolution
        );
    }
    else
    {
        minmax = compute_aabb_minmax(
            ray_origin,
            ray_direction
        );
    }

    sampled_ray_directions[ray_num][0] = ray_direction.x;
    sampled_ray_directions[ray_num][1] = ray_direction.y;
    sampled_ray_directions[ray_num][2] = ray_direction.z;

    sampled_minmaxes[ray_num][0] = minmax[0];
    sampled_minmaxes[ray_num][1] = minmax[1];

    ray_mask[ray_num] = minmax[0] < minmax[1];
}

template <bool kOccupancyMinmax>
__global__ void compute_sample_distances_kernel(
    const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> ray_indices,  // (num_rays)
    const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> grid_texture_objects,  // (num_images)
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> sampled_minmaxes,  // (num_rays, 2)
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> ray_origins,  // (num_rays, 3)
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> ray_directions,  // (num_rays, 3)
    const torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> relative_ray_end_index_per_ray,  // (num_rays)
    const torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> relative_ray_indices_per_sample,  // (num_samples)
    const int num_samples,
    const int num_pixels_per_camera,
    const float raymarching_step_size,
    torch::PackedTensorAccessor<float, 1, torch::RestrictPtrTraits, size_t> distance_per_sample,  // (num_samples)
    torch::PackedTensorAccessor<bool, 1, torch::RestrictPtrTraits, size_t> filter_mask_per_sample  // (num_samples)
)
{
    const int sample_num = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_num >= num_samples)
    {
        return;
    }

    const int relative_ray_num = relative_ray_indices_per_sample[sample_num];
    const auto sampled_ray_index = ray_indices[relative_ray_num];
    const int image_number = sampled_ray_index / num_pixels_per_camera;

    auto local_sample_index = sample_num;
    if (relative_ray_num > 0)
    {
        local_sample_index -= relative_ray_end_index_per_ray[relative_ray_num - 1];
    }
    const auto current_distance = sampled_minmaxes[relative_ray_num][0] + local_sample_index * raymarching_step_size;
    distance_per_sample[sample_num] = current_distance;

    if constexpr (kOccupancyMinmax)
    {
        const auto ray_origin = *reinterpret_cast<const glm::vec3*>(&ray_origins[relative_ray_num][0]);
        const auto ray_direction = *reinterpret_cast<const glm::vec3*>(&ray_directions[relative_ray_num][0]);
        const auto current_point = ray_origin + ray_direction * current_distance + 0.5f; // With +0.5f convert from [-0.5, 0.5] to [0, 1]
        filter_mask_per_sample[sample_num] = tex3D<float>(grid_texture_objects[image_number], current_point.x, current_point.y, current_point.z) > 0;
    }
    else
    {
        filter_mask_per_sample[sample_num] = true;
    }
}

template <bool kOccupancyMinmax, bool kGetSamples>
std::vector<torch::Tensor> get_data(
    torch::Tensor rgba,
    torch::Tensor light_mask,
    torch::Tensor frame_numbers,
    torch::Tensor camera_numbers,
    torch::Tensor grid_texture_objects,
    torch::Tensor landscape_modes,
    torch::Tensor all_ray_indices,
    torch::Tensor inverse_krs,
    torch::Tensor camera_origins,
    torch::Tensor aabb,
    const int grid_resolution,
    const int image_width,
    const int image_height,
    const float raymarching_step_size,
    const bool filter_light_bloom
)
{
    CHECK_CONTIGUITY_AND_DEVICE(rgba, torch::kCPU);
    CHECK_CONTIGUITY_AND_DEVICE(light_mask, torch::kCPU);
    CHECK_CONTIGUITY_AND_DEVICE(frame_numbers, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(camera_numbers, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(grid_texture_objects, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(landscape_modes, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(all_ray_indices, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(inverse_krs, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(camera_origins, torch::kCUDA);
    CHECK_CONTIGUITY_AND_DEVICE(aabb, torch::kCUDA);

    cudaMemcpyToSymbol(kAabb, aabb.data_ptr<float>(), 2 * sizeof(glm::vec3));

    int num_rays = all_ray_indices.size(0);
    const auto cuda_float_options = aabb.options().dtype(torch::kFloat);
    const auto cuda_int_options = aabb.options().dtype(torch::kInt);
    const auto cuda_bool_options = aabb.options().dtype(torch::kBool);

    auto all_sampled_ray_directions = torch::empty({num_rays, 3}, cuda_float_options);
    auto all_sampled_minmaxes = torch::empty({num_rays, 2}, cuda_float_options);
    auto ray_mask = torch::empty({num_rays}, cuda_bool_options);

    auto block_size = calculate_optimal_block_size(compute_minmax_kernel<kOccupancyMinmax>);
    dim3 threads(block_size, 1, 1);
    dim3 blocks(num_rays / threads.x + 1, 1, 1);
    compute_minmax_kernel<kOccupancyMinmax><<<blocks, threads>>>(
        inverse_krs.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
        camera_origins.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        landscape_modes.packed_accessor<bool, 1, torch::RestrictPtrTraits, size_t>(),
        all_ray_indices.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
        grid_texture_objects.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
        num_rays,
        grid_resolution,
        image_width,
        image_height,
        all_sampled_ray_directions.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        all_sampled_minmaxes.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        ray_mask.packed_accessor<bool, 1, torch::RestrictPtrTraits, size_t>()
    );
    if (filter_light_bloom)
    {
        ray_mask = torch::bitwise_and(ray_mask, torch::bitwise_not(light_mask.index({all_ray_indices.cpu()}).cuda()));
    }
    auto ray_indices = all_ray_indices.index({ray_mask});
    auto sampled_ray_directions = all_sampled_ray_directions.index({ray_mask});
    auto sampled_minmaxes = all_sampled_minmaxes.index({ray_mask});

    auto sampled_rgba = (rgba.index({ray_indices.cpu()}) / 255.0f).cuda();
    auto image_numbers = torch::floor_divide(ray_indices, image_width * image_height);
    auto sampled_ray_origins = camera_origins.index({image_numbers});
    auto sampled_frame_numbers = frame_numbers.index({image_numbers});
    auto sampled_camera_numbers = camera_numbers.index({image_numbers});

    if constexpr (!kGetSamples)
    {
        return {
            sampled_ray_origins,
            sampled_ray_directions,
            sampled_rgba,
            sampled_frame_numbers,
            sampled_camera_numbers,
            sampled_minmaxes,
            ray_mask,
            torch::empty({0}, cuda_float_options),
            torch::empty({0}, cuda_int_options)
        };
    }

    auto sampled_counts = ((
        sampled_minmaxes.index({"...", 1}) - sampled_minmaxes.index({"...", 0})
    ) / raymarching_step_size).to(torch::kInt);

    num_rays = ray_indices.size(0);
    auto relative_ray_end_index_per_ray = sampled_counts.cumsum(0, torch::kInt);
    auto relative_ray_indices_per_sample = torch::arange(0, num_rays, cuda_int_options).repeat_interleave(sampled_counts);
    const auto num_samples = relative_ray_indices_per_sample.size(0);

    auto distance_per_sample = torch::empty({num_samples}, cuda_float_options);
    auto filter_mask_per_sample = torch::empty({num_samples}, cuda_bool_options);

    block_size = calculate_optimal_block_size(compute_sample_distances_kernel<kOccupancyMinmax>);
    threads = dim3(block_size, 1, 1);
    blocks = dim3(num_samples / threads.x + 1, 1, 1);

    compute_sample_distances_kernel<kOccupancyMinmax><<<blocks, threads>>>(
        ray_indices.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
        grid_texture_objects.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
        sampled_minmaxes.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        sampled_ray_origins.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        sampled_ray_directions.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        relative_ray_end_index_per_ray.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
        relative_ray_indices_per_sample.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
        num_samples,
        image_width * image_height,
        raymarching_step_size,
        distance_per_sample.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>(),
        filter_mask_per_sample.packed_accessor<bool, 1, torch::RestrictPtrTraits, size_t>()
    );

    return {
        sampled_ray_origins,
        sampled_ray_directions,
        sampled_rgba,
        sampled_frame_numbers,
        sampled_camera_numbers,
        sampled_minmaxes,
        ray_mask,
        distance_per_sample.index({filter_mask_per_sample}),
        relative_ray_indices_per_sample.index({filter_mask_per_sample})
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_rays_aabb_minmax", &get_data<false, false>, "get_rays_aabb_minmax");
    m.def("get_rays_occupancy_minmax", &get_data<true, false>, "get_rays_occupancy_minmax");
    m.def("get_samples_aabb_minmax", &get_data<false, true>, "get_samples_aabb_minmax");
    m.def("get_samples_occupancy_minmax", &get_data<true, true>, "get_samples_occupancy_minmax");
}
