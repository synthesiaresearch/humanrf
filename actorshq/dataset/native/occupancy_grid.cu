#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <utils.cuh>

class OccupanyGrid
{
    public:
        OccupanyGrid(const unsigned long grid_resolution, const int buffer_size)
            : m_grid_resolution(grid_resolution)
            , m_grid_extent({grid_resolution, grid_resolution, grid_resolution})
            , m_buffer_size(buffer_size)
            , m_available_buffer_index(0)
        {
            for (int i = 0; i < m_buffer_size; ++i)
            {
                cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
                cudaArray_t grid_array = nullptr;
                cudaMalloc3DArray(&grid_array, &channel_desc, m_grid_extent);

                cudaResourceDesc resource_desc;
                memset(&resource_desc, 0, sizeof(resource_desc));
                resource_desc.resType = cudaResourceTypeArray;
                resource_desc.res.array.array = grid_array;

                cudaTextureDesc texture_desc;
                memset(&texture_desc, 0, sizeof(texture_desc));
                texture_desc.addressMode[0] = cudaAddressModeClamp;
                texture_desc.addressMode[1] = cudaAddressModeClamp;
                texture_desc.addressMode[2] = cudaAddressModeClamp;
                texture_desc.filterMode = cudaFilterModeLinear;
                texture_desc.readMode = cudaReadModeNormalizedFloat;
                texture_desc.normalizedCoords = 1;

                cudaTextureObject_t texture_object = 0;
                cudaCreateTextureObject(&texture_object, &resource_desc, &texture_desc, nullptr);

                m_grid_arrays.push_back(grid_array);
                m_texture_objects.push_back(texture_object);
            }
        }

        ~OccupanyGrid()
        {
            for (const auto& m_grid_array : m_grid_arrays)
            {
                cudaFreeArray(m_grid_array);
            }
            for (const auto& m_texture_object : m_texture_objects)
            {
                cudaDestroyTextureObject(m_texture_object);
            }
        }

        cudaTextureObject_t add_grid(torch::Tensor grid)
        {
            CHECK_CONTIGUITY_AND_DEVICE(grid, torch::kCUDA);
            if (grid.size(0) != m_grid_resolution || grid.size(1) != m_grid_resolution || grid.size(2) != m_grid_resolution)
            {
                throw std::runtime_error("Provided grid doesn't have the correct resolution!");
            }

            auto used_buffer_index = m_available_buffer_index;
            m_available_buffer_index = (m_available_buffer_index + 1) % m_buffer_size;

            cudaMemcpy3DParms memcpy_params = {0};
            memcpy_params.srcPtr.pitch = m_grid_resolution * sizeof(uint8_t);
            memcpy_params.srcPtr.xsize = m_grid_resolution;
            memcpy_params.srcPtr.ysize = m_grid_resolution;
            memcpy_params.srcPtr.ptr = grid.data_ptr<uint8_t>();
            memcpy_params.dstArray = m_grid_arrays[used_buffer_index];
            memcpy_params.extent = m_grid_extent;
            memcpy_params.kind = cudaMemcpyDefault;
            cudaMemcpy3D(&memcpy_params);

            return m_texture_objects[used_buffer_index];
        }

    private:
        const unsigned long m_grid_resolution;
        const int m_buffer_size;
        int m_available_buffer_index;
        cudaExtent m_grid_extent;
        std::vector<cudaTextureObject_t> m_texture_objects;
        std::vector<cudaArray_t> m_grid_arrays;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<OccupanyGrid>(m, "OccupanyGrid")
        .def(py::init<const unsigned long, const int>())
        .def("add_grid", &OccupanyGrid::add_grid);
}
