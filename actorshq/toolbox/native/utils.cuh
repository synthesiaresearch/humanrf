#include <string>
#include <torch/extension.h>
#include <ATen/ATen.h>

#define CHECK_CONTIGUITY_AND_DEVICE(tensor, device) { check_contiguity_and_device((tensor), (device), __FILE__, __LINE__); }
inline void check_contiguity_and_device(const torch::Tensor& tensor, const torch::DeviceType& device, const char *file, int line)
{
    if (!tensor.is_contiguous())
    {
        auto file_line_info = std::string("\nFile: ") + file + "\nLine: " + std::to_string(line);
        throw std::runtime_error(std::string("Tensor not contiguous: ") + file_line_info);
    }

    if (tensor.device().type() != device)
    {
        auto file_line_info = std::string("\nFile: ") + file + "\nLine: " + std::to_string(line);
        throw std::runtime_error(std::string("Tensor is not on the expected device: ") + file_line_info);
    }
}

template <typename T>
inline int calculate_optimal_block_size(T func)
{
    int min_grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, func, 0, 0);

    return block_size;
}
