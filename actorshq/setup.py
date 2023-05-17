import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_util_path = str(Path(os.path.realpath(__file__)).parent / "toolbox" / "native")

setup(
    name="actorshq",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            name="actorshq.dataset.ray_sampler_native",
            sources=["dataset/native/ray_sampler.cu"],
            include_dirs=[_util_path],
            extra_compile_args={"cxx": [""], "nvcc": ["--use_fast_math"]},
        ),
        CUDAExtension(
            name="actorshq.dataset.occupancy_grid_native",
            sources=["dataset/native/occupancy_grid.cu"],
            include_dirs=[_util_path],
            extra_compile_args={"cxx": [""], "nvcc": ["--use_fast_math"]},
        ),
        CUDAExtension(
            name="actorshq.toolbox.occupancy_grid_generation_native",
            sources=["toolbox/native/occupancy_grid_generation.cu"],
            include_dirs=[_util_path],
            extra_compile_args={"cxx": [""], "nvcc": ["--use_fast_math"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    description="""
        1. Ray sampler module that efficiently samples rays and skips empty space.
        2. Occupancy grid module that uses 3D cuda textures. This is used for space pruning by ray sampler.
        3. A module that generates occupancy grids from 2D masks.
    """,
)
