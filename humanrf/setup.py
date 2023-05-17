import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_util_path = str(Path(os.path.realpath(__file__)).parent.parent / "actorshq" / "toolbox" / "native")

setup(
    name="humanrf",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            name="humanrf.scene_representation.tensor_composition_native",
            sources=["scene_representation/native/tensor_composition.cu"],
            include_dirs=[_util_path],
            extra_compile_args={"cxx": [""], "nvcc": ["--use_fast_math"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    description="""
        1. A fast way to both sample from feature vectors and compose them with feature volume counterparts.
    """,
)
