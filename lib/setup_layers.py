# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from setuptools import Extension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import include_paths

requirements = ["torch", "torchvision"]

HIP_DIR = os.getenv("HIP_DIR") #would be None for building with CUDA

def HIPExtension(name, sources, *args, **kwargs):
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths() + [
        #non-public includea are needed for using hip+pytroch
        #default paths for docker-based pytorch installation
        os.getenv("PYTORCH_HIP_INCLUDE", "/pytorch/torch/include/"), 
        os.getenv("PYTORCH_HIP_INCLUDE2", "/pytorch/aten/src/"), 
        os.getenv("ROCM_INCLUDE", "/opt/rocm/include/"), 
        os.getenv("HIPRAND_INCLUDE", "/opt/rocm/hiprand/include/"), 
        os.getenv("ROCRAND_INCLUDE", "/opt/rocm/rocrand/include/"), 
    ]
    kwargs['include_dirs'] = include_dirs
    kwargs['language'] = 'c++'

    #combine nvcc compile args and cxx compile args, since all source swould be treated as cxx with hipcc compiler
    extra_compile_args_dict = kwargs.get('extra_compile_args', {})
    extra_compile_args_list = extra_compile_args_dict.get('cxx', []) + extra_compile_args_dict.get('nvcc', [])
    kwargs['extra_compile_args'] = extra_compile_args_list
    return Extension(name, sources, *args, **kwargs)


class BuildGpuExtension(torch.utils.cpp_extension.BuildExtension):
    def build_extensions(self, *args, **kwargs):
        #torch would try wraaping unix compiler for calling nvcc
        #we need other functionality from BuildExtension, but this compiler wraaper won't be used
        #So, save original compiler before calling build_extensions
        self._build_hip_original_unix_compile = self.compiler._compile
        super(BuildGpuExtension, self).build_extensions(*args, **kwargs)

    def build_extension(self, *args, **kwargs):
        if HIP_DIR:
            self.compiler._compile = self._build_hip_original_unix_compile
            self.compiler.set_executable('compiler_so', [HIP_DIR + "/bin/hipcc", "-fPIC"]) #used as compiler
            self.compiler.set_executable('compiler_cxx', [HIP_DIR + "/bin/hipcc"]) #used as linker
        super(BuildGpuExtension, self).build_extension(*args, **kwargs)


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "build", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and (HIP_DIR or CUDA_HOME is not None):
        extension = HIPExtension if HIP_DIR else CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "model._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="detectron_pytorch",
    version="0.1",
    description="detectron in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildGpuExtension},
)
