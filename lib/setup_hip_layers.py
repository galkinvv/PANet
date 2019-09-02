# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os
import types

import torch
from setuptools import find_packages
from setuptools import setup

from torch.utils.cpp_extension import CppExtension

requirements = ["torch", "torchvision"]

class LinkHipExtension(torch.utils.cpp_extension.BuildExtension):
    def build_extension(self, *args, **kwargs):
        HIP_DIR = os.environ['HIP_DIR']
        if HIP_DIR:
            orig_link = self.compiler.link
            def new_link(self_to_patch, *args, **kwargs):
                self_to_patch.linker_so[0] = HIP_DIR + "/bin/hipcc"
                orig_link(*args[:-1]) #skip target_lang so custom linker won't be replaced with c++
            self.compiler.link = types.MethodType(new_link, self.compiler)
        super(LinkHipExtension, self).build_extension(*args, **kwargs)


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "build", "hipsrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    HIP_DIR = os.environ['HIP_DIR']
    if torch.cuda.is_available() and HIP_DIR:
        sources = source_cuda + sources #try build gpu sources before cpu for faster errors
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            #default paths for docker-based pytorch installation
            "-I" + os.getenv("PYTORCH_HIP_INCLUDE", "/pytorch/torch/include/"), 
            "-I" + os.getenv("PYTORCH_HIP_INCLUDE2", "/pytorch/aten/src/"), 
            "-I" + os.getenv("ROCM_INCLUDE", "/opt/rocm/include/"), 
            "-I" + os.getenv("HIPRAND_INCLUDE", "/opt/rocm/hiprand/include/"), 
            "-I" + os.getenv("ROCRAND_INCLUDE", "/opt/rocm/rocrand/include/"), 
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
    cmdclass={"build_ext": LinkHipExtension},
)
