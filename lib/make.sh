#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

python3 setup_bbox.py build_ext --inplace
rm -rf build

# Choose cuda arch as you need
CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "
#          -gencode arch=compute_70,code=sm_70 "

python3 setup_layers.py build
mv -v build/lib.*/model/_C.*.so model
