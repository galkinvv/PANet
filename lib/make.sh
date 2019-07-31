#!/usr/bin/env bash
SELF=`readlink -f "$0"`
SELF_DIR=`dirname "$SELF"`
cd "$SELF_DIR"

#python3 setup_bbox.py build_ext --inplace
rm -rf build
HIP_DIR=/opt/rocm/hip
if [ -d "$HIP_DIR" ]; then
	export HIP_DIR
	mkdir -p build
	cp -rv model/csrc build/hipsrc
	$HIP_DIR/bin/hipify-perl --inplace --print-stats build/hipsrc/cuda/*.cu

	#torch-specific replacements
	sed -i 's!ATen/cuda/CUDAContext.h!ATen/hip/HIPContext.h!g' build/hipsrc/cuda/*.cu
	sed -i 's!THC/THC!THH/THH!g' build/hipsrc/cuda/*.cu
	sed -i 's!getCurrentCUDAStream!getCurrentHIPStream!g' build/hipsrc/cuda/*.cu
	export CUDA_PATH=$SELF_DIR #contains bin/nvcc redirector

	python3 setup_hip_layers.py build
else
	CUDA_PATH=/usr/local/cuda/

	# Choose cuda arch as you need
	CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "
	#  -gencode arch=compute_70,code=sm_70 "
	python3 setup_layers.py build
fi
mv -v build/lib.*/model/_C.*.so model
