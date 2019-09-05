#!/usr/bin/env bash
SELF=`readlink -f "$0"`
SELF_DIR=`dirname "$SELF"`
cd "$SELF_DIR"

#rm -rf build  #remove build folder for clean build
mkdir -p build
NEWER_SOURCES=`cp -rv --update model/csrc build/csrc`
echo "${NEWER_SOURCES}"
HIP_DIR=/opt/rocm/hip
if [ -d "$HIP_DIR" ]; then
	if [ ! -z "${NEWER_SOURCES}" ]; then
		$HIP_DIR/bin/hipify-perl --inplace --print-stats build/csrc/cuda/*.cu

		#torch-specific replacements for hip
		sed -i 's!ATen/cuda/CUDAContext.h!ATen/hip/HIPContext.h!g' build/csrc/cuda/*.cu
		sed -i 's!THC/THC!THH/THH!g' build/csrc/cuda/*.cu
		sed -i 's!getCurrentCUDAStream!getCurrentHIPStream!g' build/csrc/cuda/*.cu
	fi

	export HIP_DIR
else
	export CUDA_PATH=/usr/local/cuda/
fi
python3 setup_bbox.py build_ext --inplace --debug
python3 setup_layers.py build_ext --inplace --debug
