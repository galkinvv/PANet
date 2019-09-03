#!/usr/bin/env bash
SELF=`readlink -f "$0"`
SELF_DIR=`dirname "$SELF"`
cd "$SELF_DIR"

python3 setup_bbox.py build_ext --inplace
rm -rf build
mkdir -p build
cp -rv model/csrc build/csrc
HIP_DIR=/opt/rocm/hip
if [ -d "$HIP_DIR" ]; then
	export HIP_DIR
	$HIP_DIR/bin/hipify-perl --inplace --print-stats build/csrc/cuda/*.cu

	#torch-specific replacements for hip
	sed -i 's!ATen/cuda/CUDAContext.h!ATen/hip/HIPContext.h!g' build/csrc/cuda/*.cu
	sed -i 's!THC/THC!THH/THH!g' build/csrc/cuda/*.cu
	sed -i 's!getCurrentCUDAStream!getCurrentHIPStream!g' build/csrc/cuda/*.cu
	#export CUDA_PATH=$SELF_DIR #contains bin/nvcc redirector
else
	export CUDA_PATH=/usr/local/cuda/
fi
python3 setup_layers.py build --debug
mv -v build/lib.*/model/_C.*.so model
