#!/bin/bash
SELF=`readlink -f "$0"`
SELF_DIR=`dirname "$SELF"`
EXTRA_CSC_TRAIN_ARGS="--iter_size=4" CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 "$SELF_DIR"/cityscapes_train_from_coco.sh

