#!/usr/bin/env bash
SELF=`readlink -f "$0"`
SELF_DIR=`dirname "$SELF"`
cd "$SELF_DIR"

#rm -rf build  #remove build folder for clean build

python3 setup_bbox.py build_ext --inplace --debug
python3 setup_layers.py build_ext --inplace --debug
