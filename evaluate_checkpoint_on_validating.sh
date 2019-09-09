#!/bin/bash
SELF=`readlink -f "$0"`
SELF_DIR=`dirname "$SELF"`
cd "$SELF_DIR"
UPPER_DATA_DIR=`dirname "$SELF_DIR"` # expect data in upper dir outside repo to speed up git ops
CITYSCAPES_CHECK_FILE="$UPPER_DATA_DIR/cityscapes/annotations/instancesonly_gtFine_train.json"
if [ ! -s "${CITYSCAPES_CHECK_FILE}" ]; then
    echo "Cityscapes images and validation data must be downloaded (so that ${CITYSCAPES_CHECK_FILE} would be present)"
    exit 1
fi
CITYSCAPES_RAW_VAL_CHECK_DIR="$UPPER_DATA_DIR/cityscapes/raw/gtFine/val"
if [ ! -d "${CITYSCAPES_RAW_VAL_CHECK_DIR}" ]; then
    echo "Warning: final part of evaluation will be skipped since raw cityscapes groundtruth is missing at"
    echo "${CITYSCAPES_RAW_VAL_CHECK_DIR}"
    echo ""
    echo "Download and unpack gtFine_trainvaltest.zip from cityscapes-dataset.com"
fi
if [ ! -s lib/model/_C*so ]; then
    lib/make.sh #build extensions if they are not built yet
fi
CHECKPOINT="$1"
if [ -z "$CHECKPOINT" ]; then
    echo "Checkpoint not given as argument. Falling back to newest checkpoint in Outputs"
    for file in `find Outputs -type f -name '*.pth'`; do
        [[ "$file" -nt "$CHECKPOINT" ]] && CHECKPOINT="$file"
    done
fi
if [ ! -s "$CHECKPOINT" ]; then
    echo "Checkpoint $CHECKPOINT not found or is empty. Pass correct checkpoint as script argument"
    exit 1
fi
OUT_DIR="$CHECKPOINT.validation/`date +%Y%m%d%H%M%S`/"
mkdir -p "${OUT_DIR}"
echo "Evaluating saving all output to ${OUT_DIR}"
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 tools/test_net.py --dataset cityscapes --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask_csc_from_coco.yaml --load_ckpt "${CHECKPOINT}" --output_dir "$OUT_DIR" | tee "$OUT_DIR/stdout.txt"

