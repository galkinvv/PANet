#!/bin/bash
SELF=`readlink -f "$0"`
SELF_DIR=`dirname "$SELF"`
cd "$SELF_DIR"
UPPER_DATA_DIR=`dirname "$SELF_DIR"` # expect data in upper dir outside repo to speed up git ops
CHECKPOINT="${UPPER_DATA_DIR}/panet_mask_step179999.pth"
if [ ! -s "$CHECKPOINT" ]; then
    echo "Initial checkpoint from training on coco need to be downloaded to $CHECKPOINT. See model link in the README.md (botom table)"
    exit 1
fi
CITYSCAPES_CHECK_FILE="$UPPER_DATA_DIR/cityscapes/annotations/instancesonly_gtFine_train.json"
if [ ! -s "${CITYSCAPES_CHECK_FILE}" ]; then
    echo "Cityscapes images and validation data must be downloaded (so that ${CITYSCAPES_CHECK_FILE} would be present)"
    exit 1
fi
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED
python3 tools/train_net_step.py --dataset cityscapes --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask_csc_from_coco.yaml --use_tfboard --load_ckpt "${CHECKPOINT}" --skip_top_layers ${EXTRA_CSC_TRAIN_ARGS}

