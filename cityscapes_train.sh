SELF=`readlink -f "$0"`
SELF_DIR=`dirname "$SELF"`
cd "$SELF_DIR"
HIP_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /usr/bin/python3 tools/train_net_step.py --dataset cityscapes --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml --nw 0
