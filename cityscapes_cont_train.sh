SELF=`readlink -f "$0"`
SELF_DIR=`dirname "$SELF"`
cd "$SELF_DIR"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 tools/train_net_step.py --resume --use_tfboard --load_ckpt ../model_step44999.pth --dataset cityscapes --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
