SELF=`readlink -f "$0"`
SELF_DIR=`dirname "$SELF"`
cd "$SELF_DIR"
PYTHONUNBUFFERED=1 /usr/bin/python3 tools/train_net_step.py --dataset cityscapes --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask_pret.yaml --use_tfboard --load_ckpt data/pretrained_model/panet_mask_step179999.pth --skip_top_layers
