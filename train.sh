#!/bin/bash
echo "Running source"
source /root/miniforge3/bin/activate

echo "Running conda activate"
conda init
eval "$(conda shell.bash hook)"
conda activate diff

echo "enter files"
cd /zhangqi/mycodes/mygit/diffguiLab

echo "Running training"
# You can resume by setting config.train.resume to True and passing --logdir to the previous run directory.
# Default: start new run under ./logs
python scripts/train.py   --config configs/train/train_crossattn.yml   --device cuda:0   --logdir logs/diffgui_crossattn
