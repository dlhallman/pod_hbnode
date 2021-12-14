#!/bin/sh
echo "TRAINING VKS ON 2 MODES"
python src/seq.py \
    --model HBNODE \
    --modes 2 \
    --out_dir out/VKS/MODES/TWO \
    
echo "VKS HBNODE ON 12 MODES"
python src/seq.py \
    --model HBNODE \
    --modes 12 \
    --out_dir out/VKS/MODES/TWELVE \

read -p "$*"
