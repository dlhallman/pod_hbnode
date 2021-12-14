#!/bin/sh
# echo "PLOTTING SEQ PLOTS FOR VKS/SEQ"
# python src/vis.py

echo "PLOTTING SEQ PLOTS FOR VKS/TRANSIENT"
python src/vis.py \
    --pth_dir out/VKS/TRANSIENT/pth 
   
# echo "PLOTTING SEQ PLOTS FOR VKS/VAE"
# python src/vis.py \
#     --pth_dir out/VKS/VAE/pth \
#     --models NODE,HBNODE
   
# echo "PLOTTING SEQ PLOTS FOR KPP/SEQ"
# python src/vis.py \
#     --pth_dir out/KPP/SEQ/pth
   
# echo "PLOTTING SEQ PLOTS FOR EE/SEQ"
# python src/vis.py \
#     --pth_dir out/EE/SEQ/pth

echo "BASH TASK(S) COMPLETED."

read -p "$*"