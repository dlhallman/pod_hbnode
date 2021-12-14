#!/bin/sh
echo "TRAINING VKS DATA USING NODE"
python src/vae.py \
    --model NODE 
    
echo "TRAINING VKS DATA USING HBNODE"
python src/vae.py \
    --model HBNODE \
	--out_dir out/VKS/VAE/

echo "TRAINING KPP DATA USING NODE"
python src/vae.py \
    --model NODE \
    --dataset KPP \
	--data_dir data/KPP.npz \
	--out_dir out/KPP/VAE

echo "TRAINING KPP DATA USING HBNODE"
python src/vae.py \
    --model HBNODE \
    --dataset KPP \
	--data_dir data/KPP.npz \
	--out_dir out/KPP/VAE

echo "TRAINING EE DATA USING NODE"
python src/vae.py \
    --model NODE \
    --dataset EE \
	--data_dir data/EE.npz \
	--out_dir out/EE/VAE

echo "TRAINING EE DATA USING HBNODE"
python src/vae.py \
    --model HBNODE \
    --dataset EE \
	--data_dir data/EE.npz \
	--out_dir out/EE/VAE

echo "BASH TASK(S) COMPLETED."

read -p "$*"
