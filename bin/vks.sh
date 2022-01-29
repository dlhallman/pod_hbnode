#!/bin/sh
echo "TRAINING VKS DATA USING DMD"
python src/dmd.py \
	--modes 40 \
	--tstart 100 \
	--tstop 175 \
	--tpred 200 \

echo "TRAINING VKS DATA USING NODE"
python src/vae.py \
    
echo "TRAINING VKS DATA USING HBNODE"
python src/vae.py \
    --model HBNODE \
	--lr .003 \
	--latent_dim 2 \
	--units_dec 32 \

# echo "TRAINING VKS DATA USING NODE"
# python src/seq.py \
# 	--model NODE \
# 	--modes 8 \
# 	--seq_win 20 \
# 	--val_win 140 \
# 	--cooldown 2 \
# 	--epochs 100 \
   
#  echo "TRAINING VKS DATA USING HBNODE"
#  python src/seq.py \
#     --model HBNODE  \
# 	--modes 8 \
# 	--seq_win 20 \
# 	--val_win 140 \
#  	--lr .001 \
#  	--cooldown 2 \
#  	--factor .9 \
#  	--epochs 100

#  echo "TRAINING VKS DATA USING GHBNODE"
#  python src/seq.py \
#     --model GHBNODE \
# 	--modes 8 \
# 	--seq_win 20 \
# 	--val_win 140 \
#  	--lr .007 \
#  	--cooldown 2 \
#  	--factor .9 \
#  	--epochs 100

echo "PLOTTING MIXED PLOTS"
python src/vis.py

echo "BASH TASK(S) COMPLETED."

read -p "$*"
