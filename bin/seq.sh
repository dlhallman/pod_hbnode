#!/bin/sh
echo "TRAINING VKS DATA USING NODE"
python src/seq.py \
    --model NODE \
	--cooldown 2 \
	--epochs 500
    
echo "TRAINING VKS DATA USING HBNODE"
python src/seq.py \
    --model HBNODE  \
	--lr .001 \
	--cooldown 2 \
	--factor .9 \
	--epochs 500

echo "TRAINING VKS DATA USING GHBNODE"
python src/seq.py \
    --model GHBNODE \
	--lr .007 \
	--cooldown 2 \
	--factor .9 \
	--epochs 500

echo "PLOTTING MIXED PLOTS"
python src/vis.py

# !/bin/sh
echo "TRAINING KPP DATA USING NODE"
python src/seq.py \
    --dataset KPP \
    --data_dir data/KPP.npz \
    --out_dir out/KPP/SEQ \
    --model NODE \
	--modes 4 \
	--batch_size 20 \
    --tstart 100 \
	--tr_win 100 \
	--val_win 125 \
	--seq_win 10 \
    --epochs 500 \
    --lr .005 \
    --factor .95 

echo "TRAINING KPP DATA USING HBNODE"
python src/seq.py \
    --dataset KPP \
    --data_dir data/KPP.npz \
    --out_dir out/KPP/SEQ \
    --model HBNODE \
	--modes 4 \
	--batch_size 20 \
    --tstart 100 \
	--tr_win 100 \
	--val_win 125 \
	--seq_win 10 \
    --epochs 500 \
    --lr .005 \
    --factor .95 

echo "TRAINING KPP DATA USING GHBNODE"
python src/seq.py \
    --dataset KPP \
    --data_dir data/KPP.npz \
    --out_dir out/KPP/SEQ \
    --model GHBNODE \
	--modes 4 \
	--batch_size 20 \
    --tstart 100 \
	--tr_win 100 \
	--val_win 125 \
	--seq_win 10 \
    --epochs 500 \
    --lr .005 \
    --factor .95

# !/bin/sh
echo "TRAINING EE DATA USING NODE"
python src/seq.py \
    --dataset EE \
    --data_dir data/EulerEqs.npz \
    --out_dir out/EE/SEQ \
    --model NODE \
	--modes 4 \
    --tstart 100 \
	--tr_win 200 \
	--val_win 300 \
    --epochs 200

echo "TRAINING EE DATA USING HBNODE"
python src/seq.py \
    --dataset EE \
    --data_dir data/EulerEqs.npz \
    --out_dir out/EE/SEQ \
    --model HBNODE \
	--modes 4 \
    --tstart 100 \
	--tr_win 200 \
	--val_win 300 \
    --epochs 200 \
	--lr .001 

echo "TRAINING EE DATA USING GHBNODE"
python src/seq.py \
    --dataset EE \
    --data_dir data/EulerEqs.npz \
    --out_dir out/EE/SEQ \
    --model GHBNODE \
	--modes 4 \
    --tstart 100 \
	--tr_win 200 \
	--val_win 300 \
    --epochs 200 \
	--lr .001 \
	--cooldown 2 \
	--factor .9