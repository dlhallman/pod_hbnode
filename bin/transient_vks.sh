#!/bin/sh
echo "TRAINING VKS DATA USING DMD"
python src/dmd.py \
    --out_dir out/VKS/TRANSIENT \
	--modes 8 \
	--tstart 0 \
	--tstop 100 \
	--tpred 140 \

echo "TRAINING VKS TRANSIENT DATA FOR LONG TERM PREDICTION USING NODE"
python src/seq.py \
    --out_dir out/VKS/TRANSIENT \
    --model NODE \
    --tstart 0 \
    --tstop 400 \
    --tr_win 100 \
    --val_win 140 \
    --epochs 200 \
    --modes 8

echo "TRAINING VKS TRANSIENT DATA FOR LONG TERM PREDICTION USING NODE"
python src/seq.py \
    --out_dir out/VKS/TRANSIENT \
    --model HBNODE \
    --tstart 0 \
    --tstop 400 \
    --tr_win 100 \
    --val_win 140 \
    --epochs 200 \
    --modes 8

echo "TRAINING VKS TRANSIENT DATA FOR LONG TERM PREDICTION USING GHBNODE"
python src/seq.py \
    --out_dir out/VKS/TRANSIENT \
    --model GHBNODE \
    --tstart 0 \
    --tstop 400 \
    --tr_win 100 \
    --val_win 140 \
    --epochs 200 \
    --modes 8

echo "BASH TASK(S) COMPLETED."

read -p "$*"
