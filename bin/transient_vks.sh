#!/bin/sh
echo "TRAINING VKS TRANSIENT DATA FOR LONG TERM PREDICTION USING NODE"
python src/seq.py \
    --out_dir out/VKS/TRANSIENT \
    --model NODE \
    --tstart 0 \
    --tstop 500 \
    --tr_win 100 \
    --val_win 140 \
    --epochs 200 \
    --modes 4

echo "TRAINING VKS TRANSIENT DATA FOR LONG TERM PREDICTION USING NODE"
python src/seq.py \
    --out_dir out/VKS/TRANSIENT \
    --model HBNODE \
    --tstart 0 \
    --tstop 500 \
    --tr_win 100 \
    --val_win 140 \
    --epochs 200 \
    --modes 4

echo "TRAINING VKS TRANSIENT DATA FOR LONG TERM PREDICTION USING GHBNODE"
python src/seq.py \
    --out_dir out/VKS/TRANSIENT \
    --model GHBNODE \
    --tstart 0 \
    --tstop 500 \
    --tr_win 100 \
    --val_win 140 \
    --epochs 200 \
    --modes 4

echo "BASH TASK(S) COMPLETED."

read -p "$*"
