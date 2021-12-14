# !/bin/sh
echo "RUNNING FIBER DATA USING NODE"
python src/seq.py \
    --dataset FIB \
    --data_dir data/FiberData.dat \
    --out_dir out/FIB/SEQ \
    --model NODE \
    --seq_win 10 \
    --epochs 100 \
    --lr .001 \

echo "TRAINING FIBER DATA USING HBNODE"
python src/seq.py \
    --dataset FIB \
    --data_dir data/FiberData.dat \
    --out_dir out/FIB/SEQ \
    --model HBNODE \
    --seq_win 10 \
    --epochs 100 \
    --lr .001 \

echo "TRAINING FIBER DATA USING GHBNODE"
python src/seq.py \
    --dataset FIB \
    --data_dir data/FiberData.dat \
    --out_dir out/FIB/SEQ \
    --model GHBNODE \
    --seq_win 10 \
    --epochs 100 \
    --lr .001 \
    --factor .98 

read -p "$*"
