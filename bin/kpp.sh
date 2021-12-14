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

echo "BASH TASK(S) COMPLETED."

read -p "$*"
