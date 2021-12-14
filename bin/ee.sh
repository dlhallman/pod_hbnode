# !/bin/sh
echo "TRAINING EE DATA USING NODE"
python src/seq.py \
    --dataset EE \
    --data_dir data/EulerEqs.npz \
    --out_dir out/EE/SEQ \
    --model NODE \
	--modes 39 \
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
	--modes 39 \
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
	--modes 39 \
    --tstart 100 \
	--tr_win 200 \
	--val_win 300 \
    --epochs 200 \
	--lr .001 \
	--cooldown 2 \
	--factor .9

echo "BASH TASK(S) COMPLETED."

read -p "$*"
