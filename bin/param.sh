# !/bin/sh
echo "TRAINING EE DATA USING NODE"
python src/param.py \
    --dataset EE \
    --data_dir data/EulerEqs.npz \
    --out_dir out/EE/PARAM \
    --model NODE \
	--modes 4 \
    --tstart 100 \
	--tr_win 10 \
    --tstop 220 \
    --epochs 300

 echo "TRAINING EE DATA USING NODE"
 python src/param.py \
     --dataset EE \
     --data_dir data/EulerEqs.npz \
     --out_dir out/EE/PARAM \
     --model HBNODE \
 	--modes 4 \
     --tstart 100 \
 	--tr_win 10 \
     --tstop 120 \
     --epochs 300

 echo "TRAINING EE DATA USING NODE"
 python src/param.py \
     --dataset EE \
     --data_dir data/EulerEqs.npz \
     --out_dir out/EE/PARAM \
     --model GHBNODE \
 	--modes 4 \
     --tstart 100 \
 	--tr_win 10 \
     --tstop 200 \
     --epochs 300

echo "BASH TASK(S) COMPLETED."

read -p "$*"
