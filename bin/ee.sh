# !/bin/sh
echo "USING PYTHON EXECUTABLE $1"

echo "GENERATING EE PREDICTIONS"
#$1 src/run_param.py \
#    --dataset EE \
#    --data_dir ./data/EulerEqs.npz \
#    --out_dir ./out/ee/ \
#    --tstart 0 \
#    --tstop 180 \
#    --tr_ind 150 \
#    --param_ind 90 \
#    --model NODE \
#    --epochs 200 \
#    --verbose True

#$1 src/run_param.py \
#    --dataset EE \
#    --data_dir ./data/EulerEqs.npz \
#    --out_dir ./out/ee/ \
#    --tstart 0 \
#    --tstop 180 \
#    --tr_ind 150 \
#    --param_ind 90 \
#    --model GHBNODE \
#    --epochs 200 \
#    --verbose True

echo "COMPARISON PLOTS"
$1 src/compare.py \
   --out_dir ./out/ee/ \
   --file_list ./out/ee/pth/GHBNODE.csv ./out/ee/pth/NODE.csv \
   --model_list GHBNODE NODE \
   --comparisons forward_nfe backward_nfe tr_loss val_loss \
   --epoch_freq 1


echo "BASH TASK(S) COMPLETED."

read -p "$*"
