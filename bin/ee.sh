# !/bin/sh
echo "USING PYTHON EXECUTABLE $1"

echo "GENERATING EE PREDICTIONS"
$1 src/run_param.py \
    --dataset EE \
    --data_dir ./data/EulerEqs.npz \
    --load_file ./out/ee/pth/ee_0_181_pod_8.npz \
    --out_dir ./out/ee/ \
    --tr_ind 150 \
    --param_ind 90 \
    --model NODE \
    --epochs 100

$1 src/run_param.py \
    --dataset EE \
    --data_dir ./data/EulerEqs.npz \
    --load_file ./out/ee/pth/ee_0_181_pod_8.npz \
    --out_dir ./out/ee/ \
    --tr_ind 150 \
    --param_ind 90 \
    --model HBNODE \
    --layers 2 \
    --epochs 100

#echo "COMPARISON PLOTS"
#$1 src/compare.py \
#   --out_dir ./out/ee/ \
#   --file_list ./out/ee/pth/HBNODE.csv ./out/ee/pth/NODE.csv \
#   --model_list HBNODE NODE \
#   --comparisons forward_nfe backward_nfe tr_loss val_loss \
#   --epoch_freq 100


echo "BASH TASK(S) COMPLETED."

read -p "$*"
