# !/bin/sh
echo "USING PYTHON EXECUTABLE $1"

#echo "GENERATING EE BASELINES"
#$1 src/run_pod.py \
#    --dataset EE \
#    --data_dir ./data/EulerEqs.npz \
#    --out_dir ./out/ee/ \
#    --modes 8 \
#    --tstart 0 \
#    --tstop 181 \
#    --tpred 180 \
#    --eeParam 80

echo "GENERATING EE PREDICTIONS"
$1 src/run_param.py \
    --dataset EE \
    --data_dir ./data/EulerEqs.npz \
    --load_file ./out/ee/pth/ee_0_181_pod_8.npz \
    --out_dir ./out/ee/ \
    --tr_ind 100 \
    --param_ind 80 \
    --model NODE \
    --epochs 500

$1 src/run_param.py \
    --dataset EE \
    --data_dir ./data/EulerEqs.npz \
    --load_file ./out/ee/pth/ee_0_181_pod_8.npz \
    --out_dir ./out/ee/ \
    --tr_ind 100 \
    --param_ind 80 \
    --model HBNODE \
	--epocsh 500

#echo "COMPARISON PLOTS"
#$1 src/compare.py \
#   --out_dir ./out/kpp/ \
#   --file_list ./out/kpp/pth/HBNODE.csv ./out/nonT_pred/pth/NODE.csv \
#   --model_list seq_hbnode seq_node \
#   --comparisons forward_nfe backward_nfe tr_loss val_loss \
#   --epoch_freq 100


echo "BASH TASK(S) COMPLETED."

read -p "$*"
