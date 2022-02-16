
# !/bin/sh
echo "USING PYTHON EXECUTABLE $1"

echo "GENERATING VKS NON-TRANSIENT PREDICTIONS"
$1 src/run_pod.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --out_dir ./out/full/ \
    --modes 8 \
    --tstart 0 \
    --tstop 400 \
    --tpred 200

echo "GENERATING TRANSIENT VKS SEQ PREDICTIONS"
$1 src/run_seq.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --load_file ./out/full/pth/vks_0_400_pod_8.npz \
    --out_dir ./out/full/ \
    --tr_ind 80 \
    --val_ind 120 \
    --eval_ind 200 \
    --batch_size 100 \
    --model NODE \
    --epochs 2

$1 src/run_seq.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --load_file ./out/full/pth/vks_0_400_pod_8.npz \
    --out_dir ./out/full/ \
    --tr_ind 80 \
    --val_ind 120 \
    --eval_ind 200 \
    --batch_size 100 \
    --model HBNODE  \
    --epochs 2

echo "COMPARISON PLOTS"
$1 src/compare.py \
   --out_dir ./out/full/ \
   --file_list ./out/full/pth/HBNODE.csv ./out/full/pth/NODE.csv \
   --model_list seq_hbnode seq_node \
   --comparisons forward_nfe backward_nfe tr_loss val_loss \
   --epoch_freq 5

echo "BASH TASK(S) COMPLETED."

read -p "$*"
