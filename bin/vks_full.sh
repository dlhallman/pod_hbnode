
# !/bin/sh
echo "USING PYTHON EXECUTABLE $1"

#echo "GENERATING VKS NON-TRANSIENT PREDICTIONS"
#$1 src/run_pod.py \
#    --dataset VKS \
#    --data_dir ./data/VKS.pkl \
#    --out_dir ./out/full/ \
#    --modes 8 \
#    --tstart 0 \
#    --tstop 300 \
#    --tpred 300

echo "GENERATING TRANSIENT VKS VAE PREDICTIONS"
$1 src/run_seq.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --load_file ./out/full/pth/vks_0_300_pod_8.npz \
    --out_dir ./out/nonT_pred/ \
    --tr_ind 75 \
    --val_ind 150 \
    --eval_ind 200 \
    --batch_size 100 \
    --model NODE 

$1 src/run_seq.py \
    --dataset VKS \
   --data_dir ./data/VKS.pkl \
    --load_file ./out/full/pth/vks_0_300_pod_8.npz \
    --out_dir ./out/full/ \
    --tr_ind 75 \
    --val_ind 150 \
    --eval_ind 200 \
    --batch_size 100 \
    --model HBNODE

echo "BASH TASK(S) COMPLETED."

read -p "$*"
