# !/bin/sh
echo "USING PYTHON EXECUTABLE $1"

#echo "GENERATING VKS NON-TRANSIENT PREDICTIONS"
#$1 src/run_pod.py \
#    --dataset VKS \
#    --data_dir ./data/VKS.pkl \
#    --out_dir ./out/nonT_pred/ \
#    --modes 8 \
#    --tstart 100 \
#    --tstop 300 \
#    --tpred 100
#
#$1 src/run_dmd.py \
#    --dataset VKS \
#    --data_dir ./data/VKS.pkl \
#    --out_dir ./out/nonT_pred/ \
#    --modes 24 \
#    --tstart 100 \
#    --tstop 180 \
#    --tpred 100


echo "GENERATING NON-TRANSIENT VKS VAE PREDICTIONS"
#$1 src/run_vae.py \
#    --dataset VKS \
#    --data_dir ./data/VKS.pkl \
#    --load_file ./out/nonT_pred/pth/vks_100_200_pod_8.npz \
#    --out_dir ./out/nonT_pred/ \
#    --tr_ind 75 \
#    --val_ind 100 \
#    --eval_ind 200 \
#    --model NODE 

$1 src/run_vae.py \
    --dataset VKS \
   --data_dir ./data/VKS.pkl \
    --load_file ./out/nonT_pred/pth/vks_100_200_pod_8.npz \
    --out_dir ./out/nonT_pred/ \
    --tr_ind 75 \
    --val_ind 100 \
    --eval_ind 200 \
    --latent_dim 3 \
    --lr .001 \
    --model HBNODE

echo "BASH TASK(S) COMPLETED."

read -p "$*"
