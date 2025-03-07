# !/bin/sh
 echo "USING PYTHON EXECUTABLE $1"
 
 echo "GENERATING VKS NON-TRANSIENT PREDICTIONS"
 $1 src/run_pod.py \
     --dataset VKS \
     --data_dir ./data/VKS.pkl \
     --out_dir ./out/nonT_pred/ \
     --modes 8 \
     --tstart 100 \
     --tstop 400 \
     --tpred 100
 
 $1 src/run_dmd.py \
     --dataset VKS \
     --data_dir ./data/VKS.pkl \
     --out_dir ./out/nonT_pred/ \
     --lifts sin cos quad cube \
     --modes 24 \
     --tstart 100 \
     --tstop 400 \
     --tpred 399
 
 
 echo "GENERATING NON-TRANSIENT VKS VAE PREDICTIONS"
 $1 src/run_vae.py \
     --dataset VKS \
     --data_dir ./data/VKS.pkl \
     --load_file ./out/nonT_pred/pth/vks_100_400_pod_8.npz \
     --out_dir ./out/nonT_pred/ \
     --tr_ind 75 \
     --val_ind 100 \
     --eval_ind 200 \
     --model NODE
 
 $1 src/run_vae.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --load_file ./out/nonT_pred/pth/vks_100_400_pod_8.npz \
    --out_dir ./out/nonT_pred/ \
    --tr_ind 75 \
    --val_ind 100 \
    --eval_ind 200 \
    --mdel HBNODE \
    --seed 0


echo "COMPARISON PLOTS"
$1 src/compare.py \
   --out_dir ./out/nonT_pred/ \
   --file_list ./out/nonT_pred/pth/HBNODE.csv ./out/nonT_pred/pth/NODE.csv \
   --model_list HBNODE NODE \
   --comparisons forward_nfe backward_nfe tr_loss val_loss \
   --epoch_freq 100


echo "BASH TASK(S) COMPLETED."

read -p "$*"
