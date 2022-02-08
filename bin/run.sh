# !/bin/sh
echo "GENERATING BASELINES"
python src/baselines.py

echo "GENERATING VKS NON-TRANSIENT PREDICTIONS"
python src/run_pod.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --out_dir ./out/nonT_pred/ \
    --modes 8 \
    --tstart 100 \
    --tstop 200 \
    --tpred 100

python src/run_dmd.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --out_dir ./out/nonT_pred/ \
    --modes 24 \
    --tstart 100 \
    --tstop 180 \
    --tpred 100


echo "GENERATING VKS VAE PREDICTIONS"
python src/run_vae.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --load_file ./out/nonT_pred/pth/vks_100_200_pod_8.npz \
    --out_dir ./out/nonT_pred/ \
    --tr_ind 80 \
    --val_ind 100 \
    --model NODE

python src/run_vae.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --load_file ./out/nonT_pred/pth/vks_100_200_pod_8.npz \
    --out_dir ./out/nonT_pred/ \
    --tr_ind 80 \
    --val_ind 100 \
    --model HBNODE

echo "GENERATING VKS TRANSIENT PREDICTIONS"
python src/run_pod.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --out_dir ./out/full_pred/ \
    --modes 8 \
    --tstart 0 \
    --tstop 200 \
    --tpred 200

python src/run_dmd.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --out_dir ./out/full_pred/ \
    --modes 24 \
    --tstart 0 \
    --tstop 100 \
    --tpred 200

echo "BASH TASK(S) COMPLETED."

read -p "$*"
