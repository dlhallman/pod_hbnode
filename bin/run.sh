# !/bin/sh
echo "USING PYTHON EXECUTABLE $1"

echo "GENERATING BASELINES"
$1 src/baselines.py

echo "GENERATING VKS NON-TRANSIENT PREDICTIONS"
$1 src/run_pod.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --out_dir ./out/nonT_pred/ \
    --modes 8 \
    --tstart 100 \
    --tstop 300 \
    --tpred 100

$1 src/run_dmd.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --out_dir ./out/nonT_pred/ \
    --modes 24 \
    --tstart 100 \
    --tstop 180 \
    --tpred 100


echo "GENERATING NON-TRANSIENT VKS VAE PREDICTIONS"
$1 src/run_vae.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --load_file ./out/nonT_pred/pth/vks_100_200_pod_8.npz \
    --out_dir ./out/nonT_pred/ \
    --tr_ind 75 \
    --val_ind 100 \
    --eval_ind 200
    --model NODE \
	--seed 1242 \
	--verbose True

$1 src/run_vae.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --load_file ./out/nonT_pred/pth/vks_100_200_pod_8.npz \
    --out_dir ./out/nonT_pred/ \
    --tr_ind 75 \
    --val_ind 100 \
    --eval_ind 200
    --latent_dim 3 \
    --units_dec 24 \
    --factor .99 \
    --cooldown 0 \
    --seed 1242 \
    --model HBNODE \
    --verbose True

echo "GENERATING VKS TRANSIENT DATA"
$1 src/run_pod.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --out_dir ./out/full_pred/ \
    --modes 8 \
    --tstart 0 \
    --tstop 200 \
    --tpred 200

$1 src/run_dmd.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --out_dir ./out/full_pred/ \
    --modes 24 \
    --tstart 0 \
    --tstop 100 \
    --tpred 200

echo "GENERATING TRANSIENT VKS SEQ PREDICITONS"
$1 src/run_vae.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --load_file ./out/nonT_pred/pth/vks_100_200_pod_8.npz \
    --out_dir ./out/nonT_pred/ \
    --tr_ind 80 \
    --val_ind 100 \
    --model NODE \
	--seed 1242 \
	--verbose True

$1 src/run_vae.py \
    --dataset VKS \
    --data_dir ./data/VKS.pkl \
    --load_file ./out/nonT_pred/pth/vks_100_200_pod_8.npz \
    --out_dir ./out/nonT_pred/ \
    --tr_ind 80 \
    --val_ind 100 \
	--latent_dim 3 \
	--units_dec 24 \
	--factor .99 \
	--cooldown 0 \
	--seed 1242 \
    --model HBNODE \
	--verbose True

echo "BASH TASK(S) COMPLETED."

read -p "$*"
