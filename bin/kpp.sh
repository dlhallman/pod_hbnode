# !/bin/sh
echo "USING PYTHON EXECUTABLE $1"

echo "GENERATING KPP NON-TRANSIENT PREDICTIONS"
$1 src/run_pod.py \
    --dataset KPP \
    --data_dir ./data/KPP.npz \
    --out_dir ./out/kpp/ \
    --modes 8 \
    --tstart 0 \
    --tstop 1251 \
    --tpred 500

$1 src/run_dmd.py \
    --dataset KPP \
    --data_dir ./data/KPP.npz \
    --out_dir ./out/kpp/ \
    --modes 24 \
    --tstart 0 \
    --tstop 1000 \
    --tpred 1000 \
	--lifts sin cos quad cube


echo "GENERATING TRANSIENT KPP PREDICTIONS"
$1 src/run_seq.py \
    --dataset KPP \
    --data_dir ./data/KPP.npz \
    --load_file ./out/kpp/pth/kpp_0_1251_pod_8.npz \
    --out_dir ./out/kpp/ \
    --tr_ind 800 \
    --val_ind 1000 \
    --eval_ind 2000 \
    --batch_size 1000 \
    --model NODE

$1 src/run_seq.py \
    --dataset KPP \
    --data_dir ./data/KPP.npz \
    --load_file ./out/kpp/pth/kpp_0_1251_pod_8.npz \
    --out_dir ./out/kpp/ \
    --tr_ind 800 \
    --val_ind 1000 \
    --eval_ind 2000 \
    --batch_size 1000 \
    --model HBNODE 

echo "COMPARISON PLOTS"
$1 src/compare.py \
   --out_dir ./out/kpp/ \
   --file_list ./out/kpp/pth/HBNODE.csv ./out/nonT_pred/pth/NODE.csv \
   --model_list seq_hbnode seq_node \
   --comparisons forward_nfe backward_nfe tr_loss val_loss \
   --epoch_freq 100


echo "BASH TASK(S) COMPLETED."

read -p "$*"
