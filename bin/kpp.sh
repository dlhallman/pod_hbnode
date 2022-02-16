# !/bin/sh
echo "USING PYTHON EXECUTABLE $1"

echo "GENERATING KPP BASELINES"
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
    --tpred 999 \
    --lifts sin cos quad cube


echo "GENERATING KPP PREDICTIONS"
$1 src/run_seq.py \
    --dataset KPP \
    --data_dir ./data/KPP.npz \
    --load_file ./out/kpp/pth/kpp_0_1251_pod_8.npz \
    --out_dir ./out/kpp/ \
    --tr_ind 800 \
 	--seq_ind 4 \
    --val_ind 1000 \
    --eval_ind 1251 \
    --batch_size 1000 \
	--layers 2 \
	--lr .01 \
    --model NODE

$1 src/run_seq.py \
    --dataset KPP \
    --data_dir ./data/KPP.npz \
    --load_file ./out/kpp/pth/kpp_0_1251_pod_8.npz \
    --out_dir ./out/kpp/ \
    --tr_ind 800 \
	--seq_ind 4 \
    --val_ind 1000 \
    --eval_ind 1251 \
    --batch_size 1000 \
	--layers 2 \
	--lr .01 \
    --model HBNODE 

echo "COMPARISON PLOTS"
$1 src/compare.py \
   --out_dir ./out/kpp/ \
   --file_list ./out/kpp/pth/HBNODE.csv ./out/kpp/pth/NODE.csv \
   --model_list seq_hbnode seq_node \
   --comparisons forward_nfe backward_nfe tr_loss val_loss forward_stiff backward_stiff \
   --epoch_freq 5


echo "BASH TASK(S) COMPLETED."

read -p "$*"
