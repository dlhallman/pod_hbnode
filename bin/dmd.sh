# !/bin/sh
# echo "PREDICTING VKS USING DMD DECOMP"
# python src/run_dmd.py

echo "PREDICTING KPP USING DMD DECOMP"
python src/run_dmd.py\
    --dataset KPP \
    --data_dir ./data/KPP.npz \
    --tpred 120 \
    --modes 50


echo "BASH TASK(S) COMPLETED."

read -p "$*"