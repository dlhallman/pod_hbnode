# !/bin/sh
echo "PREDICTING VKS USING DMD DECOMP"
python src/run_dmd.py \

echo "PREDICTING KPP USING DMD DECOMP"
python src/run_dmd.py\
    --dataset KPP \
    --data_dir ./data/KPP.npz \
    --tpred 120 \
    --lifts sin cos quad cube \

echo "PREDICTING EE USING DMD DECOMP"
python src/run_dmd.py\
    --dataset EE \
    --data_dir ./data/EulerEqs.npz \
    --tpred 120 \

echo "BASH TASK(S) COMPLETED."

read -p "$*"