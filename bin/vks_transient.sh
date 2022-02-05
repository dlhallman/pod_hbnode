# !/bin/sh
echo "FULL VKS USING DMD DECOMP"
python src/run_dmd.py \
	--modes 4 \
	--tstart 0 \
	--tstop 398 \
	--tpred 399

echo "FULL VKS USING POD DECOMP"
python src/run_pod.py \
	--modes 4 \
	--tstart 0 \
	--tstop 399 \
	--tpred 399

echo "BASH TASK(S) COMPLETED."

read -p "$*"
