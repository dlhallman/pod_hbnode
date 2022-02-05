# !/bin/sh
echo "NON-TRANSIENT VKS USING DMD DECOMP"
python src/run_dmd.py \
	--modes 4 \
	--tstart 100 \
	--tstop 398 \
	--tpred 399

echo "NON-TRANSIENT VKS USING POD DECOMP"
python src/run_pod.py \
	--modes 4 \
	--tstart 100 \
	--tstop 399 \
	--tpred 399

echo "BASH TASK(S) COMPLETED."

read -p "$*"
