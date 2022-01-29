#!/bin/sh
echo "TRAINING VKS DATA USING DMD"
python src/dmd.py \
	--modes 8 \
	--tstart 100 \
	--tstop 175 \
	--tpred 200 \

echo "BASH TASK(S) COMPLETED."

read -p "$*"