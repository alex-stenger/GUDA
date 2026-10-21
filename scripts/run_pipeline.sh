#!/bin/bash
# Full GUDA pipeline for one source -> target scenario, with the settings of the paper:
#   1. warm-up on the source domain (200,000 iterations)
#   2. adaptation to the target domain (800,000 iterations, lambda = 1e-4)
#   3. sampling of N = 10 masks per target test image, repeated over 10 seeds
#   4. evaluation of the averaged masks (threshold 0.5)
# The SAM pseudo-labels must have been computed beforehand with tools/sam_pseudolabels.py.
#
# Usage: bash scripts/run_pipeline.sh <datasets_root> <source_domain> <target_domain>
# Example: bash scripts/run_pipeline.sh datasets FS1 WeiH
set -e

DATA=$1
SRC=$2
TRG=$3
NUM_SEEDS=${NUM_SEEDS:-10}

if [ -z "$DATA" ] || [ -z "$SRC" ] || [ -z "$TRG" ]; then
  echo "Usage: bash scripts/run_pipeline.sh <datasets_root> <source_domain> <target_domain>"
  exit 1
fi

# 1. Warm-up on the source domain
python tools/train.py --stage source --run_name ${SRC}_source \
  --dataset_source_path ${DATA}/${SRC} --result_path workdir

# 2. Adaptation to the target domain
python tools/train.py --stage adapt --run_name ${SRC}-${TRG} \
  --dataset_source_path ${DATA}/${SRC} --dataset_target_path ${DATA}/${TRG} \
  --pretrain_path workdir/${SRC}_source/last.pt --result_path workdir

# 3-4. Sampling and evaluation, for several seeds
for ((i = 0; i < NUM_SEEDS; i++)); do
  OUT=results/${SRC}-${TRG}/seed_${i}
  python tools/generate.py --weight_path workdir/${SRC}-${TRG}/last.pt \
    --dataset_path ${DATA}/${TRG} --split test --result_path ${OUT} --num_sample 10 --seed ${i}
  python tools/evaluate.py --pred_path ${OUT} --gt_path ${DATA}/${TRG}/test/lbl --n_mask 10 | tail -n 1
done
