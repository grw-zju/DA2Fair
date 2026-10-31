#!/bin/bash
# DA2Fair - run all experiments
# Usage: bash scripts/run_all.sh [GPU_ID] [DATA_DIR]

GPU_ID=${1:-0}
DATA_DIR=${2:-data}
CONFIG=${3:-configs/default.yaml}

export CUDA_VISIBLE_DEVICES=$GPU_ID

DATASETS="Epinion Gowalla"
BACKBONES="MF NeuMF VAECF LightGCN"
METHODS="Standard DA2Fair CPFair MultiFR Ada2Fair FAIR FairSort"

echo "=========================================="
echo "DA2Fair Experiment Suite"
echo "GPU: $GPU_ID, Config: $CONFIG"
echo "=========================================="

# Step 1: Prepare data
echo "[Step 1] Preparing data..."
python3 data/prepare_data.py \
  --data_dir $DATA_DIR \
  --datasets $DATASETS

# Step 2: Run all experiments
for DATASET in $DATASETS; do
  for BACKBONE in $BACKBONES; do
    for METHOD in $METHODS; do
      echo ""
      echo "[Running] $DATASET | $BACKBONE | $METHOD"
      python3 run.py \
        --dataset $DATASET \
        --backbone $BACKBONE \
        --method $METHOD \
        --config $CONFIG \
        --gpu $GPU_ID \
        --seeds 42 43 44 45 46
    done
  done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
