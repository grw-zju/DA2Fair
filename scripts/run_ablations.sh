#!/bin/bash
# Run all ablation experiments from Table IV
# Usage: bash scripts/run_ablations.sh [GPU_ID] [DATASET] [CONFIG]

GPU_ID=${1:-0}
DATASET=${2:-Epinion}
CONFIG=${3:-configs/default.yaml}

export CUDA_VISIBLE_DEVICES=$GPU_ID

COMMON_ARGS="--dataset $DATASET --backbone LightGCN --config $CONFIG --gpu $GPU_ID --seeds 42 43 44 45 46"

echo "=========================================="
echo "Running Ablation Experiments (Table IV)"
echo "Dataset: $DATASET, GPU: $GPU_ID"
echo "=========================================="

# Backbone (Standard)
echo "[Running] Backbone"
python3 run.py $COMMON_ARGS --method Standard

# U-G: user-only, G-only, activity-aware, equal
echo "[Running] U-G"
python3 run.py $COMMON_ARGS --method Ablation \
  --align_sides user --align_terms G --grad_coordinator equal

# I-G: item-only, G-only, activity-aware, equal
echo "[Running] I-G"
python3 run.py $COMMON_ARGS --method Ablation \
  --align_sides item --align_terms G --grad_coordinator equal

# UI-G: both sides, G-only, activity-aware, equal
echo "[Running] UI-G"
python3 run.py $COMMON_ARGS --method Ablation \
  --align_sides both --align_terms G --grad_coordinator equal

# UI-L: both sides, L-only, activity-aware, equal
echo "[Running] UI-L"
python3 run.py $COMMON_ARGS --method Ablation \
  --align_sides both --align_terms L --grad_coordinator equal

# UI-G+L (Uniform): both sides, G+L, uniform, equal
echo "[Running] UI-G+L (Uniform)"
python3 run.py $COMMON_ARGS --method Ablation \
  --align_sides both --align_terms GL --use_uniform_marginal --grad_coordinator equal

# UI-G+L+Nash-MTL (Uniform): both sides, G+L, uniform, Nash-MTL
echo "[Running] UI-G+L+Nash-MTL (Uniform)"
python3 run.py $COMMON_ARGS --method Ablation \
  --align_sides both --align_terms GL --use_uniform_marginal --grad_coordinator nash_mtl

# U-IDAlignM+Equal: both sides, G+L, activity-aware, equal
echo "[Running] U-IDAlignM+Equal"
python3 run.py $COMMON_ARGS --method Ablation \
  --align_sides both --align_terms GL --grad_coordinator equal

# U-IDAlignM+Nash-MTL: both sides, G+L, activity-aware, Nash-MTL
echo "[Running] U-IDAlignM+Nash-MTL"
python3 run.py $COMMON_ARGS --method Ablation \
  --align_sides both --align_terms GL --grad_coordinator nash_mtl

# DA2Fair: both sides, G+L, activity-aware, NEMO
echo "[Running] DA2Fair"
python3 run.py $COMMON_ARGS --method Ablation \
  --align_sides both --align_terms GL --grad_coordinator nemo

echo ""
echo "=========================================="
echo "All ablation experiments completed!"
echo "=========================================="
