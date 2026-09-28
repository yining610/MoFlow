#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source scripts/gnn/_paths.sh

HIDDEN=${HIDDEN:-64}
ROLE_DIM=${ROLE_DIM:-8}
LAYERS=${LAYERS:-2}

DATA_ARGS=()
if [[ -f "$DATASET" ]]; then
    DATA_ARGS=(--dataset "$DATASET")
    echo "[train] using pooled dataset from phase 1: $DATASET"
else
    echo "[train] no phase-1 dataset at $DATASET; pooling checkpoints directly"
fi

python scripts/gnn/train_gnn.py \
    --tasks "$TASKS" \
    --model-slug "$MODEL_SLUG" \
    --exp "$EXP" \
    ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} \
    --device "${TRAIN_DEVICE:-auto}" \
    --epochs 600 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --patience 100 \
    --hidden "$HIDDEN" \
    --role-dim "$ROLE_DIM" \
    --layers "$LAYERS" \
    --seed 0 \
    $WANDB_FLAG --wandb-run-name "gnn_multitask_$MODEL_SLUG" \
    --out-bundle "$GNN_BUNDLE" \
    --metrics-out "$GNN_DIR/metrics.json" \
    --curve-out "$GNN_DIR/curve.csv" \
    --plots "$GNN_DIR/plots" \
    "$@"

echo "[train] multitask GNN bundle -> $GNN_BUNDLE"
