#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source scripts/gnn/_paths.sh

python scripts/gnn/collect.py \
    --tasks "$TASKS" \
    --model-slug "$MODEL_SLUG" \
    --exp "$EXP" \
    --log-level INFO \
    --save "$DATASET" \
    "$@"
