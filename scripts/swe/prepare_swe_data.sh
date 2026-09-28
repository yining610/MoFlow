#!/usr/bin/env bash
set -euo pipefail
MODEL=${MODEL:-gpt-5-mini}
source "$(dirname "$0")/../env.sh"

python -m moo_mcts.cli data \
    --task swe \
    --val-size 20 \
    --test-size 50 \
    --n 3 \
    --temperature 0.9 \
    --seed 0 \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key-env "$API_KEY_ENV" \
    --log-level INFO \
    "$@"

echo
echo "[prepare] split manifest:"
cat data/swe_lite_splits/split.json
