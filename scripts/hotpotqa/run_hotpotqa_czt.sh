#!/usr/bin/env bash
MODEL=${MODEL:-gpt-5-mini}
source "$(dirname "$0")/../env.sh"

OUT=results/hotpotqa/$MODEL_SLUG/per_task/czt

python -m moo_mcts.cli build-task \
    --task hotpotqa \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key-env "$API_KEY_ENV" \
    --predictor gnn \
    --device "$DEVICE" \
    --max-depth 6 \
    --realizations 3 \
    --paraphrases 3 \
    --samples 3 \
    --trials 100 \
    --es-tol 0.01 \
    --es-patience 10 \
    --es-min-trials 40 \
    --max-workers 8 \
    --max-concurrency 8 \
    --seed 0 \
    --selector czt \
    $WANDB_FLAG --wandb-run-name "hotpotqa_czt_$MODEL_SLUG" \
    --log-level INFO \
    --save \
    --bundle "$OUT/gnn_bundle.pkl" \
    --out-dir "$OUT" \
    "$@"
