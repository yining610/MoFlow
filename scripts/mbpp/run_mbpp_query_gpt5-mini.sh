#!/usr/bin/env bash
MODEL=${MODEL:-gpt-5-mini}
source "$(dirname "$0")/../env.sh"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

NQUERIES=${NQUERIES:-20}
SHARD=${SHARD:-1}
NSHARDS=${NSHARDS:-1}
OUT=results/mbpp/$MODEL_SLUG/per_query/czt/shard_${SHARD}of${NSHARDS}
mkdir -p "$OUT"
[ -f "$GNN_BUNDLE" ] || echo "[run] WARN: GNN bundle $GNN_BUNDLE not found; interior nodes fall back to the analytic prior"

python -m moo_mcts.cli build-query \
    --task mbpp \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key-env "$API_KEY_ENV" \
    --predictor gnn \
    --device "$DEVICE" \
    --split test \
    --n-queries "$NQUERIES" \
    --query-shard "${SHARD}/${NSHARDS}" \
    --paraphrases 3 \
    --samples 3 \
    --max-depth 6 \
    --realizations 3 \
    --uncertainty-gate 1.0 \
    --trials 60 \
    --es-tol 0.01 \
    --es-patience 8 \
    --es-min-trials 20 \
    --max-workers 8 \
    --seed 0 \
    --selector czt \
    $WANDB_FLAG --wandb-run-name "mbpp_query_${MODEL_SLUG}_shard${SHARD}of${NSHARDS}" \
    --log-level INFO \
    --save \
    --bundle "$GNN_BUNDLE" \
    --out-dir "$OUT" \
    "$@"
