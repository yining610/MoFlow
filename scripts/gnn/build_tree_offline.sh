#!/usr/bin/env bash
TASK=${1:?usage: build_tree_offline.sh <aime|math|mbpp|swe|gpqa|hotpotqa>}
shift
MODEL=${MODEL:-gpt-5-mini}
source "$(dirname "$0")/../env.sh"

TRIALS=${TRIALS:-300}

if [[ ! -f "$GNN_BUNDLE" ]]; then
    echo "[offline] trained bundle not found: $GNN_BUNDLE" >&2
    echo "[offline] run scripts/gnn/collect_data.sh and scripts/gnn/train_gnn.sh first (or set GNN_BUNDLE=<path>)." >&2
    exit 1
fi

case "$TASK" in
    aime) TOP=aime_2026 ;;
    math|mbpp|gpqa|hotpotqa|swe) TOP=$TASK ;;
    *) echo "[offline] unknown task: $TASK" >&2; exit 1 ;;
esac
if [[ "$TASK" == "swe" ]]; then
    source scripts/swe/_swe_client.sh
fi

OUT=results/$TOP/$MODEL_SLUG/per_task/offline

echo "[offline] task=$TASK trials=$TRIALS bundle=$GNN_BUNDLE out=$OUT"
python -m moo_mcts.cli build-task \
    --task "$TASK" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key-env "$API_KEY_ENV" \
    --offline-gnn "$GNN_BUNDLE" \
    --device "${OFFLINE_DEVICE:-auto}" \
    --max-depth 6 \
    --realizations 3 \
    --paraphrases 3 \
    --samples 3 \
    --trials "$TRIALS" \
    --no-early-stop \
    --max-workers 8 \
    --max-concurrency 8 \
    --seed 0 \
    --selector czt \
    --log-level INFO \
    --save \
    --out-dir "$OUT" \
    "$@"
