cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

BASE_URL=${BASE_URL:-http://localhost:4000}
API_KEY_ENV=${API_KEY_ENV:-OPENAI_API_KEY}
MODEL=${MODEL:-gpt-5-mini}
MODEL_SLUG=${MODEL_SLUG:-${MODEL//\//_}}

DEVICE=${DEVICE:-cpu}
GNN_DIR=${GNN_DIR:-results/gnn_bundles/$MODEL_SLUG/multitask}
GNN_BUNDLE=${GNN_BUNDLE:-$GNN_DIR/accuracy-robustness-consistency.pkl}

WANDB_FLAG=""
if [ "${WANDB:-0}" = "1" ]; then WANDB_FLAG="--wandb"; fi

export PYTHONUNBUFFERED=1
