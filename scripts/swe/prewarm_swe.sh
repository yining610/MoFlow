#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

source "$ROOT/tasks/swe_server/swe_env.sh"

PY="${SWE_SERVER_PY:-python}"
SPLIT_DIR="${1:-$ROOT/data/swe_lite_splits}"
DATASET="${SWE_DATASET:-SWE-bench/SWE-bench_Lite}"
PREWARM_WORKERS="${SWE_PREWARM_WORKERS:-8}"

command -v "$PY" >/dev/null 2>&1 || { echo "[prewarm] ERROR: no interpreter '$PY' (set SWE_SERVER_PY)" >&2; exit 1; }
[ -d "$SPLIT_DIR" ] || { echo "[prewarm] ERROR: split dir not found: $SPLIT_DIR" >&2; exit 1; }

IDS=$("$PY" - "$SPLIT_DIR" <<'PYEOF'
import json, os, sys
split_dir = sys.argv[1]
ids = []
for fname in ("validate.jsonl", "test.jsonl"):
    path = os.path.join(split_dir, fname)
    if not os.path.exists(path):
        continue
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            iid = json.loads(line).get("instance_id")
            if iid:
                ids.append(iid)
print(" ".join(sorted(set(ids))))
PYEOF
)
N=$(echo $IDS | wc -w)
[ "$N" -gt 0 ] || { echo "[prewarm] ERROR: no instance_ids found under $SPLIT_DIR" >&2; exit 1; }

echo "[prewarm] pulling $N images from $SPLIT_DIR | dataset=$DATASET workers=$PREWARM_WORKERS runtime=$SWE_CONTAINER_RUNTIME"
exec "$PY" "$ROOT/tasks/swe_server/prewarm_images.py" \
    --dataset_name "$DATASET" \
    --split test \
    --namespace swebench \
    --max_workers "$PREWARM_WORKERS" \
    --pull_timeout "$SWE_DOCKER_TIMEOUT" \
    --instance_ids $IDS
