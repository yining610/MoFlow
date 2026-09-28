#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

PY="${SWE_SERVER_PY:-python}"
SERVER="$ROOT/tasks/swe_server/server.py"

source "$ROOT/tasks/swe_server/swe_env.sh"

command -v "$PY" >/dev/null 2>&1 || { echo "[swe-server] ERROR: no interpreter '$PY' (set SWE_SERVER_PY)" >&2; exit 1; }

if [ "$SWE_CONTAINER_RUNTIME" = "podman" ]; then
    SOCK="${DOCKER_HOST#unix://}"
    mkdir -p "$(dirname "$SOCK")"
    if [ ! -S "$SOCK" ]; then
        echo "[swe-server] starting podman API socket: $SOCK"
        podman system service --time=0 "unix://$SOCK" &
        for _ in $(seq 30); do [ -S "$SOCK" ] && break; sleep 0.5; done
        [ -S "$SOCK" ] || { echo "[swe-server] ERROR: podman socket did not come up" >&2; exit 1; }
    fi
fi

echo "[swe-server] $PY :${SWE_SERVER_PORT:-8100} runtime=$SWE_CONTAINER_RUNTIME workers=$SWE_SERVER_MAX_WORKERS cache=$SWE_HARNESS_CACHE_LEVEL hf_home=$HF_HOME"
exec "$PY" "$SERVER" --host "${SWE_SERVER_HOST:-0.0.0.0}" --port "${SWE_SERVER_PORT:-8100}"
