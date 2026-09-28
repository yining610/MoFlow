# Shared container/harness environment for the SWE-bench grading server AND every tool that must
# use the SAME image store (prewarm, file-context extraction). SOURCE this file; do not exec it.
# It is idempotent and honours pre-set overrides (every export uses ${VAR:-default}).

export HF_HOME="${SWE_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"   # HF dataset cache (SWE_HF_HOME wins)

# --- container runtime ------------------------------------------------------------------------
# docker (default) or podman. The harness talks to the Docker API through the docker SDK; for
# rootless Podman we serve that API from a user-writable socket (started by run_swe_server.sh).
export SWE_CONTAINER_RUNTIME="${SWE_CONTAINER_RUNTIME:-docker}"
if [ "$SWE_CONTAINER_RUNTIME" = "podman" ]; then
    if [ -z "${XDG_RUNTIME_DIR:-}" ] || [ ! -w "${XDG_RUNTIME_DIR:-/nonexistent}" ]; then
        export XDG_RUNTIME_DIR="/tmp/podman-run-$(id -u)"
        unset DOCKER_HOST   # drop an inherited socket path that is not writable here; re-derived below
    fi
    mkdir -p "$XDG_RUNTIME_DIR/podman" 2>/dev/null || true
    chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true
    export DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/podman/podman.sock}"
fi

# --- harness cache policy ---------------------------------------------------------------------
# instance: KEEP per-instance images so re-grades of the SAME instance (repeated draws, PatchRepair
# loops, several preferences) are cache hits, not re-pulls. The 70 SWE-bench Lite instances of the
# frozen split need ~180 GB of image storage; use 'env' if disk is tight (re-pulls every grade).
export SWE_HARNESS_CACHE_LEVEL="${SWE_HARNESS_CACHE_LEVEL:-instance}"

# Widen the docker-SDK client timeout so slow first pulls don't trip the 60s default
# (harness_wrapper.py reads this).
export SWE_DOCKER_TIMEOUT="${SWE_DOCKER_TIMEOUT:-1800}"

export SWE_SERVER_MAX_WORKERS="${SWE_SERVER_MAX_WORKERS:-12}"   # concurrent harness runs
