_SWE_ENDPOINT_FILE=scripts/swe/.swe_endpoint
if [ -f "$_SWE_ENDPOINT_FILE" ]; then
    export SWE_SERVER_URL="$(cat "$_SWE_ENDPOINT_FILE")"
fi
export SWE_SERVER_URL="${SWE_SERVER_URL:-http://127.0.0.1:8100}"

export SWE_POLL_MAX_CONN_ERRORS="${SWE_POLL_MAX_CONN_ERRORS:-30}"
export SWE_SUBMIT_RETRIES="${SWE_SUBMIT_RETRIES:-15}"
export SWE_DATASET="${SWE_DATASET:-SWE-bench/SWE-bench_Lite}"
export SWE_EVAL_TIMEOUT="${SWE_EVAL_TIMEOUT:-7200}"

_swe_code=$(curl -sS -m 30 -o /dev/null -w "%{http_code}" "${SWE_SERVER_URL%/}/health")
if [ "$_swe_code" != "200" ]; then
    echo "[swe] ERROR: ${SWE_SERVER_URL%/}/health -> $_swe_code (want 200)." >&2
    echo "[swe]        Start the grading server first: bash scripts/swe/start_swe_server.sh" >&2
    exit 1
fi
echo "[swe] grading via $SWE_SERVER_URL on $SWE_DATASET (timeout ${SWE_EVAL_TIMEOUT}s)"
