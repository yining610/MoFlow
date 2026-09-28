#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENDPOINT_FILE="$SCRIPT_DIR/.swe_endpoint"
HOST="${SWE_SERVER_PUBLIC_HOST:-$(hostname)}"
PY="${SWE_SERVER_PY:-python}"

PORT="$("$PY" -c '
import socket
p = 8100
while p < 8200:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", p)); print(p); break
    except OSError:
        p += 1
    finally:
        s.close()
')"
[ -n "$PORT" ] || { echo "[start_swe] ERROR: no free port in 8100-8199 on $HOST" >&2; exit 1; }
export SWE_SERVER_PORT="$PORT"

echo "http://$HOST:$PORT" > "$ENDPOINT_FILE"
trap 'rm -f "$ENDPOINT_FILE"' EXIT
echo "[start_swe] published endpoint -> $ENDPOINT_FILE (http://$HOST:$PORT)"

while true; do
    echo "[start_swe] launching server ($(date '+%F %T'))"
    bash "$SCRIPT_DIR/run_swe_server.sh"
    echo "[start_swe] server exited (code=$?) at $(date '+%F %T') -- restarting in 3s" >&2
    sleep 3
done
