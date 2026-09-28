# SWE-bench grading server

A small HTTP wrapper around the official SWE-bench harness. `tasks/swe.py`'s `swe_checker` is a
thin client: it submits `{instance_id, model_patch}` to this server and reads back whether the
instance was resolved. One server can grade for many concurrent runs.

## Requirements

- Docker, or rootless Podman (set `SWE_CONTAINER_RUNTIME=podman`).
- About 200 GB of free disk for the 70 prebuilt SWE-bench Lite instance images of the frozen
  split (`data/swe_lite_splits`, 20 build + 50 held-out). Images are pulled from Docker Hub
  (`swebench/sweb.eval.x86_64.*`) on first use.
- A Python environment with the harness installed. It can be the MoFlow environment or a
  separate one (point `SWE_SERVER_PY` at its interpreter):

  ```bash
  pip install "swebench>=4.1,<5" flask docker   # 5.x changed the harness API used here
  ```

## Start the server

```bash
bash scripts/swe/start_swe_server.sh          # Docker
SWE_CONTAINER_RUNTIME=podman bash scripts/swe/start_swe_server.sh   # rootless Podman
```

The launcher picks a free port at or after 8100, publishes `http://<host>:<port>` to
`scripts/swe/.swe_endpoint` (removed when it exits), and restarts the server if it crashes.
Keep it running for the whole experiment (e.g. in `tmux`, or as a job on a dedicated node).
The SWE run scripts read `.swe_endpoint` automatically; it wins over an inherited
`SWE_SERVER_URL`, which otherwise defaults to `http://127.0.0.1:8100`. If the clients run on a
different machine, make sure they can reach the published host name (override it with
`SWE_SERVER_PUBLIC_HOST`), or export `SWE_SERVER_URL` on the client side and delete the file.

`scripts/swe/run_swe_server.sh` runs the same server in the foreground without the restart loop.

**First grades are slow.** Each instance's first grade pulls a ~2.5 GB image. Optionally pull
all of them up front on the server machine:

```bash
bash scripts/swe/prewarm_swe.sh
```

## Check the setup

`scripts/swe/smoke_test.py` grades one instance's **gold** patch through the server. A resolved
verdict (score `1.0`) proves the whole path (client, server, harness, containers) works:

```bash
python scripts/swe/smoke_test.py --timeout 2400                     # first instance of the split
python scripts/swe/smoke_test.py --instance sqlfluff__sqlfluff-1625
curl -sS "$(cat scripts/swe/.swe_endpoint)/health"                  # {"status":"ok", ...}
```

## Configuration

Server side (`tasks/swe_server/swe_env.sh`; every value can be preset in the environment):

| variable | default | meaning |
| --- | --- | --- |
| `SWE_CONTAINER_RUNTIME` | `docker` | `docker` or `podman` (rootless; the launcher starts its API socket) |
| `SWE_SERVER_PY` | `python` | interpreter with `swebench`, `flask`, `docker` |
| `SWE_SERVER_MAX_WORKERS` | `12` | concurrent harness runs |
| `SWE_HARNESS_CACHE_LEVEL` | `instance` | `instance` keeps per-instance images (fast re-grades); `env` saves disk |
| `SWE_DOCKER_TIMEOUT` | `1800` | Docker-SDK timeout (s) for slow first pulls |
| `SWE_HF_HOME` | `HF_HOME` | Hugging Face cache used to load the SWE-bench dataset |

Client side (read by `tasks/swe.py`; set by `scripts/swe/_swe_client.sh`):

| variable | default | meaning |
| --- | --- | --- |
| `SWE_SERVER_URL` | `http://127.0.0.1:8100` | server URL (`.swe_endpoint` wins when present) |
| `SWE_EVAL_TIMEOUT` | `7200` | per-grade deadline (s); raise it when the queue is deep |
| `SWE_POLL_MAX_CONN_ERRORS` | `30` | consecutive poll failures tolerated (rides out a restart) |
| `SWE_SUBMIT_RETRIES` | `15` | submit attempts with linear backoff |

## Routes

- `GET /health` → `200 {"status":"ok", "swebench":..., "docker":...}` (else `503 degraded`)
- `GET /queue` → `{"pending","running","done","max_workers"}`
- `POST /evaluate` body `{"predictions":[{instance_id,model_patch,model_name_or_path}], "max_workers":int, "dataset":str}` → `202 {"task_id":...}`
- `GET /result/<task_id>?timeout=S` → `202` while running, else `200 {"state":"SUCCESS","result":{"stdout_tail","returncode"}}`

## Data preparation (already done)

`data/swe_lite_splits/file_context.json` holds, for every instance of the frozen split, the
source files its gold patch edits, read at the base commit from the instance image. The model
sees these files (never the patch) so that its diffs apply cleanly. It was produced by
`tasks/swe_server/extract_file_context.py` and only needs regenerating if you re-split the data.
