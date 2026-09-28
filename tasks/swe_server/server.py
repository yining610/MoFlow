"""Self-hosted SWE-bench evaluation server used to grade MoFlow's SWE-bench workflows.
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

try:
    from flask import Flask, jsonify, request
except Exception:  # pragma: no cover
    sys.stderr.write("ERROR: Flask is required -- `pip install flask`\n")
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("swe-server")

# Default dataset if a request omits it (clients normally pass it explicitly).
DEFAULT_DATASET = os.environ.get("SWE_DEFAULT_DATASET", "SWE-bench/SWE-bench_Lite")

def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except Exception:
        return default


# Number of concurrent harness RUNS the server will drive. Each run itself parallelizes
# instances via the request's max_workers, so keep this modest to bound Docker load.
SERVER_MAX_WORKERS = _int_env("SWE_SERVER_MAX_WORKERS", 4)
# Hard cap on how long a single harness run may take (seconds) before we kill it.
HARNESS_TIMEOUT = _int_env("SWE_HARNESS_TIMEOUT", 3600)
# Cap on the /result long-poll hold time (seconds) so worker threads free up.
MAX_LONGPOLL = _int_env("SWE_MAX_LONGPOLL", 60)
# Optional: pull prebuilt instance images from this Docker namespace (e.g. "swebench").
# Leave unset to use the harness default. Set to "none" to force local builds.
HARNESS_NAMESPACE = os.environ.get("SWE_HARNESS_NAMESPACE", "").strip()
HARNESS_CACHE_LEVEL = os.environ.get("SWE_HARNESS_CACHE_LEVEL", "env").strip() or "env"
# Keep per-run working dirs for debugging (default: delete on success).
KEEP_WORKDIRS = os.environ.get("SWE_KEEP_WORKDIRS", "0").lower() in {"1", "true", "yes", "on"}
# Container CLI used for best-effort cleanup of leftover eval containers/images.
CONTAINER_RUNTIME = os.environ.get("SWE_CONTAINER_RUNTIME", "docker").strip() or "docker"

_HARNESS_WRAPPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness_wrapper.py")

app = Flask(__name__)
_EXECUTOR = ThreadPoolExecutor(max_workers=SERVER_MAX_WORKERS, thread_name_prefix="swe-harness")
_TASKS: "dict[str, Future]" = {}
_TASKS_LOCK = threading.Lock()


_INSTANCE_LOCKS: "dict[str, threading.Lock]" = {}
_INSTANCE_LOCKS_GUARD = threading.Lock()

_WARMED: "set[str]" = set()
_IMAGE_PERSISTS = (HARNESS_CACHE_LEVEL == "instance")


def _instance_lock(iid: str) -> "threading.Lock":
    with _INSTANCE_LOCKS_GUARD:
        lk = _INSTANCE_LOCKS.get(iid)
        if lk is None:
            lk = threading.Lock()
            _INSTANCE_LOCKS[iid] = lk
        return lk


def _find_report(workdir: str, run_id: str) -> "dict | None":
    """Locate + parse the harness report JSON (the source of truth for resolved counts).
    """
    cands = (
        glob.glob(os.path.join(workdir, f"*{run_id}*.json"))
        + glob.glob(os.path.join(workdir, "*.json"))
        + glob.glob(os.path.join(workdir, "logs", "**", "*.json"), recursive=True)
    )
    per_instance = None
    for path in cands:
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        if not isinstance(d, dict) or not d:
            continue
        # 1) Preferred: the top-level run summary.
        if "resolved_instances" in d:
            return d
        # 2) Fallback: a per-instance report {instance_id: {resolved: bool, ...}}.
        if per_instance is None and all(
            isinstance(v, dict) and "resolved" in v for v in d.values()
        ):
            per_instance = d
    if per_instance is not None:
        submitted = len(per_instance)
        resolved = sum(1 for v in per_instance.values() if v.get("resolved"))
        applied = sum(1 for v in per_instance.values()
                      if v.get("patch_successfully_applied") or v.get("patch_exists"))
        return {
            "submitted_instances": submitted,
            "resolved_instances": resolved,
            "unresolved_instances": submitted - resolved,
            "error_instances": 0,
            "empty_patch_instances": submitted - applied,
        }
    return None


def _canonical_summary(report: dict) -> str:
    """Emit exactly the lines tasks/swe.py:_parse_stdout regexes for."""
    total = report.get("submitted_instances", report.get("total_instances", 0)) or 0
    resolved = report.get("resolved_instances", 0) or 0
    unresolved = report.get("unresolved_instances", 0) or 0
    errors = report.get("error_instances", 0) or 0
    empty = report.get("empty_patch_instances", 0) or 0
    return (
        f"\nInstances submitted: {total}"
        f"\nInstances resolved: {resolved}"
        f"\nInstances unresolved: {unresolved}"
        f"\nInstances with errors: {errors}"
        f"\nInstances with empty patches: {empty}\n"
    )


def _rm_containers_by_name(name_filter: str, log_ctx: str) -> int:
    """Force-remove containers whose name matches `name_filter`; return how many. Best-effort.
    """
    try:
        ps = subprocess.run(
            [CONTAINER_RUNTIME, "ps", "-a", "--filter", f"name={name_filter}", "--format", "{{.ID}}"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:  # noqa: BLE001 -- cleanup is best-effort
        logger.warning("%s: container list failed: %s", log_ctx, e)
        return 0
    ids = [x for x in (ps.stdout or "").split() if x]
    if not ids:
        return 0
    try:
        subprocess.run([CONTAINER_RUNTIME, "rm", "-f", *ids],
                       capture_output=True, text=True, timeout=300)
    except Exception as e:  # noqa: BLE001
        logger.warning("%s: container rm failed: %s", log_ctx, e)
        return 0
    return len(ids)


def _run_harness(task_id: str, predictions: list, max_workers: int, dataset: str) -> dict:
    """write predictions, run the official harness, return {stdout_tail,returncode}."""
    workdir = tempfile.mkdtemp(prefix=f"swe_{task_id[:8]}_")
    preds_path = os.path.join(workdir, "predictions.jsonl")
    iids = sorted({str(p.get("instance_id", "")) for p in predictions if p.get("instance_id")})

    # Decide which per-instance locks to HOLD across this run.
    build_locks = []
    if _IMAGE_PERSISTS:
        for i in iids:
            with _INSTANCE_LOCKS_GUARD:
                warm = i in _WARMED
            if warm:
                continue                       # image cached -> run concurrently, no lock
            lk = _instance_lock(i)
            lk.acquire()
            with _INSTANCE_LOCKS_GUARD:
                warm = i in _WARMED
            if warm:
                lk.release()                   # another grade built it while we waited
            else:
                build_locks.append(lk)         # we build it; hold until warmed is marked
    else:
        for i in iids:                         # image removed after each run -> serialize fully
            lk = _instance_lock(i)
            lk.acquire()
            build_locks.append(lk)

    report = None
    tail = ""
    returncode = -1
    try:
        try:
            with open(preds_path, "w") as f:
                for p in predictions:
                    f.write(json.dumps(p) + "\n")

            cmd = [
                sys.executable, _HARNESS_WRAPPER,
                "--dataset_name", dataset,
                "--predictions_path", preds_path,
                "--max_workers", str(max_workers),
                "--run_id", task_id,
                "--cache_level", HARNESS_CACHE_LEVEL,
            ]
            if HARNESS_NAMESPACE:
                cmd += ["--namespace", HARNESS_NAMESPACE]

            mode = "build-serialized" if build_locks else "concurrent"
            logger.info("[%s] running harness (%s): %s", task_id[:12], mode, " ".join(cmd))
            proc = subprocess.run(
                cmd, cwd=workdir, capture_output=True, text=True, timeout=HARNESS_TIMEOUT,
            )
            stdout = (proc.stdout or "") + "\n---STDERR---\n" + (proc.stderr or "")
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            stdout = f"HARNESS TIMEOUT after {HARNESS_TIMEOUT}s"
            returncode = -1
        except Exception as e:  # pragma: no cover
            stdout = f"HARNESS LAUNCH ERROR: {type(e).__name__}: {e}"
            returncode = -1

        # Append canonical summary from the report JSON (robust to harness print-format drift).
        report = _find_report(workdir, task_id)
        tail = stdout[-6000:]
        if report is not None:
            tail += _canonical_summary(report)
            logger.info("[%s] resolved=%s submitted=%s", task_id[:12],
                        report.get("resolved_instances"), report.get("submitted_instances"))
            if _IMAGE_PERSISTS:
                with _INSTANCE_LOCKS_GUARD:
                    _WARMED.update(iids)       # image built & kept -> unblock concurrent grades
        else:
            logger.warning("[%s] no report JSON found (rc=%s) -- client will see total=0",
                           task_id[:12], returncode)
    finally:
        for lk in reversed(build_locks):
            lk.release()

    if not KEEP_WORKDIRS:
        n = _rm_containers_by_name(task_id, f"[{task_id[:12]}] cleanup")
        if n:
            logger.info("[%s] reaped %d leaked container(s)", task_id[:12], n)

    if not KEEP_WORKDIRS and report is not None:
        try:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
    else:
        logger.info("[%s] workdir kept: %s", task_id[:12], workdir)

    return {"stdout_tail": tail, "returncode": returncode}


@app.get("/health")
def health():
    ok = True
    detail = {}
    try:
        import swebench  # noqa: F401
        detail["swebench"] = getattr(sys.modules["swebench"], "__version__", "unknown")
    except Exception as e:
        ok = False
        detail["swebench_error"] = str(e)
    try:
        import docker as _docker
        detail["docker"] = str(_docker.from_env().version().get("Version", "unknown"))[:60]
    except Exception as e:
        ok = False
        detail["docker_error"] = f"{type(e).__name__}: {str(e)[:120]} (is the Docker/Podman API socket up?)"
    return jsonify({"status": "ok" if ok else "degraded", **detail}), (200 if ok else 503)


@app.get("/queue")
def queue():
    with _TASKS_LOCK:
        futs = list(_TASKS.values())
    running = sum(1 for f in futs if f.running())
    done = sum(1 for f in futs if f.done())
    pending = sum(1 for f in futs if not f.running() and not f.done())
    return jsonify({"pending": pending, "running": running, "done": done,
                    "max_workers": SERVER_MAX_WORKERS}), 200


@app.post("/evaluate")
def evaluate():
    body = request.get_json(force=True, silent=True) or {}
    predictions = body.get("predictions") or []
    if not predictions:
        return jsonify({"error": "no predictions"}), 400
    max_workers = int(body.get("max_workers", 4) or 4)
    dataset = body.get("dataset") or DEFAULT_DATASET

    task_id = uuid.uuid4().hex
    fut = _EXECUTOR.submit(_run_harness, task_id, predictions, max_workers, dataset)
    with _TASKS_LOCK:
        _TASKS[task_id] = fut
    logger.info("submitted task=%s n_pred=%d dataset=%s", task_id[:12], len(predictions), dataset)
    return jsonify({"task_id": task_id}), 202


@app.get("/result/<task_id>")
def result(task_id):
    with _TASKS_LOCK:
        fut = _TASKS.get(task_id)
    if fut is None:
        return jsonify({"state": "FAILURE", "error": "unknown task_id"}), 200

    try:
        wait = int(request.args.get("timeout", 0))
    except Exception:
        wait = 0
    wait = max(0, min(wait, MAX_LONGPOLL))

    try:
        res = fut.result(timeout=wait)  # blocks up to `wait` s (long-poll)
    except FuturesTimeout:
        return jsonify({"state": "PENDING"}), 202
    except Exception as e:  # _run_harness never raises, but be safe
        return jsonify({"state": "FAILURE", "error": f"{type(e).__name__}: {e}"}), 200

    return jsonify({"state": "SUCCESS", "result": res}), 200


def main():
    ap = argparse.ArgumentParser(description="Self-hosted SWE-bench eval server (MoFlow)")
    ap.add_argument("--host", default=os.environ.get("SWE_SERVER_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=_int_env("SWE_SERVER_PORT", 8100))
    args = ap.parse_args()

    # Fail fast with a clear message if the harness isn't installed.
    try:
        import swebench  # noqa: F401
    except Exception:
        sys.stderr.write(
            "ERROR: `swebench` is not importable in this environment. Launch the server under "
            "an interpreter that has it (set SWE_SERVER_PY for scripts/swe/run_swe_server.sh), "
            "or `pip install 'swebench>=4.1,<5' flask docker` on a host with Docker or Podman.\n"
        )
        raise

    logger.info("SWE-bench eval server on %s:%d | server_workers=%d harness_timeout=%ds "
                "cache_level=%s namespace=%s default_dataset=%s",
                args.host, args.port, SERVER_MAX_WORKERS, HARNESS_TIMEOUT,
                HARNESS_CACHE_LEVEL, HARNESS_NAMESPACE or "(harness default)", DEFAULT_DATASET)

    n = _rm_containers_by_name("sweb.eval", "startup sweep")
    if n:
        logger.info("startup: reaped %d leftover eval container(s)", n)
    try:
        subprocess.run([CONTAINER_RUNTIME, "image", "prune", "-f"],
                       capture_output=True, text=True, timeout=300)
    except Exception as e:  # noqa: BLE001 -- best-effort
        logger.warning("startup: image prune failed: %s", e)

    # threaded=True so /result long-polls don't block other requests. SINGLE process only
    # (in-memory task registry).
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
