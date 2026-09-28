#!/usr/bin/env python
"""End-to-end smoke test of the SWE-bench eval server against SWE-bench Lite.

Loads one Lite instance's GOLD patch and grades it through the server via the same client the
MoFlow checker uses (tasks/swe.grade_patch). A resolved verdict (score 1.0) proves the whole
path works: client -> server -> harness wrapper -> Docker/Podman -> image -> patch apply
-> tests -> report parse.

Runs in the MoFlow env (needs only `datasets` + `requests`); the harness itself runs behind
the server. Point it at the server with --url or $SWE_SERVER_URL.

    python scripts/swe/smoke_test.py                       # first instance in the frozen split
    python scripts/swe/smoke_test.py --instance sqlfluff__sqlfluff-1625
    python scripts/swe/smoke_test.py --url http://<server-host>:8100 --timeout 2400

Exit code: 0 if score==1.0, 1 if the server is unreachable/unhealthy, 2 if it graded but did
not resolve.
"""
import argparse
import json
import os
import sys

# Make `import tasks.swe` work when run as `python scripts/swe/smoke_test.py` from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets import load_dataset

from tasks import swe


def _first_instance_from_split() -> str | None:
    """instance_id of the first row in the frozen validation split, if it exists."""
    path = os.path.join("data", "swe_lite_splits", "validate.jsonl")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("swe_meta")
            if isinstance(meta, str):
                meta = json.loads(meta)
            iid = (meta or {}).get("instance_id")
            if iid:
                return iid
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test the SWE-bench server on SWE-bench Lite")
    ap.add_argument("--url", default=os.environ.get("SWE_SERVER_URL", "http://127.0.0.1:8100"),
                    help="server base URL (default: $SWE_SERVER_URL or http://127.0.0.1:8100)")
    ap.add_argument("--instance", default=None,
                    help="Lite instance_id to grade with its gold patch "
                         "(default: first in the frozen split, else the first Lite row)")
    ap.add_argument("--dataset", default=swe.SWE_DATASET, help="HF dataset id to load the gold patch from")
    ap.add_argument("--timeout", type=int, default=2400, help="per-grade timeout seconds")
    a = ap.parse_args()

    os.environ["SWE_SERVER_URL"] = a.url  # tasks.swe reads this
    print(f"[smoke] server = {a.url}")

    # /health first: a clear message beats a mystifying total=0 later.
    import requests
    try:
        r = requests.get(f"{a.url.rstrip('/')}/health", timeout=10)
        print(f"[smoke] /health {r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            print("[smoke] FAIL: server not healthy (is the Docker/Podman API socket up on its host?)")
            sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] FAIL: cannot reach {a.url}: {e}")
        sys.exit(1)

    instance = a.instance or _first_instance_from_split()
    print(f"[smoke] loading gold patch from {a.dataset} ...")
    ds = load_dataset(a.dataset, split="test")
    if instance is None:
        instance = ds[0]["instance_id"]
    try:
        gold = next(x for x in ds if x["instance_id"] == instance)["patch"]
    except StopIteration:
        print(f"[smoke] FAIL: instance {instance!r} not in {a.dataset}")
        sys.exit(2)

    print(f"[smoke] grading gold patch for {instance} (first build takes minutes) ...")
    score = swe.grade_patch(instance, gold, timeout=a.timeout)
    print(f"[smoke] SCORE = {score}   (expect 1.0)")
    if score == 1.0:
        print("[smoke] PASS: server grades Lite correctly end-to-end.")
        sys.exit(0)
    print("[smoke] NOT resolved. Check the server log + the harness run_instance.log for this "
          "instance (a None score means the server/socket was unavailable or produced no report).")
    sys.exit(2)


if __name__ == "__main__":
    main()
