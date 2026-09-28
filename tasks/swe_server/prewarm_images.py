#!/usr/bin/env python
"""Prewarm the SWE-bench image cache by PULLING the prebuilt instance images once, up front.

Computes each instance's exact image key (via make_test_spec, same as the harness) and pulls it
with the container CLI ($SWE_CONTAINER_RUNTIME: docker or podman) into the store the grading
server uses. With SWE_HARNESS_CACHE_LEVEL=instance the pulled image persists, so every later
grade of that instance (repeated draws, PatchRepair loops, preferences) is a cache hit instead
of a cold pull. Pulling (rather than building) also works on rootless Podman hosts without
subordinate UID ranges, where building the images fails. Idempotent: an already-present image
is a fast no-op.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset

CONTAINER_RUNTIME = os.environ.get("SWE_CONTAINER_RUNTIME", "docker").strip() or "docker"


def _pull(ref: str, timeout: int):
    try:
        r = subprocess.run([CONTAINER_RUNTIME, "pull", ref], capture_output=True, text=True, timeout=timeout)
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        return ref, r.returncode, (tail[-1] if tail else "")
    except subprocess.TimeoutExpired:
        return ref, -1, f"pull exceeded {timeout}s"
    except Exception as e:  # noqa: BLE001
        return ref, -1, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Prewarm (pull) prebuilt SWE-bench instance images.")
    ap.add_argument("--dataset_name", default="SWE-bench/SWE-bench_Lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--namespace", default="swebench", help="dockerhub namespace of prebuilt images")
    ap.add_argument("--registry", default="docker.io", help="registry host to qualify the pull ref")
    ap.add_argument("--instance_image_tag", default="latest")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--pull_timeout", type=int, default=1800, help="per-image pull deadline (s)")
    ap.add_argument("--instance_ids", nargs="+", required=True)
    a = ap.parse_args()

    dataset = load_swebench_dataset(a.dataset_name, a.split, instance_ids=a.instance_ids)
    refs = []
    for inst in dataset:
        key = make_test_spec(inst, namespace=a.namespace,
                             instance_image_tag=a.instance_image_tag).instance_image_key
        # key already carries the namespace (e.g. "swebench/sweb.eval.x86_64.<id>:latest");
        # qualify with the registry so Podman doesn't guess an unqualified-search host.
        refs.append(key if key.startswith(f"{a.registry}/") else f"{a.registry}/{key}")

    print(f"[prewarm] pulling {len(refs)} prebuilt images (workers={a.max_workers}, timeout={a.pull_timeout}s)",
          flush=True)
    ok = fail = 0
    failed_refs = []
    with ThreadPoolExecutor(max_workers=max(1, a.max_workers)) as ex:
        futs = {ex.submit(_pull, r, a.pull_timeout): r for r in refs}
        for fu in as_completed(futs):
            ref, rc, msg = fu.result()
            if rc == 0:
                ok += 1
                print(f"[prewarm] OK   {ref}", flush=True)
            else:
                fail += 1
                failed_refs.append(ref)
                print(f"[prewarm] FAIL {ref} (rc={rc}) {msg}", flush=True)
    print(f"[prewarm] done: {ok} pulled/cached, {fail} failed", flush=True)
    if failed_refs:
        print("[prewarm] failed refs:\n  " + "\n  ".join(failed_refs), flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
