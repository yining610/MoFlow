#!/usr/bin/env python
"""Offline oracle file-context extraction for SWE-bench grounding.

The model was being asked to write a git diff for a file it never saw, so its context lines
were hallucinated and patches failed to apply even after header repair (see tasks/swe.py
_recount_hunks). This tool builds the missing ingredient: for every instance in the frozen
split, it reads the SOURCE FILE(S) THE GOLD PATCH EDITS -- at the base commit, verbatim --
out of the prebuilt instance image we already have cached, and writes {instance_id -> context
block} to a JSON cache (data/swe_lite_splits/file_context.json). `_swe_render_payload` in
tasks/swe.py appends that block to the model-facing payload, so every method sees identical
grounding.

This is DATA PREP, not inference: it runs the images offline via the container CLI
($SWE_CONTAINER_RUNTIME: docker or podman; same store as prewarm/grading), and the
workflow/operator interface stays (text -> text). It supplies file *content* only (never the
gold patch), i.e. standard "oracle retrieval".

The frozen split already ships this cache; re-run only after re-splitting. Use a Python
with swebench installed, after sourcing swe_env.sh:
    source tasks/swe_server/swe_env.sh
    python tasks/swe_server/extract_file_context.py
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset

CONTAINER_RUNTIME = os.environ.get("SWE_CONTAINER_RUNTIME", "docker").strip() or "docker"

_DELIM = "@@@__SWE_FILECTX__@@@"
_HEADER = (
    "Below is the current content of the source file(s) most relevant to this issue, shown "
    "exactly as they appear in the repository at the base commit. Edit these file(s): copy "
    "context lines into your diff verbatim (byte-for-byte) so the patch applies cleanly."
)


def _touched_files(patch: str) -> list[str]:
    """Source files the gold patch edits (oracle set), from the `diff --git a/PATH b/PATH` lines."""
    files = re.findall(r"^diff --git a/(\S+) b/\S+", patch, re.M)
    if not files:  # fall back to the ---/+++ headers for non-git-formatted gold patches
        files = re.findall(r"^\+\+\+ b/(\S+)", patch, re.M)
    return sorted(set(files))


def _render_context(files_content: dict[str, str]) -> str:
    parts = [_HEADER]
    for path, content in files_content.items():
        parts.append(f"[start of {path}]\n{content.rstrip(chr(10))}\n[end of {path}]")
    return "\n\n".join(parts)


def _extract_one(inst: dict, namespace: str, registry: str, tag: str, timeout: int):
    iid = inst["instance_id"]
    base = inst["base_commit"]
    files = _touched_files(inst.get("patch") or "")
    if not files:
        return iid, None, "no touched files in gold patch"
    ref = make_test_spec(inst, namespace=namespace, instance_image_tag=tag).instance_image_key
    if not ref.startswith(f"{registry}/"):
        ref = f"{registry}/{ref}"
    # One container run per instance; delimit each file so we can split the combined stdout.
    script = "cd /testbed || exit 3\n"
    for f in files:
        script += f'printf "%s %s\\n" "{_DELIM}" "{f}"\n'
        script += f'git show "{base}:{f}" 2>/dev/null || printf "__SWE_MISSING__\\n"\n'
    try:
        r = subprocess.run([CONTAINER_RUNTIME, "run", "--rm", ref, "sh", "-c", script],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return iid, None, f"{CONTAINER_RUNTIME} run exceeded {timeout}s"
    except Exception as e:  # noqa: BLE001
        return iid, None, f"{type(e).__name__}: {e}"
    if r.returncode != 0 and not r.stdout:
        return iid, None, f"rc={r.returncode} {(r.stderr or '').strip()[:160]}"
    # Split the stdout on the delimiter lines.
    files_content: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in r.stdout.split("\n"):
        m = re.match(rf"^{re.escape(_DELIM)} (.+)$", line)
        if m:
            if cur is not None:
                files_content[cur] = "\n".join(buf)
            cur = m.group(1)
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        files_content[cur] = "\n".join(buf)
    missing = [f for f, c in files_content.items() if c.strip() == "__SWE_MISSING__"]
    files_content = {f: c for f, c in files_content.items() if c.strip() != "__SWE_MISSING__"}
    if not files_content:
        return iid, None, f"all touched files missing at base ({missing})"
    return iid, _render_context(files_content), (f"ok ({len(files_content)} files"
                                                 + (f", MISSING {missing}" if missing else "") + ")")


def _split_instance_ids(split_dir: str) -> list[str]:
    ids: list[str] = []
    for name in ("validate.jsonl", "test.jsonl"):
        p = os.path.join(split_dir, name)
        if not os.path.isfile(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("swe_meta")
            iid = json.loads(meta)["instance_id"] if isinstance(meta, str) else rec.get("instance_id")
            if iid:
                ids.append(iid)
    return sorted(set(ids))


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract oracle file context from cached instance images.")
    ap.add_argument("--split_dir", default="data/swe_lite_splits")
    ap.add_argument("--dataset_name", default="SWE-bench/SWE-bench_Lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="output JSON (default: <split_dir>/file_context.json)")
    ap.add_argument("--namespace", default="swebench")
    ap.add_argument("--registry", default="docker.io")
    ap.add_argument("--instance_image_tag", default="latest")
    ap.add_argument("--max_workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--instance_ids", nargs="*", default=None,
                    help="restrict to these ids (default: all ids in the split)")
    a = ap.parse_args()

    ids = a.instance_ids or _split_instance_ids(a.split_dir)
    if not ids:
        print("[filectx] no instance ids found", file=sys.stderr)
        return 1
    dataset = load_swebench_dataset(a.dataset_name, a.split, instance_ids=ids)
    print(f"[filectx] extracting oracle context for {len(dataset)} instances "
          f"(workers={a.max_workers})", flush=True)

    ctx: dict[str, str] = {}
    sizes: list[tuple[str, int]] = []
    fail = 0
    with ThreadPoolExecutor(max_workers=max(1, a.max_workers)) as ex:
        futs = {ex.submit(_extract_one, inst, a.namespace, a.registry,
                          a.instance_image_tag, a.timeout): inst["instance_id"]
                for inst in dataset}
        for fu in as_completed(futs):
            iid, block, msg = fu.result()
            if block:
                ctx[iid] = block
                sizes.append((iid, len(block)))
                print(f"[filectx] OK   {iid:40s} {len(block):>8d} B  {msg}", flush=True)
            else:
                fail += 1
                print(f"[filectx] FAIL {iid:40s} {msg}", flush=True)

    out = a.out or os.path.join(a.split_dir, "file_context.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(ctx, fh)
    os.replace(tmp, out)

    if sizes:
        sizes.sort(key=lambda t: t[1])
        tot = sum(s for _, s in sizes)
        print(f"\n[filectx] wrote {len(ctx)} contexts -> {out} ({fail} failed)")
        print(f"[filectx] size bytes: total={tot} avg={tot // len(sizes)} "
              f"min={sizes[0][1]} max={sizes[-1][1]} ({sizes[-1][0]})")
        big = [(i, s) for i, s in sizes if s > 60_000]
        if big:
            print(f"[filectx] {len(big)} contexts > 60 KB: " + ", ".join(f"{i}={s}" for i, s in big[-8:]))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
