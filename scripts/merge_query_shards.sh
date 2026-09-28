#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1

BASE=${1:?usage: merge_query_shards.sh <base_dir e.g. results/<task>/<model>/per_query/czt>}

python - "$BASE" <<'PY'
import csv, glob, os, sys
from moo_mcts.objectives import default_spec
from moo_mcts.drivers.per_query import pooled_front_hv
from tasks.registry import DATASETS

base = sys.argv[1]
parts = os.path.normpath(base).split(os.sep)
DIR = parts[parts.index("results") + 1] if "results" in parts else parts[0]
by_name = {ds.name: ds for ds in DATASETS.values()}
dataset = by_name.get(DIR) or DATASETS.get(DIR)
if dataset is None:
    known = sorted(set(by_name) | set(DATASETS))
    sys.exit(f"[merge] cannot map dir '{DIR}' to a task; known result folders: {known}")

prof = dataset.task_profile
kw = {}
if getattr(prof, "cost_ref_tokens", None) is not None:    kw["cost_ref_tokens"] = prof.cost_ref_tokens
if getattr(prof, "latency_ref_calls", None) is not None:  kw["latency_ref_calls"] = prof.latency_ref_calls
spec = default_spec(**kw)

pt_files = sorted(glob.glob(os.path.join(base, "shard_*", "*_query_q*_points.csv")))
if not pt_files:
    sys.exit(f"[merge] no per-query points under {base}/shard_*/*_query_q*_points.csv")
front, hv = pooled_front_hv(pt_files, spec)
n_q = max((f["n_queries"] for f in front), default=0)
names = list(spec.names)
merged_points = os.path.join(base, "points_merged.csv")
with open(merged_points, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["preference_label"] + [f"w_{n}" for n in names] + names + ["n_queries"])
    for f in front:
        w.writerow([f["preference_label"]] + [f"{x:g}" for x in f["weights"]]
                   + [f"{f['objectives'][n]:g}" for n in names] + [f["n_queries"]])

sum_files = sorted(glob.glob(os.path.join(base, "shard_*", "*_query_summary.csv")))
mean_ho = mean_is = None
if sum_files:
    rows, header = [], None
    for f in sum_files:
        with open(f, newline="") as fh:
            r = list(csv.DictReader(fh))
        if r and header is None:
            header = list(r[0].keys())
        rows.extend(r)
    by_q = {int(row["query"]): row for row in rows}
    merged = [by_q[q] for q in sorted(by_q)]
    with open(os.path.join(base, "query_summary_merged.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header); w.writeheader(); w.writerows(merged)
    ho = [float(r["heldout_hv"]) for r in merged]
    isv = [float(r["insample_archive_hv"]) for r in merged]
    mean_ho = sum(ho) / len(ho) if ho else 0.0
    mean_is = sum(isv) / len(isv) if isv else 0.0

print(f"[merge] task={dataset.name}  {len(pt_files)} per-query fronts pooled -> {n_q} queries")
print(f"[merge] HV-of-means (averaged front) = {hv:.4g}   -> {merged_points}")
if mean_ho is not None:
    print(f"[merge] (diagnostic) per-query mean HV = {mean_ho:.4g} | mean in-sample archive HV = {mean_is:.4g}")
PY
