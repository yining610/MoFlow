#!/usr/bin/env python
"""Hypervolume of a MoFlow front, as reported in the paper.

Each input is one of
  * a points.csv written by `build-task` / `inference` (one row per served preference, raw
    held-out objectives) or points_merged.csv from scripts/merge_query_shards.sh: the
    held-out hypervolume of Tables 2 and 3;
  * a *_task_checkpoint.pkl saved by `build-task --save`: the in-sample hypervolume of the
    search's executed front (all workflows it ran on the build split), as in Table 4.

The vectors are put in maximize form [accuracy, -cost, -latency, robustness, consistency],
reduced to their Pareto front, normalized per axis with the default objective spec (cost in
units of 20k tokens, latency in units of 8 calls), and measured against the nadir
r0 = [0, -3e5 tokens, -20 calls, 0, 0]. Larger is better; the reference box has volume 37.5.

    python scripts/eval_hypervolume.py results/aime_2026/gpt-5-mini/per_task/czt/points.csv
    python scripts/eval_hypervolume.py results/*/gpt-5-mini/per_task/czt/points.csv
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from moo_mcts.ccs import geometry as geo
from moo_mcts.ccs.ccs import CCS
from moo_mcts.objectives import CANONICAL_ORDER, default_spec

PAPER_REF = "0,-300000,-20,0,0"


def csv_vectors(path: str, spec) -> np.ndarray:
    """Maximize-form vectors from a points.csv (raw objective columns)."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return np.array(
        [spec.assemble(**{n: float(r[n]) for n in CANONICAL_ORDER}) for r in rows], dtype=float
    ).reshape(-1, spec.D)


def checkpoint_vectors(path: str) -> np.ndarray:
    """Maximize-form vectors of the executed front stored in a search checkpoint.

    The checkpoint embeds GNN tensors that may have been saved on a GPU, so torch.load is
    pinned to CPU while unpickling.
    """
    try:
        import torch
    except ImportError:
        torch = None
    load = torch.load if torch is not None else None
    if torch is not None:
        torch.load = lambda *a, **kw: load(*a, **{**kw, "map_location": "cpu"})
    try:
        with open(path, "rb") as fh:
            snap = pickle.load(fh)
    finally:
        if torch is not None:
            torch.load = load
    return np.array([e.vector for e in snap["archive_entries"]], dtype=float)


def hypervolume(vectors: np.ndarray, spec, ref: np.ndarray) -> tuple[float, int]:
    """(hypervolume of the Pareto front of `vectors` w.r.t. `ref`, front size)."""
    if len(vectors) == 0:
        return 0.0, 0
    front = vectors[geo.pareto_front(vectors)]
    hv = CCS(points=spec.normalize(front)).hypervolume(spec.normalize(ref))
    return float(hv), len(front)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="points.csv / points_merged.csv / *_checkpoint.pkl")
    p.add_argument("--ref", default=PAPER_REF,
                   help="nadir in raw units: accuracy,-cost,-latency,robustness,consistency "
                        "(default: %(default)s, the paper's reference point)")
    args = p.parse_args(argv)

    spec = default_spec()
    ref = np.array([float(x) for x in args.ref.split(",")], dtype=float)
    if ref.shape != (spec.D,):
        p.error(f"--ref needs {spec.D} comma-separated values")

    for path in args.paths:
        if path.endswith(".pkl"):
            vectors, kind = checkpoint_vectors(path), "in-sample"
        else:
            vectors, kind = csv_vectors(path, spec), "held-out"
        hv, n_front = hypervolume(vectors, spec, ref)
        print(f"{path}\n  {kind} HV = {hv:.2f}  ({len(vectors)} points, {n_front} on the front)")


if __name__ == "__main__":
    main()
