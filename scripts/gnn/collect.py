#!/usr/bin/env python
"""Pool multi-task GNN training data from per-task search checkpoints.
"""
import os
import pickle
import sys
from dataclasses import dataclass, field

import numpy as np

# Make `import moo_mcts` / `import tasks` work when run as a plain script from the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from moo_mcts.logging_util import get_logger
from moo_mcts.objectives import ObjectiveSpec, default_spec
from moo_mcts.valuation.distill import ReplayBuffer, featurize
from moo_mcts.workflow.graph import WorkflowGraph
from tasks.registry import DATASETS

log = get_logger("gnn_collect")

DEFAULT_MODEL_SLUG = "gpt-5-mini"
DEFAULT_EXP = "czt"
DEFAULT_TASKS = ("aime", "math", "mbpp", "gpqa", "hotpotqa", "swe")

_RESULTS_TOPDIR = {
    "aime": "aime_2026",
    "math": "math",
    "mbpp": "mbpp",
    "gpqa": "gpqa",
    "hotpotqa": "hotpotqa",
    "swe": "swe",
}

Pair = tuple[WorkflowGraph, np.ndarray]


def default_checkpoint_path(
    task_key: str, model_slug: str = DEFAULT_MODEL_SLUG, exp: str = DEFAULT_EXP,
    results_root: str = "results",
) -> str:
    """Resolve a task's per-task checkpoint path from its name + the run layout.

    Reproduces the on-disk convention
    `results/<top>/<model_slug>/per_task/<exp>/<dataset.name>_task_checkpoint.pkl`.
    """
    if task_key not in DATASETS:
        raise KeyError(f"unknown task {task_key!r}; known: {sorted(DATASETS)}")
    top = _RESULTS_TOPDIR.get(task_key, task_key)
    stem = f"{DATASETS[task_key].name}_task_checkpoint.pkl"
    return os.path.join(results_root, top, model_slug, "per_task", exp, stem)


def _pairs_from_buffer(buffer: ReplayBuffer) -> tuple[list[Pair], list[Pair]]:
    """Return (train, test) (graph, target[3]) pairs, dropping empty graphs."""
    train = [(g, np.asarray(y, dtype=np.float32)) for g, y in buffer.graphs()
             if not g.is_empty()]
    test = [(g, np.asarray(y, dtype=np.float32)) for g, y in buffer.test_graphs()
            if not g.is_empty()]
    return train, test


def load_checkpoint_pairs(
    path: str, spec: ObjectiveSpec = None,
) -> tuple[list[Pair], list[Pair], dict | None, dict]:
    """Load one checkpoint's replay-buffer pairs, role cache, and a small meta dict.

    - path: path to a `*_task_checkpoint.pkl`.
    - spec: expected objective spec; the buffer's predicted axes must match it.

    Returns `(train_pairs, test_pairs, role_cache, meta)`.
    """
    spec = spec or default_spec()
    with open(path, "rb") as fh:
        snap = pickle.load(fh)
    pstate = snap.get("predictor_state")
    if not pstate or "buffer" not in pstate:
        raise ValueError(f"{path}: no predictor_state.buffer (not a GNN/heuristic run?)")
    buffer: ReplayBuffer = pstate["buffer"]

    got = tuple(buffer.spec.predicted_names)
    want = tuple(spec.predicted_names)
    if got != want:
        raise ValueError(
            f"{path}: predicted axes {got} != expected {want}; incompatible checkpoint"
        )

    train, test = _pairs_from_buffer(buffer)
    role_cache = pstate.get("role_cache")
    meta = {
        "total_trials": int(snap.get("total_trials", 0)),
        "trained": bool(pstate.get("trained", False)),
        "n_train": len(train),
        "n_test": len(test),
        "n_archive": len(snap.get("archive_entries", [])),
    }
    return train, test, role_cache, meta


@dataclass
class CollectConfig:
    """Knobs for regenerating a missing checkpoint via the full search pipeline.

    Only consulted when `collect_missing=True`. Mirrors the run_*_czt.sh defaults.
    """

    model: str = "gpt-5-mini"
    trials: int = 100
    max_depth: int = 6
    paraphrases: int = 3
    samples: int = 3
    realizations: int = 3
    max_workers: int = 16
    max_concurrency: int = 16
    device: str | None = "cpu"
    api_key_env: str = "OPENAI_API_KEY"
    seed: int = 0


def _regenerate_checkpoint(task_key: str, path: str, spec: ObjectiveSpec,
                           cc: CollectConfig) -> None:
    """Run the per-task search to produce `path` (heavy; needs the real backend)."""
    from moo_mcts.config import RunConfig, SearchConfig
    from moo_mcts.drivers import per_task
    from moo_mcts.serve.save_policy import default_bundle_path
    from tasks.hf_dataset import load_hf_split

    log.warning("regenerating %s via build_task_tree (task=%s, trials=%d) -- this makes real "
                "LLM calls", path, task_key, cc.trials)
    dataset_spec = DATASETS[task_key]
    task, _heldout = load_hf_split(dataset_spec)

    cfg = RunConfig()
    cfg.search = SearchConfig(n_trials=cc.trials, max_depth=cc.max_depth, seed=cc.seed)
    cfg.search.chance_samples = cc.samples
    cfg.max_workers = cc.max_workers
    cfg.backend.max_concurrency = cc.max_concurrency
    cfg.backend.realizations = cc.realizations
    cfg.backend.model = cc.model
    cfg.backend.api_key_env = cc.api_key_env
    cfg.reward.n_paraphrases = cc.paraphrases
    cfg.predictor.kind = "gnn"
    cfg.predictor.device = cc.device

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    per_task.build_task_tree(
        task, cfg, spec,
        checker=dataset_spec.checker,
        task_profile=dataset_spec.task_profile,
        checkpoint_path=path,
        resume=True,
        bundle_path=default_bundle_path(cfg.backend.model, spec),
        public_tester=dataset_spec.public_tester,
    )


def collect_task(
    task_key: str, path: str = None, spec: ObjectiveSpec = None,
    model_slug: str = DEFAULT_MODEL_SLUG, exp: str = DEFAULT_EXP,
    collect_missing: bool = False, collect_config: CollectConfig = None,
) -> tuple[list[Pair], list[Pair], dict | None, dict]:
    """Load (or optionally regenerate) one task's pooled pairs.

    Returns `(train_pairs, test_pairs, role_cache, meta)`; empties + `found=False` in meta
    when the checkpoint is absent and `collect_missing` is off.
    """
    spec = spec or default_spec()
    path = path or default_checkpoint_path(task_key, model_slug, exp)

    if not os.path.exists(path):
        if collect_missing:
            _regenerate_checkpoint(task_key, path, spec, collect_config or CollectConfig())
        else:
            log.warning("task %s: no checkpoint at %s -- skipping (pass collect_missing=True "
                        "to regenerate)", task_key, path)
            return [], [], None, {"task": task_key, "path": path, "found": False,
                                  "n_train": 0, "n_test": 0}

    train, test, role_cache, meta = load_checkpoint_pairs(path, spec)
    meta.update({"task": task_key, "path": path, "found": True})
    log.info("task %s: %d train + %d test pairs from %s",
             task_key, meta["n_train"], meta["n_test"], path)
    return train, test, role_cache, meta


def _merge_role_caches(caches: list[dict | None]) -> dict | None:
    """Union the per-checkpoint role -> embedding caches (identical roles embed equally)."""
    vectors: dict[str, np.ndarray] = {}
    dim = None
    for rc in caches:
        if not rc:
            continue
        dim = rc.get("dim", dim)
        for role, vec in (rc.get("vectors") or {}).items():
            vectors[role] = np.asarray(vec, dtype=np.float32)
    if dim is None:
        return None
    return {"dim": int(dim), "vectors": vectors}


def _build_combined_buffer(spec: ObjectiveSpec, train: list[Pair],
                           test: list[Pair]) -> ReplayBuffer:
    """Populate a fresh ReplayBuffer's train/test deques directly (partition preserved)."""
    buf = ReplayBuffer(spec)
    for g, y in train:
        buf._train_X.append(featurize(g))
        buf._train_Y.append(np.asarray(y, dtype=np.float32))
        buf._train_G.append(g)
        for node in g.nodes:
            buf._intern_role(node.role)
    for g, y in test:
        buf._test_X.append(featurize(g))
        buf._test_Y.append(np.asarray(y, dtype=np.float32))
        buf._test_G.append(g)
        for node in g.nodes:
            buf._intern_role(node.role)
    return buf


@dataclass
class MultiTaskDataset:
    """Pooled cross-task GNN training set."""

    spec: ObjectiveSpec
    train_pairs: list[Pair]
    test_pairs: list[Pair]
    role_cache: dict | None
    buffer: ReplayBuffer
    per_task_stats: list[dict] = field(default_factory=list)
    # per-task pairs (kept so training can report per-benchmark train/test performance)
    train_by_task: dict[str, list[Pair]] = field(default_factory=dict)
    test_by_task: dict[str, list[Pair]] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.train_pairs)

    @property
    def n_test(self) -> int:
        return len(self.test_pairs)

    def summary(self) -> str:
        head = f"{'task':<12}{'found':<7}{'n_train':>9}{'n_test':>9}{'trials':>8}"
        lines = [head, "-" * len(head)]
        for s in self.per_task_stats:
            lines.append(
                f"{s['task']:<12}{str(s.get('found', False)):<7}"
                f"{s.get('n_train', 0):>9}{s.get('n_test', 0):>9}"
                f"{s.get('total_trials', 0) if s.get('found') else '-':>8}"
            )
        lines.append("-" * len(head))
        lines.append(f"{'TOTAL':<12}{'':<7}{self.n_train:>9}{self.n_test:>9}")
        return "\n".join(lines)


def collect_multitask(
    task_keys=DEFAULT_TASKS, spec: ObjectiveSpec = None,
    model_slug: str = DEFAULT_MODEL_SLUG, exp: str = DEFAULT_EXP,
    checkpoint_overrides: dict[str, str] = None,
    collect_missing: bool = False, collect_config: CollectConfig = None,
) -> MultiTaskDataset:
    """Pool train/test pairs + role caches across `task_keys` into one dataset."""
    spec = spec or default_spec()
    overrides = checkpoint_overrides or {}

    all_train: list[Pair] = []
    all_test: list[Pair] = []
    caches: list[dict | None] = []
    stats: list[dict] = []
    train_by_task: dict[str, list[Pair]] = {}
    test_by_task: dict[str, list[Pair]] = {}
    for task_key in task_keys:
        train, test, role_cache, meta = collect_task(
            task_key, path=overrides.get(task_key), spec=spec,
            model_slug=model_slug, exp=exp,
            collect_missing=collect_missing, collect_config=collect_config,
        )
        all_train.extend(train)
        all_test.extend(test)
        caches.append(role_cache)
        stats.append(meta)
        if train:
            train_by_task[task_key] = train
        if test:
            test_by_task[task_key] = test

    role_cache = _merge_role_caches(caches)
    buffer = _build_combined_buffer(spec, all_train, all_test)
    return MultiTaskDataset(
        spec=spec, train_pairs=all_train, test_pairs=all_test,
        role_cache=role_cache, buffer=buffer, per_task_stats=stats,
        train_by_task=train_by_task, test_by_task=test_by_task,
    )


def save_dataset(ds: MultiTaskDataset, path: str) -> None:
    """Pickle the pooled dataset to `path` for phase 2 to consume.

    Stores only stable, module-level types (WorkflowGraph, numpy arrays, dicts) -- NOT the
    MultiTaskDataset dataclass itself -- so the pickle unpickles regardless of how this file
    was invoked (as a script `__main__` or imported as `scripts.gnn.collect`). The buffer is
    rebuilt on load from the train/test pairs.
    """
    payload = {
        "predicted_names": list(ds.spec.predicted_names),
        "train_pairs": ds.train_pairs,
        "test_pairs": ds.test_pairs,
        "role_cache": ds.role_cache,
        "per_task_stats": ds.per_task_stats,
        "train_by_task": ds.train_by_task,
        "test_by_task": ds.test_by_task,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_dataset(path: str, spec: ObjectiveSpec = None) -> MultiTaskDataset:
    """Rebuild a MultiTaskDataset from a `save_dataset` pickle (buffer rebuilt from pairs)."""
    spec = spec or default_spec()
    with open(path, "rb") as fh:
        p = pickle.load(fh)
    got, want = tuple(p.get("predicted_names", ())), tuple(spec.predicted_names)
    if got != want:
        raise ValueError(f"{path}: dataset axes {got} != expected {want}; re-run phase 1")
    buffer = _build_combined_buffer(spec, p["train_pairs"], p["test_pairs"])
    return MultiTaskDataset(
        spec=spec, train_pairs=p["train_pairs"], test_pairs=p["test_pairs"],
        role_cache=p["role_cache"], buffer=buffer,
        per_task_stats=p["per_task_stats"],
        train_by_task=p.get("train_by_task", {}), test_by_task=p.get("test_by_task", {}),
    )


def _main(argv=None) -> None:
    import argparse

    from moo_mcts.logging_util import configure as configure_logging

    p = argparse.ArgumentParser(
        description="Group the already-saved per-task checkpoints into one pooled GNN dataset.")
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                   help=f"comma-separated subset of task keys (default: all {len(DEFAULT_TASKS)} -- "
                        f"{','.join(DEFAULT_TASKS)}); e.g. --tasks aime,math")
    p.add_argument("--model-slug", default=DEFAULT_MODEL_SLUG)
    p.add_argument("--exp", default=DEFAULT_EXP)
    p.add_argument("--checkpoint", action="append", default=[], metavar="TASK=PATH",
                   help="override the checkpoint file for TASK (repeatable); the save path is "
                        "unaffected. e.g. --checkpoint aime=results/.../variant_checkpoint.pkl")
    p.add_argument("--save", default=None,
                   help="write the pooled dataset here (a .pkl phase 2 can load with --dataset)")
    p.add_argument("--collect-missing", action="store_true",
                   help="regenerate a missing checkpoint via build_task_tree (needs the backend)")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    configure_logging(args.log_level)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    overrides: dict[str, str] = {}
    for item in args.checkpoint:
        if "=" not in item:
            p.error(f"--checkpoint expects TASK=PATH, got {item!r}")
        k, v = (s.strip() for s in item.split("=", 1))
        if not k or not v:
            p.error(f"--checkpoint expects TASK=PATH, got {item!r}")
        if k not in tasks:
            log.warning("--checkpoint %s: not in --tasks (%s); ignoring", k, ",".join(tasks))
        overrides[k] = v

    ds = collect_multitask(tasks, model_slug=args.model_slug, exp=args.exp,
                           checkpoint_overrides=overrides,
                           collect_missing=args.collect_missing)
    print(ds.summary())
    n_roles = len((ds.role_cache or {}).get("vectors", {}))
    dim_note = f" (dim {ds.role_cache['dim']})" if ds.role_cache else ""
    print(f"\nrole-embedding cache: {n_roles} unique roles{dim_note}")
    if args.save:
        save_dataset(ds, args.save)
        print(f"[collect] pooled dataset -> {args.save}")


if __name__ == "__main__":
    _main()
