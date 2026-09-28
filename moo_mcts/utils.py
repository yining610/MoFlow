"""Shared helpers for the moo_mcts CLI.
"""
import csv
import json
import os

import numpy as np

from .objectives import CANONICAL_ORDER, default_spec
from .search.debug import dump_tree
from .serve.policy import policy_dict_signature, policy_signature
from .serve.save_policy import _slug
from .workflow.serialize_policy import to_python_policy, to_yaml_policy
from tasks.registry import DATASETS

_DEFAULT_TESTING_WEIGHTS: tuple[tuple[float, ...], ...] = (
    (0.6, 0.1, 0.1, 0.1, 0.1),
    (0.1, 0.6, 0.1, 0.1, 0.1),
    (0.1, 0.1, 0.6, 0.1, 0.1),
    (0.1, 0.1, 0.1, 0.6, 0.1),
    (0.1, 0.1, 0.1, 0.1, 0.6),
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0),
    (0.2, 0.2, 0.2, 0.2, 0.2),
)


def _parse_w(s: str, D: int) -> np.ndarray:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != D:
        raise SystemExit(f"--w must have {D} comma-separated values, got {len(parts)}")
    w = np.array(parts, dtype=float)
    if w.sum() <= 0:
        raise SystemExit("--w must sum to a positive value")
    return w / w.sum()


def _ref_kwargs(args) -> dict:
    """Cost/latency hypervolume reference overrides, precedence CLI > TaskProfile > catalogue.
    """
    kw = {}
    task = getattr(args, "task", None)
    if task and task in DATASETS:
        profile = DATASETS[task].task_profile
        if profile is not None:
            if getattr(profile, "cost_ref_tokens", None) is not None:
                kw["cost_ref_tokens"] = profile.cost_ref_tokens
            if getattr(profile, "latency_ref_calls", None) is not None:
                kw["latency_ref_calls"] = profile.latency_ref_calls
    if getattr(args, "cost_ref_tokens", None) is not None:
        kw["cost_ref_tokens"] = args.cost_ref_tokens
    if getattr(args, "latency_ref_calls", None) is not None:
        kw["latency_ref_calls"] = args.latency_ref_calls
    return kw


def _ref_source(args, cli_attr: str, profile_attr: str) -> str:
    """Where the effective reference for one axis came from (CLI > task profile > catalogue)."""
    if getattr(args, cli_attr, None) is not None:
        return f"CLI --{cli_attr.replace('_', '-')}"
    task = getattr(args, "task", None)
    if task and task in DATASETS:
        prof = DATASETS[task].task_profile
        if prof is not None and getattr(prof, profile_attr, None) is not None:
            return "task profile default"
    return "catalogue default"


def _log_refs(args, spec) -> None:
    """Print the effective cost/latency hypervolume reference for this run and its source.

    Makes the reference explicit on every CLI run -- a wrong (too-small) reference silently
    pins hypervolume at 0 and clips the CZT reward, so surfacing it turns a silent failure
    into a visible number.
    """
    parts = []
    for o in spec.objectives:
        if o.name == "cost":
            parts.append(f"cost_ref={o.span:g} tok ({_ref_source(args, 'cost_ref_tokens', 'cost_ref_tokens')})")
        elif o.name == "latency":
            parts.append(f"latency_ref={o.span:g} calls ({_ref_source(args, 'latency_ref_calls', 'latency_ref_calls')})")
    if parts:
        print("[refs] " + ", ".join(parts))


def _spec_from_args(args):
    objectives = None
    if getattr(args, "objectives", None):
        objectives = [x.strip() for x in args.objectives.split(",") if x.strip()]
    try:
        spec = default_spec(objectives=objectives, **_ref_kwargs(args))
    except ValueError as e:
        raise SystemExit(f"--objectives: {e}")
    _log_refs(args, spec)
    return spec


def save_result(engine, results_dir: str, stem: str, spec) -> tuple[str, str]:
    """Save tree and hypervolume trajectory."""
    tree_path = os.path.join(results_dir, f"{stem}_tree.json")
    dump_tree(engine.root, tree_path, spec)
    print(f"[save] search tree -> {tree_path}")
    metrics_path = os.path.join(results_dir, f"{stem}_metrics.csv")
    engine.save_metrics(metrics_path)
    print(f"[save] HV trajectory ({len(engine.hv_log)} checkpoints) -> {metrics_path}")
    weights_path = os.path.join(results_dir, f"{stem}_weights.csv")
    engine.save_weights(weights_path)
    print(f"[save] sampled preference weights ({len(engine.w_log)} trials) -> {weights_path}")
    return tree_path, metrics_path


def _w_label(w) -> str:
    """Format a weight vector as an underscore-joined numeric string, e.g. '0.8_0.1_0.1'."""
    return "_".join(f"{v:.4g}" for v in w)


def _heldout_policy_path(results_dir: str, name: str) -> str:
    """Path dump_policy would write for preference `name` (mirrors save_policy.dump_policy)."""
    return os.path.join(results_dir, f"policy_{_slug(name)}.json")


def _reusable_heldout(policy_path: str, served_sig: str | None, n_expected: int) -> dict | None:
    """Return the saved held-out raw metrics from `policy_path`, but only if they cover
    `n_expected` problems and their policy matches `served_sig` (else None, so the caller re-evaluates).
    """
    try:
        with open(policy_path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    ho = d.get("heldout")
    if not (isinstance(ho, dict) and int(ho.get("n_problems", 0)) == int(n_expected)):
        return None
    raw = ho.get("raw")
    if not isinstance(raw, dict):
        return None
    if served_sig is None or served_sig != policy_dict_signature(d.get("policy")):
        return None
    return raw


def _parse_shard(s: str | None) -> tuple[int, int]:
    if not s:
        return 1, 0
    try:
        i_str, n_str = s.split("/")
        i, n = int(i_str), int(n_str)
    except (ValueError, AttributeError):
        raise SystemExit(f"--query-shard must be 'i/N' (e.g. 3/10), got {s!r}")
    if not (n >= 1 and 1 <= i <= n):
        raise SystemExit(f"--query-shard i/N requires 1<=i<=N and N>=1, got {s!r}")
    return i, n


def _demo_preferences(spec) -> dict[str, list[float]]:
    names = spec.names
    n = len(names)
    if names == CANONICAL_ORDER:
        return {_w_label(w): list(w) for w in _DEFAULT_TESTING_WEIGHTS}
    demos: dict[str, list[float]] = {}
    for focus in names:
        w = [0.0] * n
        rest = 0.2 / max(1, n - 1)
        for i, nm in enumerate(names):
            w[i] = 0.8 if nm == focus else rest
        demos[_w_label(w)] = w
    bal = [1.0 / n] * n
    demos[_w_label(bal)] = bal
    return demos


def _show_entry(spec, policy, emit: str | None) -> None:
    if policy is None:
        print("  (no workflow found)")
        return
    if policy.value is not None:
        raw = spec.display(policy.value)
        print("  value:", {k: round(v, 4) for k, v in raw.items()})
    print("  policy:")
    print(policy.render(spec))
    if emit == "python":
        print("\n  --- Python ---")
        print(to_python_policy(policy))
    elif emit == "yaml":
        print("\n  --- YAML ---")
        print(to_yaml_policy(policy))


def _points_row_count(path: str) -> int:
    """Number of served-preference rows already saved in a points.csv (0 if absent).
    """
    if not path or not os.path.exists(path):
        return 0
    with open(path, newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _upsert_row(path: str, row: dict, fields: list[str], key: str) -> None:
    """Add a CSV row, or replace the existing one with the same `key`, so re-running a
    query updates its summary row instead of adding a duplicate."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows: dict = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = {r[key]: r for r in csv.DictReader(fh)}
    rows[str(row[key])] = {k: row[k] for k in fields}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows.values())


def save_points(points_path, spec, name, w, held, n_heldout, policy_file) -> int:

    w = np.asarray(w, dtype=float)
    raw = held.raw if held is not None else {}
    row = {"preference_label": str(name)}
    for i, ax in enumerate(spec.names):
        row[f"w_{ax}"] = round(float(w[i]), 6)
        row[ax] = round(float(raw[ax]), 6) if ax in raw else ""
    row["n_heldout"] = int(n_heldout)
    row["policy_file"] = policy_file

    fields = (["preference_label"] + [f"w_{a}" for a in spec.names] + list(spec.names) + ["n_heldout", "policy_file"])
    rows = {}
    if points_path and os.path.exists(points_path):
        with open(points_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if "preference_label" in (reader.fieldnames or []):
                rows = {r["preference_label"]: r for r in reader}
            else:
                print(f"[save] WARNING: {points_path} header {reader.fieldnames} "
                      f"lacks 'preference_label'; rebuilding the file")
    rows[name] = row
    os.makedirs(os.path.dirname(points_path) or ".", exist_ok=True)
    with open(points_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows.values())
    return len(rows)


def serve_heldout_dedup(grid, spec, serve, evaluate, save, *, emit=None, cache=None):
    """Held-out re-scoring over a preference grid with a UNIQUE-WORKFLOW guarantee.

    - grid: list of (name, w) preferences to serve + score.
    - serve(w) -> policy: cheap deterministic retrieval, used only for the dedup key.
    - evaluate(w) -> (policy, held): the EXPENSIVE held-out eval (called per unique).
    - save(name, w, policy, held, reused): persist this weight's policy + points row.
    - cache: optional pre-seeded {signature: (policy, held)} (e.g. from a resumed run)
             so weights matching an already-computed workflow skip re-execution too.

    Returns the list of (name, w, policy, held, reused) in grid order.
    """
    cache = cache if cache is not None else {}
    results = []
    n_unique = 0
    for j, (name, w) in enumerate(grid, 1):
        sig = policy_signature(serve(w))
        reused = sig is not None and sig in cache
        if reused:
            policy, held = cache[sig]
        else:
            policy, held = evaluate(w)
            if sig is not None:
                cache[sig] = (policy, held)
            n_unique += 1
        tag = "  (reused served workflow -- no re-execution)" if reused else ""
        print(f"\n[serve {j}/{len(grid)} w={name}]{tag}")
        _show_entry(spec, policy, emit)
        if held is not None:
            shown = {k: round(v, 4) for k, v in held.raw.items() if k in spec.names}
            print(f"  held-out: {shown}")
        save(name, w, policy, held, reused)
        results.append((name, w, policy, held, reused))
    print(f"[dedup] {len(grid)} preference weights -> {n_unique} unique workflows "
          f"({len(grid) - n_unique} reused, no re-execution)")
    return results
