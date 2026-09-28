import argparse
import csv
import glob
import json
import os
from types import SimpleNamespace

import wandb

import numpy as np
from tqdm.auto import tqdm
from dataclasses import asdict

from .config import BackendConfig, RunConfig, SearchConfig
from .drivers import per_query, per_task
from .drivers.common import build_backends
from .logging_util import configure as configure_logging
from .objectives import available_objectives, default_spec
from .reward.evaluator import Evaluator
from .search.checkpoint import Checkpointer
from .search.debug import load_tree, render_tree
from .serve.policy import extract_policy, policy_signature
from .serve.save_policy import default_bundle_path, dump_policy, result_dir
from .backends.openai_compatible import make_client
from tasks.hf_dataset import (
    _split_dir,
    dataset_exists,
    download_hf_dataset,
    load_hf_split,
    load_split_queries,
    make_default_split,
)
from tasks.paraphrase import Paraphraser, augment_dataset_with_paraphrases
from tasks.registry import DATASETS
from .utils import (
    _demo_preferences,
    _heldout_policy_path,
    _log_refs,
    _parse_shard,
    _parse_w,
    _points_row_count,
    _ref_kwargs,
    _reusable_heldout,
    _show_entry,
    _spec_from_args,
    _upsert_row,
    _w_label,
    save_points,
    save_result,
    serve_heldout_dedup,
)


def cmd_build_task(args) -> None:
    configure_logging(args.log_level)

    spec = _spec_from_args(args)
    cfg = RunConfig()
    cfg.search = SearchConfig(n_trials=args.trials, max_depth=args.max_depth, seed=args.seed)
    cfg.search.chance_samples = args.samples
    cfg.max_workers = args.max_workers
    cfg.backend.max_concurrency = (
        args.max_concurrency if args.max_concurrency is not None else args.max_workers
    )
    _apply_early_stop(cfg, args)
    cfg.backend.realizations = args.realizations
    cfg.predictor.kind = args.predictor
    cfg.predictor.device = _DEVICE_TO_TORCH[args.device]
    cfg.search.selector = args.selector
    cfg.reward.n_paraphrases = args.paraphrases
    cfg.backend.api_key_env = args.api_key_env
    cfg.backend.base_url = args.base_url
    cfg.backend.timeout_s = args.timeout
    if args.model:
        cfg.backend.model = args.model

    offline = args.offline_gnn is not None
    if offline:
        cfg.predictor.kind = "gnn"  # offline valuation is driven by the GNN predictor

    heldout: list = []
    dataset_spec = DATASETS[args.task]
    task, heldout = load_hf_split(dataset_spec)
    checker = dataset_spec.checker
    public_tester = dataset_spec.public_tester
    task_profile = dataset_spec.task_profile

    results_dir = None
    metrics_path = None
    weights_path = None
    stem = f"{dataset_spec.name}_task"
    checkpoint_path = args.checkpoint
    if args.save:
        results_dir = args.out_dir or result_dir(dataset_spec.name, "per_task", cfg.backend.model)
        os.makedirs(results_dir, exist_ok=True)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(results_dir, f"{stem}_checkpoint.pkl")
        metrics_path = os.path.join(results_dir, f"{stem}_metrics.csv")
        weights_path = os.path.join(results_dir, f"{stem}_weights.csv")

    if offline:
        bundle_path = args.offline_gnn  # loaded read-only; drives node valuation, never overwritten
    elif args.no_bundle:
        bundle_path = None
    else:
        bundle_path = args.bundle or default_bundle_path(cfg.backend.model, spec)

    print(f"[build-task] task={task.name} trials={args.trials} m={cfg.reward.n_paraphrases} backend={cfg.backend.kind} model={cfg.backend.model}")
    if checkpoint_path:
        print(f"[build-task] checkpoint -> {checkpoint_path} (resume={not args.no_resume})")
    if offline:
        print(f"[build-task] OFFLINE GNN valuation: every node scored by {bundle_path} "
              f"(no workflow execution; proposals still use the backend)")
    elif bundle_path:
        print(f"[build-task] gnn bundle (shared across tasks) -> {bundle_path}")
    
    _init_wandb(cfg, args, run_name=f"{dataset_spec.name}_per_task")

    result = per_task.build_task_tree(
        task, cfg, spec, 
        checker=checker, 
        task_profile=task_profile,
        checkpoint_path=checkpoint_path, 
        resume=not args.no_resume,
        metrics_path=metrics_path,
        weights_path=weights_path,
        bundle_path=bundle_path,
        public_tester=public_tester,
        offline=offline,
    )

    if cfg.wandb.enabled and wandb.run is not None:
        wandb.finish()
    print(f"[build-task] archive size = {len(result.archive)} | hypervolume = {result.archive.normalized_hypervolume():.4g}")

    if args.save:
        save_result(result.engine, results_dir, stem, spec)
        print("[save] full search tree:")
        print(render_tree(result.engine.root, spec))

    # retrieve + held-out-score the demo preferences, with a UNIQUE-WORKFLOW deduplication guarantee
    demos = _demo_preferences(spec)
    saved: list[str] = []
    points_path = os.path.join(results_dir, "points.csv") if args.save else None
    cache: dict = {}          # signature -> (policy, held); seeds resumed prefs + twins
    pending: list = []
    for name, raw_w in demos.items():
        w = _parse_w(",".join(map(str, raw_w)), spec.D)
        if args.save and not args.no_resume and heldout:
            # Reuse an on-disk result only if its saved policy still matches the workflow
            path = _heldout_policy_path(results_dir, name)
            resumed_policy = per_task.serve(result, w)
            sig = policy_signature(resumed_policy)
            raw = _reusable_heldout(path, sig, len(heldout))
            if raw is not None:
                print(f"\n[serve w={name}] resume: held-out already computed -> skipping (points.csv backfilled)")
                saved.append(path)
                held = SimpleNamespace(raw=raw)
                # The policy json is saved before its points.csv row, so a crash between the two skips this preference on resume.
                if points_path is not None:
                    save_points(points_path, spec, name, w, held, len(heldout), path)
                # Seed the cache so a not-yet-done twin of this workflow reuses it.
                cache.setdefault(sig, (resumed_policy, held))
                continue
        pending.append((name, w))

    def _serve(w):
        return per_task.serve(result, w)

    def _eval(w):
        return per_task.evaluate_heldout(
            result, w, heldout, n_paraphrases=cfg.reward.n_paraphrases,
            samples=args.samples,
        )

    def _save(name, w, policy, held, reused):
        if not args.save:
            return
        path = dump_policy(policy, spec, w, name, heldout_res=held,
                           n_heldout=len(heldout), out_dir=results_dir)
        saved.append(path)
        save_points(points_path, spec, name, w, held, len(heldout), path)
        print(f"  [save] saved policy -> {path}")

    serve_heldout_dedup(pending, spec, _serve, _eval, _save, emit=args.emit, cache=cache)

def cmd_build_query(args) -> None:
    """Per-query mode: build one tree per query, reusing the cross-query buffer.
    """
    configure_logging(args.log_level)

    spec = _spec_from_args(args)
    cfg = RunConfig()
    cfg.search = SearchConfig(n_trials=args.trials, max_depth=args.max_depth, seed=args.seed)
    cfg.search.chance_samples = args.samples
    cfg.search.uncertainty_gate = args.uncertainty_gate
    cfg.search.beam_k = args.beam_k
    cfg.max_workers = args.max_workers
    cfg.backend.max_concurrency = (
        args.max_concurrency if args.max_concurrency is not None else args.max_workers
    )
    _apply_early_stop(cfg, args)
    cfg.backend.realizations = args.realizations
    cfg.predictor.kind = args.predictor
    cfg.predictor.device = _DEVICE_TO_TORCH[args.device]
    cfg.search.selector = args.selector
    cfg.reward.n_paraphrases = args.paraphrases
    cfg.backend.api_key_env = args.api_key_env
    cfg.backend.base_url = args.base_url
    cfg.backend.timeout_s = args.timeout
    if args.model:
        cfg.backend.model = args.model

    dataset_spec = DATASETS[args.task]
    queries = load_split_queries(
        dataset_spec, subset=args.split, n_queries=args.n_queries,
    )
    checker = dataset_spec.checker
    task_profile = dataset_spec.task_profile
    target_w = _parse_w(args.target_w, spec.D) if args.target_w else None

    if args.no_bundle:
        bundle_path = None
    else:
        bundle_path = args.bundle or default_bundle_path(cfg.backend.model, spec)

    session = per_query.QuerySession(cfg, spec, checker=checker, task_profile=task_profile,
                                     public_tester=dataset_spec.public_tester,
                                     bundle_path=bundle_path)
    print(f"[build-query] task={args.task} split={args.split} queries={len(queries)} trials={args.trials} "
          f"m={cfg.reward.n_paraphrases} backend={cfg.backend.kind} model={cfg.backend.model}")
    if bundle_path:
        print(f"[build-query] offline GNN (frozen, read-only valuation) -> {bundle_path}")

    results_dir = (args.out_dir or result_dir(dataset_spec.name, "per_query", cfg.backend.model)) if args.save else None
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)

    session_ckpt = None
    ckpt_path = args.checkpoint
    if ckpt_path is None and args.save:
        ckpt_path = os.path.join(results_dir, f"{dataset_spec.name}_query_session.pkl")
    if ckpt_path:
        session_ckpt = Checkpointer(ckpt_path)
        if not args.no_resume and session_ckpt.exists():
            session.restore(session_ckpt.load())
            print(f"[build-query] resumed session from {ckpt_path}: "
                  f"{session.n_solved} queries done, buffer={len(session.buffer)}")
        print(f"[build-query] session checkpoint -> {ckpt_path} (resume={not args.no_resume})")

    shard_i, shard_n = _parse_shard(args.query_shard)

    _init_wandb(cfg, args, run_name=f"{dataset_spec.name}_per_query")
    demos = _demo_preferences(spec)
    if target_w is not None:
        grid = [(f"target_{_w_label(target_w)}", np.asarray(target_w, dtype=float))]
    else:
        grid = [(name, _parse_w(",".join(map(str, raw_w)), spec.D)) for name, raw_w in demos.items()]

    shard_queries = [(qi, q) for qi, q in enumerate(queries)
                     if not shard_n or (qi % shard_n) == (shard_i - 1)]
    if shard_n:
        print(f"[build-query] query shard {shard_i}/{shard_n}: {len(shard_queries)} of "
              f"{len(queries)} queries (global idx {[qi for qi, _ in shard_queries]})")

    summary_path = os.path.join(results_dir, f"{dataset_spec.name}_query_summary.csv") if args.save else None

    # A query is done iff its held-out points.csv already holds the full preference grid.
    # So a relaunch never re-runs the held-out for a query already saved
    def _done(qi: int) -> bool:
        if not results_dir:
            return False
        p = os.path.join(results_dir, f"{dataset_spec.name}_query_q{qi}_points.csv")
        return _points_row_count(p) >= len(grid)

    n_done = sum(1 for qi, _ in shard_queries if _done(qi))
    if n_done:
        print(f"[build-query] resume: {n_done}/{len(shard_queries)} queries already have complete "
              f"held-out results on disk -- skipping them")

    qbar = tqdm(shard_queries, desc="queries", unit="query")
    for spos, (qi, query) in enumerate(qbar):
        if _done(qi):
            continue  # held-out already saved for this query (durable, disk-driven skip)
        preview = query.text.replace("\n", " ")[:70]
        qbar.set_postfix_str(preview)
        print(f"\n=== query {qi + 1}/{len(queries)}: {preview!r} "
              f"({len(query.paraphrases)} paraphrases) ===")
        result = session.solve(query, target_w=target_w, desc=f"q{qi + 1}/{len(queries)}")
        insample_hv = result.archive.normalized_hypervolume()
        print(f"[build-query] archive size = {len(result.archive)} | "
              f"in-sample archive HV = {insample_hv:.4g}")

        stem = f"{dataset_spec.name}_query_q{qi}"
        if args.save:
            save_result(result.engine, results_dir, stem, spec)
        points_path = os.path.join(results_dir, f"{stem}_points.csv") if args.save else None

        # serve each grid preference + re-score HELD-OUT, deduping identical workflows
        # (many weights extract the same served workflow -> execute held-out only once).
        print(f"[build-query] q{qi}: serving + held-out-scoring {len(grid)} preference weights ...")
        ho_vectors: list = []

        def _serve(w):
            return per_query.serve(result, w)

        def _eval(w):
            return per_query.evaluate_heldout(
                result, query, w, search_paraphrases=cfg.reward.n_paraphrases,
                n_paraphrases=cfg.reward.n_paraphrases, samples=cfg.search.chance_samples,
            )

        def _save(name, w, policy, held, reused):
            if held is not None:
                ho_vectors.append(held.vector)
            if args.save:
                path = dump_policy(policy, spec, w, f"q{qi}_{name}",
                                   heldout_res=held, n_heldout=1, out_dir=results_dir)
                if points_path is not None:
                    save_points(points_path, spec, name, w, held, 1, path)

        serve_heldout_dedup(grid, spec, _serve, _eval, _save, emit=args.emit)

        ho_hv = per_query.heldout_hypervolume(ho_vectors, spec)
        print(f"[build-query] q{qi}: held-out {len(ho_vectors)}-point HV = {ho_hv:.4g} | "
              f"in-sample archive HV = {insample_hv:.4g}")
        if summary_path is not None:
            _upsert_row(summary_path,
                        {"query": qi, "n_points": len(ho_vectors), "heldout_hv": ho_hv,
                         "insample_archive_hv": insample_hv, "preview": preview},
                        ["query", "n_points", "heldout_hv", "insample_archive_hv", "preview"],
                        key="query")

        # record session progress (n_solved + predictor) once the query is fully solved and saved
        if session_ckpt is not None:
            session_ckpt.save(session.snapshot())

        pt_files = sorted(glob.glob(
            os.path.join(results_dir, f"{dataset_spec.name}_query_q*_points.csv")))
        if pt_files:
            front, pooled_hv = per_query.pooled_front_hv(pt_files, spec)
            n_q = max((f["n_queries"] for f in front), default=0)
            names = list(spec.names)
            pooled_path = os.path.join(
                results_dir, f"{dataset_spec.name}_query_pooled_points.csv")
            with open(pooled_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["preference_label"] + [f"w_{n}" for n in names] + names + ["n_queries"])
                for f in front:
                    w.writerow([f["preference_label"]]
                               + [f"{x:g}" for x in f["weights"]]
                               + [f"{f['objectives'][n]:g}" for n in names]
                               + [f["n_queries"]])
            print(f"\n[build-query] HV-of-means over {n_q} queries (averaged front) = {pooled_hv:.4g}")
            print(f"[build-query] averaged front -> {pooled_path}")

    # secondary diagnostic: per-query spread (mean of the individual per-query HVs).
    if summary_path is not None and os.path.exists(summary_path):
        with open(summary_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            mean_ho = float(np.mean([float(r["heldout_hv"]) for r in rows]))
            mean_is = float(np.mean([float(r["insample_archive_hv"]) for r in rows]))
            print(f"[build-query] (diagnostic) per-query mean HV = {mean_ho:.4g} | "
                  f"mean in-sample archive HV = {mean_is:.4g}")
            print(f"[build-query] per-query summary -> {summary_path}")

    if cfg.wandb.enabled and wandb.run is not None:
        wandb.finish()


def cmd_inference(args) -> None:

    configure_logging(args.log_level)

    with open(args.tree, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    ref_kw = _ref_kwargs(args)
    if getattr(args, "objectives", None):
        spec = _spec_from_args(args)  # logs refs itself
    elif data.get("objectives"):
        spec = default_spec(objectives=list(data["objectives"]), **ref_kw)
        _log_refs(args, spec)
    else:
        spec = default_spec(**ref_kw)
        _log_refs(args, spec)

    try:
        root = load_tree(data, spec)
    except ValueError as e:
        raise SystemExit(
            f"[inference] tree axes don't match spec {list(spec.names)}: {e}"
        )

    # --w accepts one OR many weight vectors; a single value stays behavior-identical.
    w_args = args.w if isinstance(args.w, list) else [args.w]
    grid = [(_w_label(w), w) for w in (_parse_w(x, spec.D) for x in w_args)]
    labels = ", ".join(name for name, _ in grid)
    print(f"[inference] tree={args.tree} objectives={list(spec.names)} "
          f"{len(grid)} weight(s): {labels}")

    if not args.task:
        for name, w in grid:
            print(f"\n[serve w={name}]")
            _show_entry(spec, extract_policy(root, w, spec), args.emit)
        return

    cfg = RunConfig()
    cfg.max_workers = args.max_workers
    cfg.backend.max_concurrency = (
        args.max_concurrency if args.max_concurrency is not None else args.max_workers
    )
    if args.model:
        cfg.backend.model = args.model
    cfg.reward.n_paraphrases = args.paraphrases
    cfg.backend.api_key_env = args.api_key_env
    cfg.backend.base_url = args.base_url
    cfg.backend.timeout_s = args.timeout

    dataset_spec = DATASETS[args.task]
    task, heldout = load_hf_split(dataset_spec)
    if not heldout:
        raise SystemExit(
            f"[inference] no held-out problems in the frozen default split for task "
            f"{args.task!r}; create it first with `python -m moo_mcts.cli data --task {args.task}`"
        )
    checker = dataset_spec.checker
    task_profile = dataset_spec.task_profile
    proposer, make_executor, critic = build_backends(
        cfg.backend, spec, task_profile=task_profile
    )
    evaluator = Evaluator(spec, make_executor=make_executor, checker=checker,
                          make_tester=dataset_spec.public_tester,
                          max_workers=cfg.max_workers)

    engine = type("_LoadedEngine", (), {"root": root})()
    result = per_task.TaskResult(
        archive=None, spec=spec, engine=engine, evaluator=evaluator, checker=checker,
    )

    out_dir = args.out_dir or result_dir(dataset_spec.name, "per_task", cfg.backend.model)
    os.makedirs(out_dir, exist_ok=True)
    points_path = os.path.join(out_dir, "points.csv")

    print(f"[inference] held-out re-scoring on {len(heldout)} problems "
          f"(m={cfg.reward.n_paraphrases}, R={args.samples}, model={cfg.backend.model})")

    def _serve(w):
        return per_task.serve(result, w)

    def _eval(w):
        return per_task.evaluate_heldout(
            result, w, heldout, n_paraphrases=cfg.reward.n_paraphrases,
            samples=args.samples,
        )

    def _save(name, w, policy, held, reused):
        path = dump_policy(policy, spec, w, name, heldout_res=held,
                           n_heldout=len(heldout), out_dir=out_dir)
        n = save_points(points_path, spec, name, w, held, len(heldout), path)
        print(f"  [save] policy -> {path}")
        print(f"  [save] points -> {points_path} ({n} rows)")

    cache: dict = {}
    stale = 0
    for name, w in grid:
        served = per_task.serve(result, w)
        served_sig = policy_signature(served)
        raw = _reusable_heldout(_heldout_policy_path(out_dir, name), served_sig, len(heldout))
        if raw is not None:
            cache.setdefault(served_sig, (served, SimpleNamespace(raw=raw)))
        elif os.path.exists(_heldout_policy_path(out_dir, name)):
            stale += 1
    if cache or stale:
        print(f"[inference] resume: reusing {len(cache)} completed workflow(s) from "
              f"{out_dir}" + (f"; {stale} stale/mismatched row(s) will be re-scored" if stale else ""))

    serve_heldout_dedup(grid, spec, _serve, _eval, _save, emit=args.emit, cache=cache)


def cmd_data_process(args) -> None:
    configure_logging(args.log_level)
    spec = DATASETS[args.task]

    raw_path = args.data_path or spec.path
    para_path = args.paraphrase_path or spec.paraphrase_path or (raw_path.rstrip("/") + "_paraphrase")
    split_out = args.out_dir or spec.split_dir or _split_dir(para_path)

    # step 1: raw dataset -> disk
    if args.force or not dataset_exists(raw_path):
        print(f"[data] 1/3 downloading {spec.hf_id!r} -> {raw_path}")
        download_hf_dataset(spec, out_path=raw_path)
    else:
        print(f"[data] 1/3 raw dataset present -> {raw_path} (skip; --force to redo)")

    # step 2: paraphrases
    if args.force or not dataset_exists(para_path):
        cfg = RunConfig()
        if args.model:
            cfg.backend.model = args.model
        cfg.backend.api_key_env = args.api_key_env
        cfg.backend.base_url = args.base_url
        cfg.backend.timeout_s = args.timeout
        client = make_client(cfg.backend)
        paraphraser = Paraphraser(client, n=args.n, temperature=args.temperature)
        print(f"[data] 2/3 paraphrasing n={args.n}/problem (model={cfg.backend.model}) -> {para_path}")
        para_path, paras = augment_dataset_with_paraphrases(
            spec, paraphraser, in_path=raw_path, out_path=para_path
        )
        avg = sum(len(p) for p in paras) / max(1, len(paras))
        print(f"[data]     {len(paras)} problems | avg {avg:.1f} paraphrases/problem")
    else:
        print(f"[data] 2/3 paraphrase dataset present -> {para_path} (skip; --force to redo)")

    # step 3: frozen val/test split (carries paraphrases)
    manifest = make_default_split(
        spec, 
        path=para_path, 
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        out_dir=split_out, 
    )
    para = manifest.get("paraphrase_col")
    print(f"[data] 3/3 split seed={manifest['seed']} total={manifest['total']} "
          f"validate={manifest['val_size']} test={manifest['test_size']} "
          f"unused={manifest.get('unused', 0)} "
          f"paraphrases={'yes' if para else 'no'} -> {split_out}")

def _add_early_stop_args(parser) -> None:
    """Add the shared hypervolume-plateau early-stop flags to a build subparser."""
    parser.add_argument("--no-early-stop", action="store_true",
                        help="disable early stopping (run the full --trials budget)")
    parser.add_argument("--es-tol", type=float, default=0.01,
                        help="early-stop tolerance: min relative HV gain over the trailing window to keep going (higher = stops sooner)")
    parser.add_argument("--es-patience", type=int, default=20,
                        help="trailing window, in trials, over which the HV gain is measured")
    parser.add_argument("--es-min-trials", type=int, default=20,
                        help="never stop before this many trials have run")


def _apply_early_stop(cfg, args) -> None:
    cfg.search.early_stop = not args.no_early_stop
    cfg.search.es_tol = args.es_tol
    cfg.search.es_patience = args.es_patience
    cfg.search.es_min_trials = args.es_min_trials


def _add_objective_ref_args(parser) -> None:
    """Add the shared hypervolume reference-point overrides for the cost/latency axes.

    These set where each axis normalizes to 0 (and its span). They must sit above the run's
    worst-case cost/latency, else normalized cost/latency go negative, the archive never
    dominates the origin (hypervolume pins at 0), and the CZT reward clips to 0 -- blinding
    action selection. Model-dependent: a reasoning model needs a far larger cost reference
    than a cheap one on the same task. Unset -> TaskProfile default -> generic catalogue default.
    """
    parser.add_argument("--cost-ref-tokens", type=float, default=None,
                        help="hypervolume reference / normalization span for the cost axis, in tokens "
                             "(set above the worst-case per-workflow token cost; overrides the task default)")
    parser.add_argument("--latency-ref-calls", type=float, default=None,
                        help="hypervolume reference / normalization span for the latency axis, in LLM calls "
                             "(set above the worst-case per-workflow call count; overrides the task default)")


def _add_backend_args(parser) -> None:
    """Add shared real-backend flags to a subparser."""
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="name of the environment variable holding the backend API key (default: OPENAI_API_KEY)")
    parser.add_argument("--base-url", default=BackendConfig().base_url,
                        help="base URL of the OpenAI-compatible proxy (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=BackendConfig().timeout_s,
                        help="per-call backend timeout in seconds (default: %(default)s); "
                             "raise this for slow reasoning models whose long traces exceed 60s")


def _add_concurrency_arg(parser) -> None:
    """Add the shared concurrency flags: --max-workers (thread pool) and --max-concurrency (LLM ceiling)."""
    parser.add_argument("--max-workers", type=int, default=32,
                        help="workflow thread-pool size for the eval fan-out, chance-node draws, and completions (how many workflows run concurrently).")
    parser.add_argument("--max-concurrency", type=int, default=None,
                        help="max concurrent in-flight LLM calls (the global client semaphore). Defaults to --max-workers; set it lower than --max-workers to throttle LLM calls below the workflow thread count (e.g. proxy rate limits).")


_DEVICE_TO_TORCH = {"auto": None, "cpu": "cpu", "gpu": "cuda"}
def _add_device_arg(parser) -> None:
    """Add the shared --device flag for the GNN predictor + role encoder."""
    parser.add_argument("--device", default="auto", choices=list(_DEVICE_TO_TORCH),
                        help="device for the GNN predictor + role encoder: 'cpu', 'gpu' (=cuda), or 'auto' (cuda if available, else cpu).")


def _add_wandb_args(parser) -> None:
    """Add the shared Weights & Biases logging flags to a build subparser."""
    parser.add_argument("--wandb", action="store_true",
                        help="log GNN train/test loss + accuracy to Weights & Biases (needs --predictor gnn)")
    parser.add_argument("--wandb-project", default="moo-mcts-gnn", help="wandb project name")
    parser.add_argument("--wandb-run-name", default=None, help="wandb run name (defaults to an auto name)")


def _init_wandb(cfg, args, run_name: str) -> None:
    """Populate cfg.wandb from CLI flags and start the wandb run (no-op unless --wandb)."""

    cfg.wandb.enabled = args.wandb
    cfg.wandb.project = args.wandb_project
    cfg.wandb.run_name = args.wandb_run_name or run_name
    if not args.wandb:
        return
    if cfg.predictor.kind != "gnn":
        print("[wandb] warning: --predictor is not 'gnn'; there are no GNN metrics to log")

    wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, config=asdict(cfg))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="moo_mcts")
    sub = p.add_subparsers(dest="cmd", required=True)

    obj_help = (
        "comma-separated objective subset (>=2) from "
        f"{','.join(available_objectives())}; default all five"
    )

    bt = sub.add_parser("build-task", help="build a per-task tree and demo retrieval")
    bt.add_argument("--objectives", default=None, help=obj_help)
    bt.add_argument("--task", default=next(iter(DATASETS)), choices=[*DATASETS], help="a registered real dataset")
    bt.add_argument("--trials", type=int, default=200)
    bt.add_argument("--max-depth", type=int, default=6)
    bt.add_argument("--paraphrases", type=int, default=6,
                    help="number of paraphrase variants per problem used for the robustness signal during search and held-out eval")
    bt.add_argument("--samples", type=int, default=3,
                    help="R: repeated draws whose cross-draw accuracy variance gives consistency.")
    bt.add_argument("--model", default=None, help="model name for the real backend")
    bt.add_argument("--predictor", default="heuristic", choices=["heuristic", "gnn"])
    bt.add_argument("--selector", default="czt",
                    choices=["czt", "pareto", "hypervolume", "chebyshev"], help="per-node action selection policy")
    bt.add_argument("--realizations", type=int, default=3,
                    help="LLM role-realizations per decision (chance-node branching); 1 = deterministic single-successor transitions")
    bt.add_argument("--emit", default=None, choices=["python", "yaml"])
    bt.add_argument("--save", action="store_true",
                    help="save tree/metrics/policies (and a full checkpoint incl. the GNN) under --out-dir")
    bt.add_argument("--out-dir", default=None,
                    help="dir for saved tree/metrics/policies/checkpoint (with --save); default results/<dataset>/<model>/per_task")
    bt.add_argument("--checkpoint", default=None,
                    help="full-search checkpoint (tree+archive+GNN) for same-task resume; auto-set under --out-dir when --save. Overrides --bundle's GNN on resume")
    bt.add_argument("--no-resume", action="store_true",
                    help="ignore the checkpoint and start the search fresh")
    bt.add_argument("--bundle", default=None,
                    help="standalone GNN bundle (model + replay buffer) to warm-start from and overwrite; default shared results/gnn_bundles/<model>/<axes>.pkl")
    bt.add_argument("--no-bundle", action="store_true",
                    help="don't load or save the shared GNN bundle this run")
    bt.add_argument("--offline-gnn", default=None,
                    help="path to a trained GNN bundle (e.g. a multitask bundle from scripts/gnn/train_gnn.py). When set, MOO-MCTS values EVERY node "
                         "(terminal + interior) from the GNN instead of executing workflows DURING THE SEARCH -- no eval-set runs while building the tree")
    bt.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="logging verbosity for the search/backend")
    bt.add_argument("--seed", type=int, default=0)
    _add_early_stop_args(bt)
    _add_wandb_args(bt)
    _add_backend_args(bt)
    _add_device_arg(bt)
    _add_concurrency_arg(bt)
    _add_objective_ref_args(bt)
    bt.set_defaults(func=cmd_build_task)

    # =============================================================================
    bq = sub.add_parser("build-query", help="build a per-query tree per query and demo retrieval")
    bq.add_argument("--objectives", default=None, help=obj_help)
    bq.add_argument("--task", default=next(iter(DATASETS)), choices=[*DATASETS], help="a registered real dataset")
    bq.add_argument("--split", default="val", choices=["val", "test", "all"],
                    help="which frozen-split subset to build per-query trees for")
    bq.add_argument("--n-queries", type=int, default=None,
                    help="cap on how many queries (problems) to build trees for; omit to use the whole selected split subset")
    bq.add_argument("--query-shard", default=None, metavar="i/N",
                    help="process only shard i of N (1<=i<=N), round-robin over the global query index, so N jobs can fan the query set across the cluster. Give each shard its own --out-dir. Omit for all queries")
    bq.add_argument("--paraphrases", type=int, default=5,
                    help="m: paraphrase neighborhood size (incl. the original)")
    bq.add_argument("--samples", type=int, default=3,
                    help="R: execution draws per paraphrase")
    bq.add_argument("--target-w", default=None,
                    help="comma-separated preference to targeted-refine + serve; omit to serve demo preferences from the uniform build")
    bq.add_argument("--trials", type=int, default=200)
    bq.add_argument("--max-depth", type=int, default=6)
    bq.add_argument("--model", default=None, help="model name for the real backend")
    bq.add_argument("--predictor", default="heuristic", choices=["heuristic", "gnn"])
    bq.add_argument("--selector", default="czt",
                    choices=["czt", "pareto", "hypervolume", "chebyshev"], help="per-node action selection policy")
    bq.add_argument("--realizations", type=int, default=3,
                    help="LLM role-realizations per decision (chance-node branching); 1 = deterministic single-successor transitions")
    bq.add_argument("--uncertainty-gate", type=float, default=0.15,
                    help="fire a completion beam on an interior node only when the predictor's uncertainty >= this. Raise toward 1.0 to value all interiors by the (warm-started) GNN and execute only at terminals -- faster, still query-specific")
    bq.add_argument("--beam-k", type=int, default=3,
                    help="completion-beam width (real executions per uncertain interior node); ignored once --uncertainty-gate is high enough that beams never fire")
    bq.add_argument("--emit", default=None, choices=["python", "yaml"])
    bq.add_argument("--save", action="store_true",
                    help="save the full search tree (and served policies) per query to disk")
    bq.add_argument("--out-dir", default=None,
                    help="directory for saved trees/metrics/policies/session checkpoint (with --save); "
                         "defaults to results/<dataset>/<model>/per_query.")
    bq.add_argument("--checkpoint", default=None,
                    help="path for the session checkpoint. On resume it skips already-solved queries and restores the shared cross-query replay buffer + predictor")
    bq.add_argument("--no-resume", action="store_true",
                    help="ignore any existing session checkpoint and start fresh")
    bq.add_argument("--bundle", default=None,
                    help="trained GNN bundle used FROZEN (read-only) to value interior nodes; never refit or written back. Default shared results/gnn_bundles/<model>/<axes>.pkl (same bundle as per-task)")
    bq.add_argument("--no-bundle", action="store_true",
                    help="don't load a GNN bundle this run; the frozen predictor then falls back to the analytic prior (no learned valuation)")
    bq.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="logging verbosity for the search/backend")
    bq.add_argument("--seed", type=int, default=0)
    _add_early_stop_args(bq)
    _add_wandb_args(bq)
    _add_backend_args(bq)
    _add_device_arg(bq)
    _add_concurrency_arg(bq)
    _add_objective_ref_args(bq)
    bq.set_defaults(func=cmd_build_query)

    # =============================================================================
    inf = sub.add_parser(
        "inference",
        help="serve a stored tree: retrieve the workflow optimal for a preference",
    )
    inf.add_argument("--tree", required=True,
                     help="path to a saved tree JSON.")
    inf.add_argument("--w", required=True, nargs="+",
                     help="one or more comma-separated preference weight vectors over the "
                          "tree's objectives (e.g. --w 1,0,0,0,0 0.2,0.2,0.2,0.2,0.2); "
                          "held-out re-scoring dedups identical served workflows across them")
    inf.add_argument("--objectives", default=None,
                     help="override the tree's objective axes (must match the dumped axes); defaults to the axes recorded in the tree")
    inf.add_argument("--emit", default=None, choices=["python", "yaml"],
                     help="also print the retrieved workflow as Python or YAML")
    inf.add_argument("--task", default=None, choices=[*DATASETS],
                     help="re-score the served policy on this dataset's held-out split and append a row to points.csv; omit for retrieval-only")
    inf.add_argument("--model", default=None,
                     help="model name for the held-out evaluation backend")
    inf.add_argument("--paraphrases", type=int, default=6,
                     help="m: paraphrase variants per held-out problem used for the robustness signal")
    inf.add_argument("--samples", type=int, default=3,
                     help="R: repeated draws when re-scoring on held-out data (their cross-draw accuracy variance gives consistency)")
    inf.add_argument("--out-dir", default=None,
                     help="dir for saved policies/points.csv; default results/<dataset>/<model>/per_task")
    inf.add_argument("--log-level", default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     help="logging verbosity")
    _add_backend_args(inf)
    _add_concurrency_arg(inf)
    _add_objective_ref_args(inf)
    inf.set_defaults(func=cmd_inference)

    # =============================================================================
    dp = sub.add_parser(
        "data",
        help="one-shot dataset preprocessing: download -> paraphrase -> freeze split",
    )
    dp.add_argument("--task", default=next(iter(DATASETS)), choices=[*DATASETS],
                    help="a registered real dataset")
    dp.add_argument("--model", default=None,
                    help="model name for the paraphrase backend (step 2)")
    dp.add_argument("--n", type=int, default=40,
                    help="Number of paraphrases generated per problem (step 2).")
    dp.add_argument("--temperature", type=float, default=0.9,
                    help="sampling temperature for paraphrase diversity (step 2)")
    dp.add_argument("--val-size", type=int, default=None,
                    help="validation examples in the frozen split (step 3); defaults to the dataset's default_val_size")
    dp.add_argument("--test-size", type=int, default=None,
                    help="held-out test examples in the frozen split (step 3); defaults to the dataset's default_test_size, or all remaining after validation. Both subsets are seeded-random draws from the pool")
    dp.add_argument("--seed", type=int, default=0,
                    help="shuffle seed defining the frozen split (step 3)")
    dp.add_argument("--force", action="store_true",
                    help="redo the download + paraphrase steps even if their output exists")
    dp.add_argument("--data-path", default=None,
                    help="override the raw dataset dir (step 1 output / step 2 input)")
    dp.add_argument("--paraphrase-path", default=None,
                    help="override the paraphrase dataset dir (step 2 output / step 3 input)")
    dp.add_argument("--out-dir", default=None,
                    help="override the frozen split dir (step 3 output); defaults to the dataset's canonical split_dir")
    dp.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    _add_backend_args(dp)
    dp.set_defaults(func=cmd_data_process)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
