#!/usr/bin/env python
"""Train the workflow-value GNN on pooled multi-task search data.
"""
import argparse
import csv
import json
import os
import sys

import torch
import numpy as np
from tqdm.auto import tqdm

# Make `import moo_mcts` / `import tasks` / `import scripts.gnn.collect` work as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from moo_mcts.config import PredictorConfig
from moo_mcts.logging_util import configure as configure_logging
from moo_mcts.logging_util import get_logger
from moo_mcts.objectives import default_spec
from moo_mcts.serve.save_policy import default_bundle_path, load_bundle, save_bundle
from moo_mcts.valuation import graph_encoder as ge
from moo_mcts.valuation.gnn_predictor import GNNValuePredictor

from scripts.gnn.collect import (DEFAULT_EXP, DEFAULT_MODEL_SLUG, DEFAULT_TASKS,
                                  collect_multitask, load_dataset)

log = get_logger("gnn_train")

_DEVICE_TO_TORCH = {"auto": None, "cpu": "cpu", "gpu": "cuda"}


def _encode(pairs, role_encode):
    """Pre-encode (graph, y) pairs into (per-graph sample dicts, stacked targets Y)."""
    samples = [ge.graph_to_tensors(g, role_encode) for g, _ in pairs]
    Y = (np.stack([np.asarray(y, dtype=np.float32) for _, y in pairs])
         if pairs else np.zeros((0, 0), np.float32))
    return samples, Y


def _forward(predictor, samples, idx):
    """Deterministic (eval-mode) forward over a subset of pre-encoded samples -> (n, P)."""
    batch = ge.collate([samples[i] for i in idx])
    inp = predictor._to_torch(batch)
    predictor.model.eval()
    with torch.no_grad():
        pred = predictor.model(**inp)
    return pred.cpu().numpy()


def _axis_metrics(pred: np.ndarray, y: np.ndarray, names) -> dict:
    """Overall + per-axis MSE / MAE / R^2 / Pearson r (R^2 & r are nan on zero-variance axes)."""
    err = pred - y
    out = {
        "n": int(len(y)),
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
    }
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
    out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    for i, name in enumerate(names):
        yi, pi = y[:, i], pred[:, i]
        tot = float(np.sum((yi - yi.mean()) ** 2))
        out[f"mse/{name}"] = float(np.mean((pi - yi) ** 2))
        out[f"mae/{name}"] = float(np.mean(np.abs(pi - yi)))
        out[f"r2/{name}"] = 1.0 - float(np.sum((pi - yi) ** 2)) / tot if tot > 0 else float("nan")
        if yi.std() > 0 and pi.std() > 0:
            out[f"r/{name}"] = float(np.corrcoef(pi, yi)[0, 1])
        else:
            out[f"r/{name}"] = float("nan")
    return out


def _fmt_metrics(tag: str, m: dict, names) -> str:
    cols = " ".join(f"{n}:R2={m[f'r2/{n}']:+.3f}/MAE={m[f'mae/{n}']:.3f}" for n in names)
    return f"{tag:<5} n={m['n']:<4} MSE={m['mse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:+.3f} | {cols}"


def _clone_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train(args) -> None:
    configure_logging(args.log_level)
    spec = default_spec()
    names = list(spec.predicted_names)  # [accuracy, robustness, consistency]
    device = _DEVICE_TO_TORCH[args.device]
    rng = np.random.default_rng(args.seed)

    # ---- 1. get the pooled dataset
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    overrides = {}
    for spec_str in (args.checkpoint or []):
        if "=" not in spec_str:
            raise SystemExit(f"--checkpoint expects task=path, got {spec_str!r}")
        k, v = spec_str.split("=", 1)
        overrides[k.strip()] = v.strip()

    if args.dataset:
        ds = load_dataset(args.dataset, spec=spec)
        print(f"[data] loaded pooled dataset from {args.dataset}")
    else:
        ds = collect_multitask(tasks, spec=spec, model_slug=args.model_slug, exp=args.exp,
                               checkpoint_overrides=overrides, collect_missing=args.collect_missing)
    print(ds.summary())
    if ds.n_train < 8:
        raise SystemExit(f"only {ds.n_train} pooled train pairs (<8); nothing to train on. "
                         "Check --tasks / --model-slug / --exp or run the searches first.")

    # ---- 2. build the production GNN
    predictor = GNNValuePredictor(
        spec, critic=None, max_depth=args.max_depth,
        hidden=args.hidden, role_dim=args.role_dim, num_layers=args.layers,
        buffer=ds.buffer, device=device, seed=args.seed,
        role_encoder_model=args.role_encoder_model,
    )
    if predictor._degenerate:
        raise SystemExit("spec has no predicted axes; nothing to learn")
    if ds.role_cache:
        predictor.encoder.load_cache(ds.role_cache)

    # ---- 3. optional finetune warm-start
    if args.init_bundle:
        meta = load_bundle(args.init_bundle, predictor, spec)
        if meta is None:
            log.warning("init bundle %s not applied (missing/incompatible); training fresh",
                        args.init_bundle)
        predictor.buffer = ds.buffer
        print(f"[finetune] warm-started from {args.init_bundle}")

    model = predictor.model
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    predictor.opt = opt  # so the saved bundle carries this optimizer state
    loss_fn = torch.nn.MSELoss()

    # ---- 4. pre-encode once (role encoder caches; misses hit sentence-transformers)
    tr_samples, tr_Y = _encode(ds.train_pairs, predictor.encoder.encode)
    va_samples, va_Y = _encode(ds.test_pairs, predictor.encoder.encode)
    tr_Yt = torch.tensor(tr_Y).to(predictor.device)
    n = len(tr_samples)
    bs = args.batch_size if args.batch_size and args.batch_size > 0 else n

    run = _init_wandb(args, ds, spec)

    # ---- 5. train with early stopping on val MSE, keep the best epoch
    best_val = float("inf")
    best_state = _clone_state(model)
    best_epoch = 0
    bad = 0
    curve: list[dict] = []
    has_val = len(va_samples) > 0
    
    bar = tqdm(range(1, args.epochs + 1), desc="train GNN", unit="epoch", disable=None)
    for epoch in bar:
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            batch = ge.collate([tr_samples[i] for i in idx])
            inp = predictor._to_torch(batch)
            yt = tr_Yt[torch.as_tensor(idx, device=predictor.device)]
            opt.zero_grad()
            loss = loss_fn(model(**inp), yt)
            loss.backward()
            opt.step()

        tr_m = _axis_metrics(_forward(predictor, tr_samples, range(n)), tr_Y, names)
        va_m = _axis_metrics(_forward(predictor, va_samples, range(len(va_samples))), va_Y, names) if has_val else None
        row = {"epoch": epoch, "train_mse": tr_m["mse"], "train_mae": tr_m["mae"],
               "train_r2": tr_m["r2"]}
        if va_m:
            row.update({"val_mse": va_m["mse"], "val_mae": va_m["mae"], "val_r2": va_m["r2"]})
        curve.append(row)
        _log_wandb_epoch(run, epoch, tr_m, va_m)

        monitor = va_m["mse"] if va_m else tr_m["mse"]
        improved = monitor < best_val - args.min_delta
        if improved:
            best_val, best_state, best_epoch, bad = monitor, _clone_state(model), epoch, 0
        else:
            bad += 1

        # live metrics in the bar: train/test MSE+R2, best-so-far, and the patience counter
        post = {"tr_mse": f"{tr_m['mse']:.4f}", "tr_R2": f"{tr_m['r2']:+.2f}"}
        if va_m:
            post.update({"te_mse": f"{va_m['mse']:.4f}", "te_R2": f"{va_m['r2']:+.2f}"})
        post["best"] = f"{best_val:.4f}@{best_epoch}"
        post["patience"] = f"{bad}/{args.patience}"
        bar.set_postfix(post, refresh=False)

        if epoch == 1 or epoch % args.log_every == 0:
            tqdm.write(f"epoch {epoch:>4}/{args.epochs}  " + _fmt_metrics("train", tr_m, names)
                       + (f"\n              {_fmt_metrics('test', va_m, names)}" if va_m else ""))
        if not improved and bad >= args.patience:
            tqdm.write(f"[early-stop] no val improvement for {args.patience} epochs "
                       f"(best {'val' if has_val else 'train'} MSE {best_val:.4f} @ epoch {best_epoch})")
            break
    bar.close()

    # ---- 6. restore best weights, mark trained
    model.load_state_dict(best_state)
    model.to(predictor.device)
    predictor._trained = True

    # ---- 7. final report
    tr_pred = _forward(predictor, tr_samples, range(n))
    tr_m, tr_err = _axis_metrics(tr_pred, tr_Y, names), (tr_pred - tr_Y)
    va_m = va_err = None
    if has_val:
        va_pred = _forward(predictor, va_samples, range(len(va_samples)))
        va_m, va_err = _axis_metrics(va_pred, va_Y, names), (va_pred - va_Y)
    per_task_train = _per_task_metrics(predictor, ds.train_by_task, names)
    per_task_test = _per_task_metrics(predictor, ds.test_by_task, names)

    print(f"\n=== best epoch {best_epoch} ===")
    print(_fmt_metrics("train", tr_m, names))
    if va_m:
        print(_fmt_metrics("test", va_m, names))
    if per_task_test:
        print("per-task (test partition):")
        for task, m in per_task_test.items():
            print(f"  {task:<10} " + _fmt_metrics("test", m, names))

    # ---- 8. save a production-loadable bundle + metrics artifacts
    out_bundle = args.out_bundle or _default_out_bundle(spec, args.model_slug, tasks)
    save_bundle(out_bundle, predictor, spec, trials=None)
    print(f"\n[save] GNN bundle -> {out_bundle}")

    _write_metrics(args, ds, spec, names, tr_m, va_m, best_epoch, curve,
                   per_task_train, per_task_test)
    _write_curve(args, curve)
    plot_path = _write_plots(args, predictor, va_samples, va_Y, names)
    _log_wandb_final(run, names, best_epoch, tr_m, va_m, tr_err, va_err,
                     per_task_train, per_task_test, plot_path)
    if run is not None:
        run.finish()


def _default_out_bundle(spec, model_slug: str, tasks: list) -> str:
    """Default bundle path: results/gnn_bundles/<slug>/<task_tag>/<axes>.pkl.
    """
    base = os.path.dirname(default_bundle_path(model_slug, spec))
    axes = "-".join(spec.predicted_names) or "none"
    task_tag = "-".join(tasks) if tasks else "multitask"
    return os.path.join(base, task_tag, f"{axes}.pkl")


def _write_metrics(args, ds, spec, names, tr_m, va_m, best_epoch, curve,
                   per_task_train=None, per_task_test=None) -> None:
    if not args.metrics_out:
        return
    payload = {
        "tasks": [t.strip() for t in args.tasks.split(",") if t.strip()],
        "predicted_axes": names,
        "per_task_stats": ds.per_task_stats,
        "n_train": ds.n_train, "n_test": ds.n_test,
        "config": {"epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
                   "batch_size": args.batch_size, "patience": args.patience,
                   "hidden": args.hidden, "role_dim": args.role_dim, "num_layers": args.layers,
                   "seed": args.seed,
                   "init_bundle": args.init_bundle, "role_encoder_model": args.role_encoder_model},
        "best_epoch": best_epoch,
        "final": {"train": tr_m, "test": va_m},
        "per_task": {"train": per_task_train or {}, "test": per_task_test or {}},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)) or ".", exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[save] metrics -> {args.metrics_out}")


def _write_curve(args, curve) -> None:
    if not args.curve_out or not curve:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.curve_out)) or ".", exist_ok=True)
    keys = list(curve[0].keys())
    with open(args.curve_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(curve)
    print(f"[save] training curve -> {args.curve_out}")


def _write_plots(args, predictor, va_samples, va_Y, names) -> str | None:
    """Write a test-partition parity plot; return its path (or None)."""
    if not args.plots or len(va_samples) == 0:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib optional
        log.warning("--plots requested but matplotlib unavailable (%s); skipping", exc)
        return None
    pred = _forward(predictor, va_samples, range(len(va_samples)))
    os.makedirs(args.plots, exist_ok=True)
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    axes = np.atleast_1d(axes)
    for i, name in enumerate(names):
        ax = axes[i]
        ax.scatter(va_Y[:, i], pred[:, i], s=14, alpha=0.6)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"true {name}"); ax.set_ylabel(f"pred {name}")
        ax.set_title(name)
    fig.tight_layout()
    path = os.path.join(args.plots, "test_parity.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[save] test parity plot -> {path}")
    return path


def _per_task_metrics(predictor, pairs_by_task, names) -> dict:
    """Per-task axis metrics (MSE/MAE/R2 overall + per axis) on the given pairs."""
    out: dict = {}
    for task, pairs in pairs_by_task.items():
        samples, Y = _encode(pairs, predictor.encoder.encode)
        if not samples:
            continue
        pred = _forward(predictor, samples, range(len(samples)))
        out[task] = _axis_metrics(pred, Y, names)
    return out


def _init_wandb(args, ds, spec):
    if not args.wandb:
        return None
    try:
        import wandb
    except Exception as exc:
        log.warning("--wandb requested but wandb unavailable (%s); skipping", exc)
        return None
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"gnn_multitask_{'-'.join(tasks)}",
        config={
            "tasks": tasks, "model_slug": args.model_slug, "exp": args.exp,
            "n_train": ds.n_train, "n_test": ds.n_test,
            "per_task_counts": {s["task"]: {"n_train": s.get("n_train", 0),
                                            "n_test": s.get("n_test", 0)}
                                for s in ds.per_task_stats},
            "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "patience": args.patience,
            "hidden": args.hidden, "role_dim": args.role_dim, "num_layers": args.layers,
            "seed": args.seed,
            "init_bundle": args.init_bundle, "axes": list(spec.predicted_names),
        },
    )
    # plot every metric against the epoch counter (guarded: older wandb lacks define_metric)
    try:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("test/*", step_metric="epoch")
    except Exception:
        pass
    return run


def _log_wandb_epoch(run, epoch, tr_m, va_m) -> None:
    """Per-epoch scalar curves for the pooled train + test partitions (overall + per axis)."""
    if run is None:
        return
    metrics = {"epoch": epoch}
    metrics.update({f"train/{k}": v for k, v in tr_m.items()})
    if va_m:
        metrics.update({f"test/{k}": v for k, v in va_m.items()})
    run.log(metrics, step=epoch)


def _log_wandb_final(run, names, best_epoch, tr_m, va_m, tr_err, va_err,
                     per_task_train, per_task_test, plot_path) -> None:
    """Log best-epoch summary, per-axis error histograms, a per-task table, and the parity plot."""
    if run is None:
        return
    import wandb

    run.summary["best_epoch"] = best_epoch
    # headline: final pooled train/test metrics (overall + per axis)
    for tag, m in (("train", tr_m), ("test", va_m)):
        if m:
            for k, v in m.items():
                run.summary[f"final/{tag}/{k}"] = v
    # per-axis error distributions on each partition
    for tag, err in (("train", tr_err), ("test", va_err)):
        if err is None:
            continue
        run.summary[f"final/{tag}/err_hist"] = wandb.Histogram(err.reshape(-1))
        for i, name in enumerate(names):
            run.summary[f"final/{tag}/err_hist/{name}"] = wandb.Histogram(err[:, i])
    # per-task train/test breakdown: a table + per-task summary scalars
    cols = ["task", "split", "n", "mse", "mae", "r2"] + [f"r2/{n}" for n in names]
    tbl = wandb.Table(columns=cols)
    for split, per_task in (("train", per_task_train), ("test", per_task_test)):
        for task, m in (per_task or {}).items():
            tbl.add_data(task, split, m["n"], m["mse"], m["mae"], m["r2"],
                         *[m[f"r2/{n}"] for n in names])
            for k in ("mse", "mae", "r2"):
                run.summary[f"per_task/{split}/{task}/{k}"] = m[k]
    run.log({"per_task_metrics": tbl})
    if plot_path and os.path.exists(plot_path):
        run.log({"test_parity": wandb.Image(plot_path)})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                   help=f"comma-separated subset of task keys to pool (default: all "
                        f"{len(DEFAULT_TASKS)} -- {','.join(DEFAULT_TASKS)})")
    p.add_argument("--model-slug", default=DEFAULT_MODEL_SLUG,
                   help="model dir slug used in the checkpoint paths")
    p.add_argument("--exp", default=DEFAULT_EXP, help="run subdir under results/<task>/<model_slug>/per_task/ (default: %(default)s)")
    p.add_argument("--dataset", default=None,
                   help="load a pooled dataset .pkl produced by phase 1 "
                        "(scripts/gnn/collect.py --save) instead of re-pooling the checkpoints")
    p.add_argument("--checkpoint", action="append", default=None, metavar="TASK=PATH",
                   help="override a task's checkpoint path (repeatable)")
    p.add_argument("--collect-missing", action="store_true",
                   help="regenerate a missing checkpoint via build_task_tree (needs the backend)")
    p.add_argument("--init-bundle", default=None,
                   help="warm-start (finetune) the GNN weights from this bundle before training")
    p.add_argument("--out-bundle", default=None,
                   help="output bundle path (default results/gnn_bundles/<slug>/<task_tag>/<axes>.pkl)")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam lr. 1e-3 suits the pooled set; production's per-task 1e-2 overfits here")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=0, help="0 = full-batch (matches production)")
    p.add_argument("--patience", type=int, default=100,
                   help="early-stop after this many epochs without val-MSE improvement")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="minimum val-MSE decrease that counts as improvement")
    p.add_argument("--hidden", type=int, default=64, help="GNN width (message-passing channel dim)")
    p.add_argument("--role-dim", type=int, default=8, help="role-embedding width")
    p.add_argument("--layers", type=int, default=2,
                   help="GNN depth (number of message-passing layers)")
    p.add_argument("--max-depth", type=int, default=8,
                   help="predictor max_depth (unused during training; stored for compatibility)")
    p.add_argument("--role-encoder-model", default=PredictorConfig().role_encoder_model)
    p.add_argument("--device", default="auto", choices=list(_DEVICE_TO_TORCH))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--metrics-out", default=None, help="write a JSON metrics summary here")
    p.add_argument("--curve-out", default=None, help="write the per-epoch training curve CSV here")
    p.add_argument("--plots", default=None, help="dir for a val parity plot (needs matplotlib)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="moo-mcts-gnn")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
