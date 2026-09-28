"""Per-task driver: build one tree, then serve any preference by retrieval."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from ..config import RunConfig
from ..logging_util import get_logger
from ..objectives import ObjectiveSpec, default_spec
from ..reward.evalset import TaskValidationSplit
from ..reward.evaluator import EvalResult, Evaluator, consistency_from_draws
from ..search.archive import Archive
from ..search.checkpoint import Checkpointer
from ..search.engine import Engine
from ..serve.policy import PolicyNode, extract_policy
from ..serve.save_policy import load_bundle, save_bundle
from ..valuation.distill import ReplayBuffer
from ..valuation.predictor import make_predictor
from .common import build_backends

log = get_logger("per_task")


@dataclass
class TaskResult:
    archive: Archive
    spec: ObjectiveSpec
    engine: Engine
    evaluator: Optional[Evaluator] = None
    checker: Optional[Callable] = None


def build_task_tree(
    task,
    cfg: RunConfig,
    spec: ObjectiveSpec = None,
    checker: Callable = None,
    task_profile: Any = None,
    checkpoint_path: str = None,
    resume: bool = True,
    metrics_path: str = None,
    weights_path: str = None,
    bundle_path: str = None,
    public_tester: Callable = None,
    offline: bool = False,
) -> TaskResult:
    """Build one per-task search tree and return its archive and engine.

    Args:
        task: The task object providing the validation split.
        cfg: Run configuration.
        spec: Objective specification.
        checker: Optional callable to check solutions.
        task_profile: Optional task profile.
        checkpoint_path: Path to checkpoint for resuming.
        resume: Whether to resume from checkpoint.
        metrics_path: Path to save metrics.
        weights_path: Path to save weights.
        bundle_path: Path to a GNN bundle for valuation.
        public_tester: Optional callable for public testing.
        offline: Whether to run in offline mode (GNN-driven valuation).

    Returns:
        TaskResult containing the archive, spec, engine, evaluator, and checker.
    """
    spec = spec or default_spec()
    proposer, make_executor, critic = build_backends(cfg.backend, spec, task_profile=task_profile)
    evaluator = Evaluator(
        spec,
        make_executor=make_executor,
        checker=checker,
        make_tester=public_tester,
        max_workers=cfg.max_workers,
    )
    buffer = ReplayBuffer(spec)
    predictor = make_predictor(
        cfg.predictor.kind, spec, critic, cfg.search.max_depth,
        buffer=buffer, 
        refit_every=cfg.predictor.refit_every,
        seed=cfg.predictor.seed,
        evidence_scale=cfg.predictor.evidence_scale,
        uncertainty_gate=cfg.search.uncertainty_gate,
        role_encoder_model=cfg.predictor.role_encoder_model,
        device=cfg.predictor.device,
    )

    # warm-start model + buffer from a prior bundle
    load_bundle(bundle_path, predictor, spec)
    evalset = task.validation_split(n_paraphrases=cfg.reward.n_paraphrases)
    checkpointer = Checkpointer(checkpoint_path) if checkpoint_path else None
    engine = Engine(
        spec=spec,
        config=cfg.search,
        proposer=proposer,
        evaluator=evaluator,
        predictor=predictor,
        evalset=evalset,
        checkpointer=checkpointer,
        resume=resume,
        metrics_path=metrics_path,
        weights_path=weights_path,
        bundle_path=bundle_path,
        offline=offline,
    )
    if offline:
        load_bundle(bundle_path, engine.predictor, spec)
        if not getattr(engine.predictor, "_trained", False):
            raise ValueError(
                f"offline mode requires a trained GNN bundle; {bundle_path!r} is missing, "
                "untrained, or has incompatible objective axes"
            )
    engine.run()
    if not offline:
        save_bundle(bundle_path, engine.predictor, spec, trials=engine.total_trials)

    return TaskResult(
        archive=engine.archive, spec=spec, engine=engine,
        evaluator=evaluator, checker=checker,
    )


def serve(result: TaskResult, w: np.ndarray) -> PolicyNode | None:
    """Extract the conditional policy optimal for preference w.

    Recursively extracts from the root V-hat: at each decision pick the action whose
    Q-hat supports the w-optimal vector, branching on each realized successor by its
    empirical P(s'|s,a).
    """
    return extract_policy(result.engine.root, np.asarray(w, dtype=float), result.spec)


def evaluate_heldout(
    result: TaskResult, 
    w: np.ndarray, 
    heldout: list[tuple], 
    seeds: int = 1,
    n_paraphrases: int = 1, 
    samples: int = 3,
):
    """Re-score the served policy on held-out data, grounding each axis separately.

    Args:
        result: TaskResult containing the engine and evaluation setup.
        w: Preference vector for which the policy is extracted.
        heldout: List of held-out examples to evaluate on.
        seeds: Number of random seeds for held-out evaluation.
        n_paraphrases: Number of paraphrases per example.
        samples: Number of evaluation samples per seed.

    Returns:
        A tuple of the served policy and the evaluation result (EvalResult) on held-out data.
    """
    policy = serve(result, w)
    if policy is None or not heldout or result.evaluator is None:
        return policy, None

    spec = result.spec
    full = TaskValidationSplit(
        examples=list(heldout), seeds=seeds, n_paraphrases=n_paraphrases
    )
    rng = np.random.default_rng(0)

    # compute consistency
    R = max(1, samples)
    acc_view = full.accuracy_view()
    eval_draws = [
        result.evaluator.evaluate_policy(policy, acc_view, rng, sample_offset=r)
        for r in range(R)
    ]
    draws = np.array([res.vector for res in eval_draws])
    vec = draws.mean(axis=0)

    acc_idx = spec.names.index("accuracy") if spec.has("accuracy") else None
    cons_idx = spec.names.index("consistency") if spec.has("consistency") else None
    if acc_idx is not None and cons_idx is not None:
        vec[cons_idx] = consistency_from_draws(
            [res.per_group_acc for res in eval_draws]
        )

    # compute robustness
    if spec.has("robustness"):
        rob_idx = spec.names.index("robustness")
        rob_vec = result.evaluator.evaluate_policy(policy, full, rng, sample_offset=0).vector
        vec[rob_idx] = rob_vec[rob_idx]

    res = EvalResult(vector=vec, raw=spec.display(vec), per_group_acc=[])
    return policy, res
