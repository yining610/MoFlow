import csv
from dataclasses import dataclass

import numpy as np

from ..ccs.ccs import CCS
from ..config import RunConfig
from ..objectives import ObjectiveSpec, default_spec
from ..reward.evalset import QueryNeighborhood
from ..reward.evaluator import EvalResult, Evaluator, consistency_from_draws
from ..search.archive import Archive
from ..search.engine import Engine
from ..serve.policy import PolicyNode, extract_policy
from ..serve.save_policy import load_bundle
from ..valuation.distill import ReplayBuffer
from ..valuation.predictor import make_predictor
from .common import build_backends


@dataclass
class QueryResult:
    archive: Archive
    spec: ObjectiveSpec
    engine: Engine
    evaluator: Evaluator = None  # kept so served policies can be re-scored held-out


def serve(result: QueryResult, w: np.ndarray) -> PolicyNode | None:
    """Extract the conditional policy optimal for `w` from the per-query tree."""
    return extract_policy(result.engine.root, np.asarray(w, dtype=float), result.spec)


def _heldout_neighborhood(query, search_m: int, m_ho: int, base_seed: int) -> QueryNeighborhood:
    """A held-out neighborhood of `query`: the paraphrase variants the search did NOT use."""
    pool = [query.text] + list(query.paraphrases)
    unseen = pool[search_m:]
    base = unseen if unseen else pool
    paras = base[:m_ho] if m_ho else list(base)
    if m_ho and paras and len(paras) < m_ho:  # pad to the requested width
        paras = paras + [paras[-1]] * (m_ho - len(paras))
    return QueryNeighborhood(paraphrases=paras, gold=query.gold, base_seed=base_seed)


def evaluate_heldout(
        result: QueryResult,
        query,
        w: np.ndarray,
        *,
        search_paraphrases: int,
        n_paraphrases: int = 1,
        samples: int = 3,
        seed: int = 0,
        offset_base: int = 10_000,
) -> tuple[PolicyNode | None, EvalResult | None]:
    """Serve the w-optimal policy and RE-SCORE it on held-out draws of the same query.
    """
    policy = serve(result, w)
    if policy is None or result.evaluator is None:
        return policy, None

    spec = result.spec
    R = max(1, int(samples))
    m_ho = max(1, int(n_paraphrases or 1))
    full = _heldout_neighborhood(query, int(search_paraphrases), m_ho, base_seed=seed + 777)
    rng = np.random.default_rng(seed)

    # accuracy + consistency from R fresh draws of the held-out variant-0 item
    acc_view = full.accuracy_view()
    draws = [
        result.evaluator.evaluate_policy(policy, acc_view, rng, sample_offset=offset_base + r)
        for r in range(R)
    ]
    vec = np.array([d.vector for d in draws]).mean(axis=0)
    if spec.has("accuracy") and spec.has("consistency"):
        vec[spec.names.index("consistency")] = consistency_from_draws(
            [d.per_group_acc for d in draws]
        )
    # robustness from spread across the held-out paraphrase variants
    if spec.has("robustness"):
        rob_idx = spec.names.index("robustness")
        rob = result.evaluator.evaluate_policy(policy, full, rng, sample_offset=offset_base).vector
        vec[rob_idx] = rob[rob_idx]

    return policy, EvalResult(vector=vec, raw=spec.display(vec), per_group_acc=[])


def heldout_hypervolume(vectors: list[np.ndarray], spec: ObjectiveSpec) -> float:
    """Normalized HV of a set of (internal) objective vectors."""
    if not vectors:
        return 0.0
    pts = spec.normalize(np.asarray(vectors, dtype=float))
    return float(CCS(points=pts).hypervolume(np.zeros(spec.D)))


def pooled_front_hv(points_files: list[str], spec: ObjectiveSpec) -> tuple[list[dict], float]:
    """HV-of-means aggregation across queries.
    """
    names = list(spec.names)
    groups: dict[str, list[dict]] = {}
    weights: dict[str, list[float]] = {}
    for path in points_files:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lbl = r["preference_label"]
                groups.setdefault(lbl, []).append({n: float(r[n]) for n in names})
                weights.setdefault(lbl, [float(r[f"w_{n}"]) for n in names])
    front, vectors = [], []
    for lbl, rows in groups.items():
        mean_raw = {n: float(np.mean([d[n] for d in rows])) for n in names}
        vectors.append(spec.assemble(**mean_raw))
        front.append({"preference_label": lbl, "weights": weights[lbl],
                      "objectives": mean_raw, "n_queries": len(rows)})
    return front, heldout_hypervolume(vectors, spec)


class QuerySession:
    """One search session over a stream of queries.
    """

    def __init__(
            self,
            cfg: RunConfig,
            spec: ObjectiveSpec = None,
            checker=None,
            task_profile=None,
            public_tester=None,
            bundle_path: str = None,
    ):
        self.cfg = cfg
        self.spec = spec or default_spec()
        self.checker = checker
        self.task_profile = task_profile
        self.public_tester = public_tester
        self.bundle_path = bundle_path
        self.n_solved = 0

        # backends built ONCE and reused across queries (one client, one role encoder).
        self.proposer, self.make_executor, self.critic = build_backends(
            cfg.backend, self.spec, task_profile=task_profile
        )
        # ONE predictor + buffer, shared across every query in this session.
        self.predictor = make_predictor(
            cfg.predictor.kind, self.spec, self.critic, cfg.search.max_depth,
            buffer=ReplayBuffer(self.spec),
            refit_every=cfg.predictor.refit_every,
            seed=cfg.predictor.seed,
            evidence_scale=cfg.predictor.evidence_scale,
            uncertainty_gate=cfg.search.uncertainty_gate,
            role_encoder_model=cfg.predictor.role_encoder_model,
            device=cfg.predictor.device,
        )
        # warm-start model + buffer from a prior bundle (shared cross-task/cross-mode f_theta)
        load_bundle(bundle_path, self.predictor, self.spec)
        # per-query consumes the GIVEN offline GNN as a fixed valuation function
        if hasattr(self.predictor, "frozen"):
            self.predictor.frozen = True

    @property
    def buffer(self) -> ReplayBuffer:
        """The live replay buffer. Single source of truth: load_bundle / load_state_dict
        reassign predictor.buffer, so always read it off the predictor."""
        return self.predictor.buffer

    def snapshot(self) -> dict:
        """Picklable session state for cross-query crash recovery (n_solved + predictor)."""
        pred_state = None
        if hasattr(self.predictor, "state_dict"):
            pred_state = self.predictor.state_dict()
        return {"version": 2, "n_solved": self.n_solved, "predictor_state": pred_state}

    def restore(self, snap: dict) -> None:
        """Reload a `snapshot()` dict onto this session (in place)."""
        self.n_solved = int(snap.get("n_solved", 0))
        pred_state = snap.get("predictor_state")
        if pred_state is not None and hasattr(self.predictor, "load_state_dict"):
            self.predictor.load_state_dict(pred_state)
        elif snap.get("buffer") is not None:  # v1 back-compat: buffer-only snapshot
            self.predictor.buffer = snap["buffer"]

    def solve(self, query, target_w: np.ndarray = None,
              desc: str = "search") -> QueryResult:
        spec, cfg = self.spec, self.cfg
        # fresh evaluator per query so per-query budget meters stay isolated.
        evaluator = Evaluator(
            spec,
            make_executor=self.make_executor,
            checker=self.checker,
            make_tester=self.public_tester,
            max_workers=cfg.max_workers,
        )
        evalset = query.neighborhood(m=cfg.reward.n_paraphrases)
        engine = Engine(
            spec=spec, config=cfg.search, proposer=self.proposer, evaluator=evaluator,
            predictor=self.predictor, evalset=evalset,  # SHARED predictor across queries
        )

        engine.run(desc=desc)

        if target_w is not None:
            _refine(engine, np.asarray(target_w, dtype=float), cfg)
        self.n_solved += 1
        # No write-back: the offline GNN is frozen (read-only), so there is nothing to save.
        return QueryResult(archive=engine.archive, spec=spec, engine=engine, evaluator=evaluator)


def _refine(engine: Engine, target_w: np.ndarray, cfg: RunConfig) -> None:
    """Reallocate extra trials to a Dirichlet concentrated on `target_w`.

    - target_w: preference the refinement concentrates around.
    """
    n = max(8, cfg.search.n_trials // 5)  # extra 20% of the budget on the target preference
    orig = engine._sample_preference

    def tight():
        # Dirichlet concentrated on target_w (alpha = kappa * w + 1)
        alpha = 40.0 * target_w + 1.0
        return engine.rng.dirichlet(alpha)

    engine._sample_preference = tight  # type: ignore[method-assign]
    try:
        engine.run(n_trials=engine.total_trials + n, desc="refine")
    finally:
        engine._sample_preference = orig  # type: ignore[method-assign]


def build_query_tree(
        query,
        cfg: RunConfig,
        spec: ObjectiveSpec = None,
        target_w: np.ndarray = None,
        checker=None,
        task_profile=None,
        public_tester=None,
        bundle_path: str = None,
    ) -> QueryResult:

    session = QuerySession(
        cfg, spec, checker=checker, task_profile=task_profile,
        public_tester=public_tester, bundle_path=bundle_path,
    )
    return session.solve(query, target_w=target_w)
