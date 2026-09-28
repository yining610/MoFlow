import asyncio
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..concurrency import parallel_map
from ..logging_util import get_logger
from ..objectives import ObjectiveSpec
from ..workflow.graph import WorkflowGraph
from ..workflow.interpret_yaml import run_graph
from ..serve.execute import sample_terminal
from .evalset import EvalItem, EvalSetProvider

log = get_logger("evaluator")


@dataclass
class EvalResult:
    vector: np.ndarray  # maximize-form R in R^D
    raw: dict  # raw {accuracy, cost, latency, robustness}
    per_group_acc: list[float]  # per-problem accuracy on the original (variant-0) query


def consistency_from_draws(group_accs: list) -> float:
    arr = np.asarray(group_accs, dtype=float)  # (R, n_problems)
    per_problem_var = arr.var(axis=0)  # variance across draws, per problem
    return float(np.clip(1.0 - 4.0 * float(per_problem_var.mean()), 0.0, 1.0))


class Evaluator:
    """Scores a completed workflow against an eval set, returning a reward vector.
    """

    def __init__(self, spec: ObjectiveSpec, make_executor, checker, make_tester=None, max_workers: int = 8):
        if checker is None:
            raise ValueError("Evaluator requires a checker to grade produced answers")
        self.spec = spec
        self.make_executor = make_executor
        self.checker = checker
        self.make_tester = make_tester
        self.max_workers = max_workers
        self._meter_lock = threading.Lock()
        self.n_evaluations = 0  # evaluate() calls = workflow x full-eval-set runs
        self.n_executions = 0   # item-level executions summed over all calls
        self.total_tokens = 0   # raw token budget (sum of per-item tokens)
        self.total_calls = 0    # raw sequential LLM calls (sum of per-item calls)

    def evaluate(
        self, graph: WorkflowGraph, evalset: EvalSetProvider, sample_offset: int = 0
    ) -> EvalResult:
        return asyncio.run(
            self._evaluate_async(
                graph, evalset, sample_offset
            )
        )

    def evaluate_policy(self, policy, evalset: EvalSetProvider, rng, sample_offset: int = 0) -> EvalResult:
        """Score a conditional policy, aggregating to its expected reward vector.
        """

        return asyncio.run(
            self._evaluate_async(
                None, evalset, sample_offset, selector=lambda: sample_terminal(policy, rng)
            )
        )

    def _run_one(self, item, offset: int, graph):
        """Run `graph` on one eval item at `offset`; return its per-item record.
        """
        item_seed = item.seed + offset * 1_000_003  # large stride per sample
        ex = self.make_executor(item_seed)  # fresh executor
        ex.reset()
        if self.make_tester is not None and hasattr(ex, "set_public_tester"):
            ex.set_public_tester(self.make_tester(item.gold))
        t0 = time.monotonic()
        asyncio.run(run_graph(graph, item.payload, ex))
        acc = float(self.checker(ex.solution(), item))
        log.debug(
            "  item group=%s variant=%s offset=%d done in %.1fs (acc=%g, %d tokens, %d calls)",
            item.group, item.variant, offset, time.monotonic() - t0, acc,
            ex.meta.tokens, ex.meta.sequential_calls,
        )
        return acc, ex.meta.tokens, ex.meta.sequential_calls, item.group, item.variant

    def evaluate_terminal(
        self, graph: WorkflowGraph, evalset: EvalSetProvider, R: int,
        want_consistency: bool = True, want_robustness: bool = True,
    ) -> tuple[list[np.ndarray], float | None, float | None]:
        """Value a terminal workflow in one deduplicated, fully-overlapped pass.
        """
        R = max(1, int(R))
        items = evalset.items()
        if not items:
            raise ValueError("eval set is empty")
        variant0 = [it for it in items if it.variant == 0]
        others = [it for it in items if it.variant != 0]
        if not variant0:
            raise ValueError("eval set has no variant-0 (original) items")

        # union of work units: (item, offset)
        units: list[tuple] = [(it, r) for r in range(R) for it in variant0]
        if want_robustness:
            units += [(it, 0) for it in others]

        def _runner(u):
            it, offset = u
            return (offset, *self._run_one(it, offset, graph))

        results = parallel_map(_runner, units, self.max_workers)

        # meter: n_evaluations keeps the logical R-draws + 1-robustness-pass semantics so
        # the saved budget curve stays comparable; n_executions/tokens/calls reflect the
        # (deduplicated) real work actually done.
        with self._meter_lock:
            self.n_evaluations += R + (1 if want_robustness else 0)
            self.n_executions += len(units)
            self.total_tokens += int(sum(res[2] for res in results))
            self.total_calls += int(sum(res[3] for res in results))

        # per-draw accuracy (variant 0 only), keyed by offset r
        draw_acc: dict[int, list[float]] = {}
        draw_tokens: dict[int, list[int]] = {}
        draw_calls: dict[int, list[int]] = {}
        draw_group_accs: dict[int, dict[int, list[float]]] = {}  # offset -> group -> [accs]
        # robustness cell: (group, variant) -> [accs] at offset 0
        rob_cell: dict[tuple[int, int], list[float]] = {}

        for offset, acc, tokens, calls, group, variant in results:
            if variant == 0:
                draw_acc.setdefault(offset, []).append(acc)
                draw_tokens.setdefault(offset, []).append(tokens)
                draw_calls.setdefault(offset, []).append(calls)
                draw_group_accs.setdefault(offset, {}).setdefault(group, []).append(acc)
            if offset == 0:
                rob_cell.setdefault((group, variant), []).append(acc)

        samples: list[np.ndarray] = [
            self.spec.assemble(
                accuracy=float(np.mean(draw_acc[r])),
                cost=float(np.mean(draw_tokens[r])),
                latency=float(np.mean(draw_calls[r])),
                robustness=1.0,
                consistency=1.0,  # placeholder; overwritten by the caller
            )
            for r in range(R)
        ]

        consistency = None
        if want_consistency:
            per_group_acc_draws = [
                [float(np.mean(draw_group_accs[r][g])) for g in sorted(draw_group_accs[r])]
                for r in range(R)
            ]
            consistency = consistency_from_draws(per_group_acc_draws)

        robustness = None
        if want_robustness:
            variant_means: dict[int, list[float]] = {}
            for (group, _variant), accs in rob_cell.items():
                variant_means.setdefault(group, []).append(float(np.mean(accs)))
            robustness = 1.0
            if max((len(m) for m in variant_means.values()), default=0) >= 2:
                per_problem_rob: list[float] = []
                for group, means in variant_means.items():
                    assert len(means) >= 2, (
                        f"problem {group} has only {len(means)} paraphrase variant(s); a "
                        f"paraphrase neighbourhood must carry >= 2 variants per problem"
                    )
                    var = float(np.var(means))  # 4x: var in [0,.25] for binary accs
                    per_problem_rob.append(float(np.clip(1.0 - 4.0 * var, 0.0, 1.0)))
                robustness = float(np.mean(per_problem_rob))

        return samples, consistency, robustness

    async def _evaluate_async(
        self,
        graph: WorkflowGraph | None,
        evalset: EvalSetProvider,
        sample_offset: int = 0,
        selector=None,
    ) -> EvalResult:
        """Run `graph` on the eval set and aggregate one maximize-form reward.
        """
        items = evalset.items()
        if not items:
            raise ValueError("eval set is empty")

        per_item_tokens: list[int] = []
        per_item_calls: list[int] = []
        orig_acc: list[float] = []  # accuracy on original items only
        # cell[(problem, paraphrase)]
        cell: dict[tuple[int, int], list[float]] = {}

        log.debug("eval over %d items (sample_offset=%d)", len(items), sample_offset)

        graphs = [
            (selector() if selector is not None else graph) for _ in items
        ]

        def _run_item(idx_it_g):
            _idx, it, g = idx_it_g
            return self._run_one(it, sample_offset, g)

        # Items are independent; run them concurrently in threads
        results = await asyncio.to_thread(
            parallel_map, _run_item, list(zip(range(len(items)), items, graphs)),
            self.max_workers,
        )
        for acc, tokens, calls, group, variant in results:
            per_item_tokens.append(tokens)
            per_item_calls.append(calls)
            cell.setdefault((group, variant), []).append(acc)
            if variant == 0:
                orig_acc.append(acc)

        with self._meter_lock:
            self.n_evaluations += 1
            self.n_executions += len(items)
            self.total_tokens += int(sum(per_item_tokens))
            self.total_calls += int(sum(per_item_calls))

        mean_acc = float(np.mean(orig_acc))
        mean_tokens = float(np.mean(per_item_tokens))  # structural; over all items
        mean_calls = float(np.mean(per_item_calls))

        variant_means: dict[int, list[float]] = {}
        orig_means: dict[int, float] = {}
        for (group, variant), accs in cell.items():
            m = float(np.mean(accs))
            # record the accuracy of each variant for this problem
            variant_means.setdefault(group, []).append(m)
            if variant == 0:
                orig_means[group] = m

        robustness = 1.0
        if max((len(m) for m in variant_means.values()), default=0) >= 2:
            per_problem_rob: list[float] = []
            for group, means in variant_means.items():
                assert len(means) >= 2, (
                    f"problem {group} has only {len(means)} paraphrase variant(s); a "
                    f"paraphrase neighbourhood must carry >= 2 variants per problem"
                )
                var = float(np.var(means))  # 4x: var in [0,.25] for binary accs
                per_problem_rob.append(float(np.clip(1.0 - 4.0 * var, 0.0, 1.0)))
            robustness = float(np.mean(per_problem_rob))

        vector = self.spec.assemble(
            accuracy=mean_acc,
            cost=mean_tokens,
            latency=mean_calls,
            robustness=robustness,
            consistency=1.0, # placeholder value; the real value is computed at engine
        )
        raw = self.spec.display(vector)
        return EvalResult(
            vector=vector,
            raw=raw,
            per_group_acc=[orig_means[g] for g in sorted(orig_means)], # accuracy for each original problem
        )