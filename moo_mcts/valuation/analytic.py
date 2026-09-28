from dataclasses import dataclass

import numpy as np

from ..objectives import ObjectiveSpec
from ..workflow import operators as ops
from ..workflow.graph import WorkflowGraph


@dataclass
class AnalyticBounds:
    tokens_committed: int
    calls_committed: int
    complexity_committed: int  # node count
    tokens_max: int  # committed + worst-case completion
    calls_max: int
    complexity_max: int


def _committed(g: WorkflowGraph) -> tuple[int, int, int]:
    # node_tokens/node_calls honor control semantics (loop=sequential,
    # branch=parallel), so committed cost/latency == what the interpreter measures.
    tokens = sum(ops.node_tokens(n.operator, n.width, n.control_kind) for n in g.nodes)
    calls = sum(ops.node_calls(n.operator, n.width, n.control_kind) for n in g.nodes)
    return tokens, calls, len(g.nodes)


def _cheapest_costliest(max_width: int = 5) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the cheapest and costliest (tokens, calls) any single node can take.

    Scans every operator x width x control kind. Only the costliest pair is used
    (by `bounds`) to size the worst-case completion of a partial graph.
    """
    tok_min = calls_min = 10**9
    tok_max = calls_max = 0
    for spec in ops.all_specs():
        for w in (1, max_width):
            for ck in ("", "loop", "branch"):
                t = ops.node_tokens(spec.name, w, ck)
                c = ops.node_calls(spec.name, w, ck)
                tok_min, calls_min = min(tok_min, t), min(calls_min, c)
                tok_max, calls_max = max(tok_max, t), max(calls_max, c)
    return (tok_min, calls_min), (tok_max, calls_max)


def bounds(g: WorkflowGraph, max_depth: int) -> AnalyticBounds:
    tokens, calls, complexity = _committed(g)
    remaining = max(0, max_depth - g.depth)
    (_, _), (tmax, cmax) = _cheapest_costliest()
    return AnalyticBounds(
        tokens_committed=tokens,
        calls_committed=calls,
        complexity_committed=complexity,
        tokens_max=tokens + remaining * tmax,
        calls_max=calls + remaining * cmax,
        complexity_max=complexity + remaining,
    )


def analytic_vector(
    g: WorkflowGraph,
    spec: ObjectiveSpec,
    *,
    max_depth: int = None,
    **predicted: float,
) -> np.ndarray:

    if g.terminated or max_depth is None:
        tokens, calls, _ = _committed(g)
        tokens, calls = float(tokens), float(calls)
    else:
        bnds = bounds(g, max_depth=max_depth)
        tokens = 0.5 * (bnds.tokens_committed + bnds.tokens_max)
        calls = 0.5 * (bnds.calls_committed + bnds.calls_max)
    # offer all candidate axes; assemble() keeps only those in the chosen set.
    return spec.assemble(
        cost=float(tokens),
        latency=float(calls),
        **{k: float(v) for k, v in predicted.items()},
    )
