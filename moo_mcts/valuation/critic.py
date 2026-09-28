"""Cheap structural value prior (no API call) used as the f_theta bootstrap.

Estimates a maximize-form objective vector from graph structure alone, via the
single source of truth (operators.node_*), so the prior is free and deterministic.
HeuristicPredictor re-pins the exact analytic axes on top.
"""


import numpy as np

from ..objectives import ObjectiveSpec
from ..workflow import operators as ops


class HeuristicPrior:
    """Cheap structural value prior (no API call) reused as the f_theta bootstrap.

    Estimates a maximize-form vector from graph structure alone (via the single
    source of truth, operators.node_*), so the prior is free and deterministic;
    """

    def __init__(self, spec: ObjectiveSpec, seed: int = 0):
        self.spec = spec
        self.seed = seed

    def value(self, graph) -> tuple[np.ndarray, float]:
        # structural prior from the single source of truth (control-aware), so the
        # bootstrap matches the analytic axes.
        acc_logit = 0.0
        rob = 0.0
        cons = 0.0
        for n in graph.nodes:
            a_gain, r_gain, c_gain = ops.node_quality_gain(n.operator, n.width, n.control_kind)
            acc_logit += a_gain
            rob = min(1.0, rob + r_gain)
            cons = min(1.0, cons + c_gain)
        acc = float(1.0 / (1.0 + np.exp(-acc_logit)))
        tokens = sum(ops.node_tokens(n.operator, n.width, n.control_kind) for n in graph.nodes)
        calls = sum(ops.node_calls(n.operator, n.width, n.control_kind) for n in graph.nodes)
        # offer all candidate axes; assemble() keeps only those in the chosen set.
        est = self.spec.assemble(
            accuracy=acc, cost=float(tokens), latency=float(calls),
            robustness=float(rob), consistency=float(cons),
        )
        unc = float(np.clip(0.5 / (1 + graph.depth), 0.02, 0.5))
        return est, unc
