import numpy as np

from ..objectives import ObjectiveSpec
from ..valuation.analytic import analytic_vector
from ..workflow.graph import WorkflowGraph


class HeuristicPredictor:

    def __init__(self, spec: ObjectiveSpec, critic, max_depth: int, buffer=None):
        self.spec = spec
        self.critic = critic
        self.max_depth = max_depth
        self.buffer = buffer  # optional distill.ReplayBuffer

    def predict(self, graph: WorkflowGraph, *, stochastic: bool = True) -> tuple[np.ndarray, float]:
        est, unc = self.critic.value(graph)
        raw = self.spec.display(est)

        predicted = {n: raw[n] for n in self.spec.predicted_names}
        vec = analytic_vector(graph, self.spec, max_depth=self.max_depth, **predicted)
        return vec, float(unc)

    def observe(self, graph: WorkflowGraph, target_vector: np.ndarray) -> None:
        if self.buffer is not None:
            self.buffer.add(graph, target_vector)

    def maybe_refit(self) -> bool:
        return False

    def state_dict(self) -> dict:
        """Picklable snapshot. No model to train, so only the replay buffer."""
        return {"buffer": self.buffer}

    def load_state_dict(self, state: dict) -> None:
        self.buffer = state.get("buffer", self.buffer)

def make_predictor(kind: str, spec: ObjectiveSpec, critic, max_depth: int, buffer=None, **kw):
    if kind == "heuristic":
        return HeuristicPredictor(spec, critic, max_depth, buffer=buffer)
    if kind == "gnn":
        from .gnn_predictor import GNNValuePredictor

        return GNNValuePredictor(spec, critic, max_depth, buffer=buffer, **kw)
    raise ValueError(f"unknown predictor kind {kind!r}")
