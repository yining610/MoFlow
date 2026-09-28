
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..objectives import ObjectiveSpec
from ..workflow import operators as ops
from ..workflow.graph import WorkflowGraph
from .role_encoder import canonicalize_role

WARMUP_REFIT = 8

# role vocabulary: reserved ids 0 = "default", 1 = "<unk>"; learned roles follow.
ROLE_DEFAULT_ID = 0
ROLE_UNK_ID = 1
ROLE_VOCAB_CAP = 64  # fixed nn.Embedding size; roles beyond cap fold into <unk>

def featurize(graph: WorkflowGraph) -> np.ndarray:
    """Return a fixed-length structural feature vector for a (partial) workflow."""
    op_names = ops.names()
    counts = {name: 0 for name in op_names}
    total_width = 0
    parallel_width = 0
    for n in graph.nodes:
        counts[n.operator] = counts.get(n.operator, 0) + 1
        total_width += n.width
        if ops.get(n.operator).parallel:
            parallel_width += n.width
    tokens = sum(ops.get(n.operator).tokens(n.width) for n in graph.nodes)
    calls = sum(ops.get(n.operator).llm_calls(n.width) for n in graph.nodes)
    feats = [counts[name] for name in op_names]
    feats += [
        graph.depth,
        total_width,
        parallel_width,
        np.log1p(tokens),
        np.log1p(calls),
        1.0 if graph.terminated else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)


def feature_dim() -> int:
    return len(ops.names()) + 6

@dataclass
class ReplayBuffer:
    """FIFO-capped store of (graph, predicted-axis target) training pairs.
    """

    spec: ObjectiveSpec
    capacity: int = 5000
    test_frac: float = 0.30
    _train_X: deque = field(default_factory=deque)
    _train_Y: deque = field(default_factory=deque)
    _train_G: deque = field(default_factory=deque)
    _test_X: deque = field(default_factory=deque)
    _test_Y: deque = field(default_factory=deque)
    _test_G: deque = field(default_factory=deque)
    _since_refit: int = 0

    _refit_total: int = 0
    _roles: dict = field(default_factory=lambda: {"default": ROLE_DEFAULT_ID})

    def __post_init__(self) -> None:
        test_cap = max(1, round(self.capacity * self.test_frac))
        train_cap = max(1, self.capacity - test_cap)
        self._train_X = deque(self._train_X, maxlen=train_cap)
        self._train_Y = deque(self._train_Y, maxlen=train_cap)
        self._train_G = deque(self._train_G, maxlen=train_cap)
        self._test_X = deque(self._test_X, maxlen=test_cap)
        self._test_Y = deque(self._test_Y, maxlen=test_cap)
        self._test_G = deque(self._test_G, maxlen=test_cap)

    def __setstate__(self, state: dict) -> None:
        """Restore a pickle, migrating checkpoints that predate the train/test split.
        """
        self.__dict__.update(state)
        self.__dict__.setdefault("test_frac", 0.30)
        self.__dict__.setdefault("_since_refit", 0)
        self.__dict__.setdefault("_refit_total", 0)
        self.__dict__.setdefault("_roles", {"default": ROLE_DEFAULT_ID})
        if hasattr(self, "_train_X"):
            return  # current format: nothing to migrate
        old_X = list(self.__dict__.pop("_X", []) or [])
        old_Y = list(self.__dict__.pop("_Y", []) or [])
        old_G = list(self.__dict__.pop("_G", []) or [])
        for attr in ("_train_X", "_train_Y", "_train_G", "_test_X", "_test_Y", "_test_G"):
            setattr(self, attr, deque())
        self.__post_init__()  # wrap the empty deques at the right partition capacity
        for x, y, g in zip(old_X, old_Y, old_G):
            if self._is_test(g):
                self._test_X.append(x); self._test_Y.append(y); self._test_G.append(g)
            else:
                self._train_X.append(x); self._train_Y.append(y); self._train_G.append(g)

    @property
    def target_dim(self) -> int:
        """Width of each learning target, i.e. the number of predicted axes in the spec."""
        return len(self.spec.predicted_names)

    def _is_test(self, graph: WorkflowGraph) -> bool:
        """Route a graph to the test partition by hashing its content signature.

        Deterministic and stable across runs/resume: the same graph always lands on the
        same side, so no signature ever appears in both train and test partitions.
        """
        bucket = int(graph.signature(), 16) % 100
        return bucket < round(self.test_frac * 100)

    def add(self, graph: WorkflowGraph, target_vector: np.ndarray) -> None:
        raw = self.spec.display(target_vector)
        # learning target = the chosen set's PREDICTED axes, in vector order.
        y = np.array([raw[n] for n in self.spec.predicted_names], dtype=np.float32)
        if self._is_test(graph):
            self._test_X.append(featurize(graph))
            self._test_Y.append(y)
            self._test_G.append(graph)
        else:
            self._train_X.append(featurize(graph))
            self._train_Y.append(y)
            self._train_G.append(graph)
        for node in graph.nodes:  # register roles into the shared vocabulary
            self._intern_role(node.role)
        self._since_refit += 1

    def _intern_role(self, role: str) -> int:
        """Return the id for a role string, assigning a new one (or <unk> past the cap)."""
        role = canonicalize_role(role)
        if role in self._roles:
            return self._roles[role]
        if len(self._roles) + 1 >= ROLE_VOCAB_CAP:  # +1 reserves <unk>
            return ROLE_UNK_ID
        rid = max(self._roles.values(), default=ROLE_UNK_ID) + 1
        self._roles[role] = rid
        return rid

    def role_to_id(self, role: str) -> int:
        return self._roles.get(canonicalize_role(role), ROLE_UNK_ID)

    @property
    def role_vocab_size(self) -> int:
        return ROLE_VOCAB_CAP

    def __len__(self) -> int:
        return len(self._train_X)

    @property
    def n_train(self) -> int:
        return len(self._train_X)

    @property
    def n_test(self) -> int:
        return len(self._test_X)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the train partition as stacked (X, Y) arrays."""
        if not self._train_X:
            d = feature_dim()
            return np.empty((0, d), np.float32), np.empty((0, 3), np.float32)
        return np.stack(list(self._train_X)), np.stack(list(self._train_Y))

    def graphs(self) -> list[tuple[WorkflowGraph, np.ndarray]]:
        """Return train (graph, predicted-axis target) pairs for GNN training."""
        return list(zip(self._train_G, self._train_Y))

    def test_graphs(self) -> list[tuple[WorkflowGraph, np.ndarray]]:
        """Return held-out test (graph, predicted-axis target) pairs for GNN validation."""
        return list(zip(self._test_G, self._test_Y))

    def due(self, every: int) -> bool:
        return self._since_refit >= every and len(self._train_X) >= max(8, every)

    def mark_refit(self) -> None:
        self._since_refit = 0
        self._refit_total += 1

    def request_refit(self) -> None:
        """Mark every current train pair as not yet learned so `due()` fires on the next check.
        """
        self._since_refit = max(self._since_refit, self.n_train)
