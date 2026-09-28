from dataclasses import dataclass, field

import numpy as np

from ..ccs.ccs import CCS
from ..mdp.state import State
from .selection import ActionSelector
from .trace import ConstructionTrace


class DecisionNode:
    def __init__(self, state: State, D: int, selector: ActionSelector, trace: ConstructionTrace):
        self.state = state
        self.D = D
        self.V = CCS.empty(D)  # V-hat(s)
        self.selector = selector  # action selection policy
        self.children: dict[tuple, "ChanceNode"] = {}  # action_key -> ChanceNode
        self.trace = trace  # construction trace from root to this state
        self.prior: tuple[np.ndarray, float] = None  # (estimate, uncertainty)
        self._leaf_samples: list[np.ndarray] = None

    def n_executed_descendants(self) -> int:
        """Total real reward samples executed under this state."""
        return sum(c.n_samples for c in self.children.values())


@dataclass
class Successor:
    """One realized successor state reached from a chance node (a branch of P(s'|s,a)).

    - child: the successor DecisionNode.
    - visits: how many times this realization was sampled (the empirical transition mass).
    - sample_rewards: the R execution draws when `child` is terminal; averaged to E[R]
      at backup so evaluation noise of a single realization is not exploited.
    """

    child: "DecisionNode" = None
    visits: int = 0
    sample_rewards: list = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        """Number of execution reward draws recorded for this successor."""
        return len(self.sample_rewards)

    def add_samples(self, vecs) -> None:
        """Append reward vectors to this successor's execution draws.

        - vecs: iterable of reward vectors, each copied as a float ndarray.
        """
        for v in vecs:
            self.sample_rewards.append(np.asarray(v, dtype=float).copy())


class ChanceNode:
    def __init__(self, parent_key: tuple, edit, rationale: str, D: int):
        self.action_key = parent_key
        self.edit = edit  # the nominal decision (realized into concrete successors)
        self.rationale = rationale
        self.D = D
        self.Q = CCS.empty(D)  # Q-hat(s,a)
        self.successors: dict[str, Successor] = {}
        self.visits = 0  # total realizations sampled (sum of successor visits)

    def observe(self, succ_key: str, child: "DecisionNode") -> Successor:
        """Record that this realization landed in successor `succ_key`.

        - succ_key: state key of the realized successor (buckets stochastic branches).
        - child: the DecisionNode for that successor state.
        """
        s = self.successors.get(succ_key)
        if s is None:
            s = Successor()
            self.successors[succ_key] = s
        s.child = child
        s.visits += 1
        self.visits += 1
        return s

    def prob(self, succ_key: str) -> float:
        """Empirical P(s'|s,a) for the given successor.

        - succ_key: state key of the successor whose transition mass is returned.
        """
        if self.visits <= 0:
            return 0.0
        return self.successors[succ_key].visits / self.visits

    @property
    def n_samples(self) -> int:
        """Real execution draws observed across all successor realizations."""
        return sum(s.n_samples for s in self.successors.values())
