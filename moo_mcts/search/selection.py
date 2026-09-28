"""Pluggable action selection policy (the per-decision-node tree policy).
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..ccs import geometry as geo
from ..ccs.ccs import CCS


class ActionSelector(ABC):

    @abstractmethod
    def ensure_actions(self, proposals: list) -> None:
        """Register any newly-proposed action not seen at this node."""

    @abstractmethod
    def select(self, w: np.ndarray, action_keys: set, node=None):
        """Return the chosen Selection for context `w`, or None to stop the descent.
        """

    @abstractmethod
    def record(self, selection, w: np.ndarray, value_vectors: list) -> None:
        """Update the selected action's statistics with this trial's leaf value.
        """


@dataclass
class Selection:

    action_key: tuple
    edit: object = None
    rationale: str = ""
    prior: float = 0.0


class _QSetReader(ActionSelector):
    """score each legal action by a scalarization of its backed-up Pareto set Q-hat(s,a), 
    plus the UCB1 exploration bonus `c * sqrt(log N(s) / N(s,a))`.
    """

    def __init__(self, spec, c: float = 1.0):
        self.spec = spec
        self.D = spec.D
        self.c = c
        self.actions: dict[tuple, Selection] = {}

    def ensure_actions(self, proposals: list) -> None:
        for p in proposals:
            k = p.edit.key()
            if k not in self.actions:
                self.actions[k] = Selection(
                    action_key=k, edit=p.edit, rationale=p.rationale,
                    prior=float(getattr(p, "prior", 0.0)),
                )

    def record(self, selection, w: np.ndarray, value_vectors: list) -> None:
        # value comes from the CHVI backups on the chance nodes, not from here.
        return

    def _prepare(self, legal, children, w):
        return None

    def _normalized_points(self, q_ccs):
        """Rows of Q-hat normalized to [0,1]^D."""
        if q_ccs is None or q_ccs.is_empty():
            return np.empty((0, self.D))
        return np.stack([self.spec.normalize(p) for p in q_ccs.points])

    def select(self, w: np.ndarray, action_keys: set, node=None):
        """Shared selection logic for all CHMCTS variants: pick one action for the given preference
        """
        legal = [k for k in action_keys if k in self.actions]
        if not legal:
            return None
        children = getattr(node, "children", {}) if node is not None else {}

        # forced exploration: play each arm once if any legal action with no chance node yet (N(s,a)=0).
        unvisited = [
            k for k in legal if k not in children or children[k].visits == 0
        ]
        if unvisited:
            return max((self.actions[k] for k in unvisited), key=lambda s: s.prior)

        n_parent = sum(children[k].visits for k in legal)  # N(s) = sum_a N(s,a)
        ctx = self._prepare(legal, children, w)
        best_k, best_score = legal[0], float("-inf")
        for k in legal:
            ch = children[k]
            exploit = self._zeta(ch, w, n_parent, ctx)
            bonus = self.c * math.sqrt(math.log(max(n_parent, 2)) / ch.visits)
            score = exploit + bonus
            if score > best_score:
                best_score, best_k = score, k
        return self.actions[best_k]


class HypervolumeSelector(_QSetReader):
    """Hypervolume CHMCTS (context-free).
    """

    def _zeta(self, chance, w, n_parent, ctx):
        pts = self._normalized_points(chance.Q)
        if pts.shape[0] == 0:
            return 0.0
        hv = CCS.of(pts).hypervolume(np.zeros(self.D))
        return hv / max(chance.visits, 1)  # average HV per visit, N(s,a)


class ChebyshevSelector(_QSetReader):
    """Chebychev CHMCTS.

    Context-aware: scores each action by the (negated) weighted Chebychev distance from
    its closest value-set point to the utopia z.
    """

    def _prepare(self, legal, children, w):
        # z_i = max over all legal actions' normalized Q-hat points (estimate of max_pi V_i^pi)
        rows = [
            self._normalized_points(children[k].Q)
            for k in legal
            if k in children
        ]
        rows = [r for r in rows if r.shape[0]]
        if not rows:
            return np.ones(self.D)  # normalized ideal until a front exists
        return np.max(np.vstack(rows), axis=0)

    def _zeta(self, chance, w, n_parent, ctx):
        pts = self._normalized_points(chance.Q)
        if pts.shape[0] == 0:
            return -float(np.max(w))  # worst utility for a (degenerate) empty set
        z = ctx
        w = np.asarray(w, dtype=float)
        dist = float(np.min(np.max(w * np.abs(pts - z), axis=1)))
        return -dist


class ParetoSelector(_QSetReader):
    """Pareto CHMCTS: the ParetoUCB selection of Chen & Liu (Pareto MCTS, 2021), run
    over the union of the children's backed-up value sets U_a Q-hat(s,a).
    """

    def __init__(self, spec, seed: int = 0):
        super().__init__(spec, c=1.0)
        self.rng = np.random.default_rng(seed)

    def select(self, w: np.ndarray, action_keys: set, node=None):
        """Pick one action via ParetoUCB over the union of the backed-up value sets."""
        legal = [k for k in action_keys if k in self.actions]
        if not legal:
            return None
        children = getattr(node, "children", {}) if node is not None else {}

        unvisited = [k for k in legal if k not in children or children[k].visits == 0]
        if unvisited:
            return max((self.actions[k] for k in unvisited), key=lambda s: s.prior)

        n_parent = sum(children[k].visits for k in legal)                # n = sum_k n_k
        bonus_num = 4.0 * math.log(max(n_parent, 2)) + math.log(self.D)  # 4 ln n + ln D

        pooled, tags = [], []  # pooled U(k) vectors
        for k in legal:
            ch = children[k]
            c_a = math.sqrt(bonus_num / (2.0 * ch.visits))  # sqrt((4 ln n + ln D)/(2 n_k))
            pts = self._normalized_points(ch.Q)
            ipts = pts + c_a if pts.shape[0] else pts       # U(k): inflate every Q-hat point
            for v in ipts:
                pooled.append(v)
                tags.append(k)
        if not pooled:
            return self.actions[legal[0]]

        # find the Pareto front of the set of value vectors over all children
        front_idx = geo.pareto_front(np.stack(pooled))
        front_actions = []
        for i in front_idx:
            k = tags[int(i)]
            if k not in front_actions:
                front_actions.append(k)

        choice = front_actions[int(self.rng.integers(len(front_actions)))]
        return self.actions[choice]


def make_selector(kind: str, spec, cfg) -> ActionSelector:
    kind = (kind or "czt").lower()
    if kind == "czt":
        from .czt import CZTSelector  # local import avoids a czt<->selection cycle
        return CZTSelector(spec=spec, czt_c=cfg.czt_c, U=1.0)
    if kind == "pareto":
        return ParetoSelector(spec=spec, seed=getattr(cfg, "seed", 0))
    if kind == "hypervolume":
        return HypervolumeSelector(spec=spec, c=getattr(cfg, "ucb_c", 1.0))
    if kind == "chebyshev":
        return ChebyshevSelector(spec=spec, c=getattr(cfg, "ucb_c", 1.0))
    raise ValueError(
        f"unknown selector kind: {kind!r} "
        "(expected 'czt', 'pareto', 'hypervolume' or 'chebyshev')"
    )