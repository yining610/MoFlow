import math
from dataclasses import dataclass, field

import numpy as np

from .selection import ActionSelector


@dataclass
class Ball:
    action_key: tuple   # identity of the atomic edit (AtomicEdit.key())
    center: np.ndarray  # a preference vector
    radius: float       # how far in preference it covers
    n: int = 0          # times selected
    nu: float = 0.0     # mean scalarized reward
    # keep the concrete edit + rationale for reconstruction/expansion
    edit: object = None
    rationale: str = ""

    def update(self, scalar_reward: float) -> None:
        self.n += 1
        self.nu += (scalar_reward - self.nu) / self.n


class CZTSelector(ActionSelector):
    """Contextual Zooming for Trees
    """

    def __init__(self, spec=None, czt_c: float = 4.0, U: float = 1.0):
        self.spec = spec  # ObjectiveSpec, used to scalarize the leaf value in record()
        self.c = czt_c
        self.U = U  # cross-action distance / max scalar reward bound
        self.balls: list[Ball] = []
        self.t = 0  # total selections at this node (the "round" counter)

    def _balls_for(self, action_key: tuple) -> list[Ball]:
        return [b for b in self.balls if b.action_key == action_key]

    def ensure_actions(self, proposals: list) -> None:
        """Create an initial radius-U ball for any newly-proposed action not yet seen.

        - proposals: candidate actions from the proposer, each carrying an `edit`
          (with a `.key()`) and a `rationale`.
        """
        existing = {b.action_key for b in self.balls}
        for p in proposals:
            k = p.edit.key()
            if k not in existing:
                # center is set lazily on the first real context (centroid proxy)
                self.balls.append(
                    Ball(
                        action_key=k,
                        center=None,  # every new proposed action is eligible for selection at least once
                        radius=self.U,
                        edit=p.edit,
                        rationale=p.rationale,
                    )
                )

    def _covers(self, b: Ball, w: np.ndarray) -> bool:
        """Whether ball `b` covers context `w` (an un-anchored ball covers any w)."""
        if b.center is None:
            return True
        return float(np.max(np.abs(w - b.center))) <= b.radius + 1e-9

    def _relevant(self, w: np.ndarray, action_keys: set) -> list[Ball]:
        """Balls relevant to w
        """
        covering = [
            b for b in self.balls if b.action_key in action_keys and self._covers(b, w)
        ]
        rel = []
        for b in covering:
            if any(
                o.action_key == b.action_key and o.radius < b.radius - 1e-12
                for o in covering
            ):
                continue  # a finer same-arm ball covers w -> b is not in its own domain
            rel.append(b)
        return rel

    def _conf(self, b: Ball) -> float:
        return self.c * math.sqrt(math.log(max(self.t, 2)) / (1 + b.n))

    def _index(self, b: Ball, all_balls: list[Ball]) -> float:
        """Optimistic index I_k(B)
        """

        def i_pre(x: Ball) -> float:
            return x.nu + x.radius + self._conf(x)

        def dist(x: Ball, y: Ball) -> float:
            if x.action_key != y.action_key:
                return self.U
            if x.center is None or y.center is None:
                return 0.0
            return float(np.max(np.abs(x.center - y.center)))

        best = min(i_pre(bp) + dist(b, bp) for bp in all_balls)
        return b.radius + best

    def select(self, w: np.ndarray, action_keys: set, node=None):
        """Choose the relevant ball with the highest optimistic index, or None if none are relevant.

        - w: the current preference context.
        - action_keys: keys of the actions currently legal at this node.
        - node: unused (CZT is self-accumulating; kept for interface uniformity).
        """
        self.t += 1
        rel = self._relevant(w, action_keys)
        if not rel:
            return None
        # max-index selection; each index minimises over the full active set A_k (Eq. 25)
        scored = [(self._index(b, self.balls), b) for b in rel]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _leaf_scalar(self, value_vectors: list, w: np.ndarray) -> float:
        """Best normalized scalarized leaf value under `w`.
        """
        if not value_vectors or self.spec is None:
            return 0.0
        return max(
            float(np.clip(self.spec.scalarize(v, w), 0.0, 1.0)) for v in value_vectors
        )

    def record(self, selection: Ball, w: np.ndarray, value_vectors: list) -> None:
        """Scalarize the leaf value under `w`, fold it into the ball, and zoom.

        - selection: the ball that was selected this round.
        - w: the preference context for this trial.
        - value_vectors: the RAW leaf value vectors; scalarized internally.
        """
        b = selection
        scalar_reward = self._leaf_scalar(value_vectors, w)
        if b.center is None:
            b.center = np.array(w, dtype=float)  # anchor the initial ball
        b.update(scalar_reward)
        # activation rule: if confidence has shrunk to within the radius, zoom in
        if self._conf(b) <= b.radius and b.radius > 1e-3:
            child = Ball(
                action_key=b.action_key,
                center=np.array(w, dtype=float),
                radius=b.radius / 2.0,
                edit=b.edit,
                rationale=b.rationale,
            )
            self.balls.append(child)
