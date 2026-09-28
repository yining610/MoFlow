"""Convex Coverage Set (CCS): the value object stored at every search node.

Decision nodes store V-hat(s) ~ CCS, chance nodes store Q-hat(s,a). Backups:
  - chance:   Q-hat(s,a) = E[ R + V-hat(s') ]   (Minkowski sum of point-sets)
  - decision: V-hat(s)   = prune( union_a Q-hat(s,a) )

This class implements the set algebra those backups need. By default it maintains
the non-dominated set (Pareto front); `to_ccs()` reduces it to CCS vertices.
"""


from dataclasses import dataclass
from typing import Iterable

import numpy as np

from . import geometry as geo


@dataclass
class CCS:
    """A Pareto-non-dominated set of maximize-form value vectors.

    - points: an (N, D) float array, one value vector per row.
    - labels: optional per-point payload (e.g. an archive id), one entry per row, pruned alongside `points`.
    """

    points: np.ndarray  # shape (N, D)
    labels: list = None

    @classmethod
    def empty(cls, D: int) -> "CCS":
        return cls(points=np.empty((0, D), dtype=float), labels=[])

    @classmethod
    def of(cls, vectors: Iterable, labels: list = None) -> "CCS":
        pts = np.atleast_2d(np.asarray(list(vectors), dtype=float))
        if pts.size == 0:
            raise ValueError("CCS.of needs at least one vector or use CCS.empty(D)")
        c = cls(points=pts, labels=list(labels) if labels is not None else None)
        return c.pruned()

    @property
    def D(self) -> int:
        return self.points.shape[1]

    def __len__(self) -> int:
        return self.points.shape[0]

    def is_empty(self) -> bool:
        return self.points.shape[0] == 0

    def copy(self) -> "CCS":
        return CCS(
            points=self.points.copy(),
            labels=list(self.labels) if self.labels is not None else None,
        )

    def pruned(self) -> "CCS":
        """Return a copy with all Pareto-dominated points removed."""
        if self.is_empty():
            return self.copy()
        idx = geo.pareto_front(self.points)
        labels = None
        if self.labels is not None:
            labels = [self.labels[i] for i in idx]
        return CCS(points=self.points[idx], labels=labels)

    def to_ccs(self) -> "CCS":
        """Return a copy reduced from the Pareto front to its convex-coverage-set vertices."""
        if len(self) <= 1:
            return self.copy()
        idx = geo.convex_hull_vertices(self.points)
        labels = None
        if self.labels is not None:
            labels = [self.labels[i] for i in idx]
        return CCS(points=self.points[idx], labels=labels)

    def union(self, other: "CCS") -> "CCS":
        """Return the pruned union of this set with `other` (the decision-node backup over children).

        - other: the CCS to merge with this one.
        """
        if self.is_empty():
            return other.pruned()
        if other.is_empty():
            return self.pruned()
        pts = np.vstack([self.points, other.points])
        labels = None
        if self.labels is not None and other.labels is not None:
            labels = list(self.labels) + list(other.labels)
        return CCS(points=pts, labels=labels).pruned()

    @staticmethod
    def union_all(sets: Iterable["CCS"], D: int) -> "CCS":
        """Return the pruned union of many CCS sets.

        - sets: the CCS sets to union together.
        - D: objective dimension, used to build the empty starting set.
        """
        acc = CCS.empty(D)
        for s in sets:
            acc = acc.union(s)
        return acc

    def shift(self, r: np.ndarray) -> "CCS":
        """Translate every point by the reward vector `r` (CHMCTS Eq. 9).

        - r: reward vector added to each value vector.
        """
        if self.is_empty():
            return self.copy()
        return CCS(
            points=self.points + np.asarray(r, dtype=float),
            labels=list(self.labels) if self.labels is not None else None,
        )

    def minkowski_sum(self, other: "CCS") -> "CCS":
        if self.is_empty() or other.is_empty():
            return CCS.empty(self.D if not self.is_empty() else other.D)
        sums = (self.points[:, None, :] + other.points[None, :, :]).reshape(-1, self.D)
        return CCS(points=sums, labels=None).pruned()

    @staticmethod
    def expectation(weighted: list[tuple[float, "CCS"]], D: int) -> "CCS":
        weighted = [(p, c) for (p, c) in weighted if p > 0 and not c.is_empty()]
        if not weighted:
            return CCS.empty(D)
        # Start from a zero set, fold in p_i * CCS_i via Minkowski sum.
        acc = CCS.of([np.zeros(D)])
        for p, c in weighted:
            scaled = CCS(points=c.points * float(p), labels=None)
            acc = acc.minkowski_sum(scaled)
        return acc

    def scalarized(self, w: np.ndarray) -> np.ndarray:
        if self.is_empty():
            return np.empty(0)
        return geo.scalarize(self.points, w)

    def best_value(self, w: np.ndarray) -> float:
        """Return max_v (w . v) over the set, or -inf if empty.

        - w: weight vector for the linear scalarization.
        """
        if self.is_empty():
            return float("-inf")
        return float(self.scalarized(w).max())

    def argmax(self, w: np.ndarray) -> tuple[np.ndarray, object | None]:
        """Return the (best vector, its label) pair maximizing w . v over the set (S4.8.1).

        - w: weight vector for the linear scalarization.
        """
        if self.is_empty():
            raise ValueError("argmax on empty CCS")
        s = self.scalarized(w)
        i = int(np.argmax(s))
        label = self.labels[i] if self.labels is not None else None
        return self.points[i].copy(), label

    def hypervolume(self, ref: np.ndarray) -> float:
        """Return the exact dominated hypervolume of the set relative to reference point.
        """
        ref = np.asarray(ref, dtype=float)
        if self.is_empty():
            return 0.0
        pts = self.points[np.all(self.points >= ref, axis=1)]
        if pts.shape[0] == 0:
            return 0.0
        return float(_hv_sweep(pts, ref))

    def sparsity(self, ref: np.ndarray) -> float:
        """Return the PD-MORL sparsity: the mean squared gap between adjacent sorted front points.

        Lower means a denser front. Computed per-axis on the sorted coordinates
        and averaged over axes. Only defined for |CCS| >= 2 (returns 0.0 otherwise).

        - ref: reference point (unused here; kept for a consistent metric signature).
        """
        if len(self) < 2:
            return 0.0
        total = 0.0
        for d in range(self.D):
            col = np.sort(self.points[:, d])
            diffs = np.diff(col)
            total += float(np.mean(diffs**2)) if len(diffs) else 0.0
        return total / self.D


def _hv_sweep(pts: np.ndarray, ref: np.ndarray) -> float:
    """Compute dominated hypervolume by recursive dimension sweep (Fonseca et al. 2006), maximize-form.

    - pts: points that dominate `ref`, contributing to the volume.
    - ref: reference point the volume is measured from.
    """
    m = ref.shape[0]
    if m == 1:
        return float(pts[:, 0].max() - ref[0])
    pts = pts[np.argsort(pts[:, 0])]  # ascending on axis 0
    hv = 0.0
    base = ref[0]
    n = pts.shape[0]
    i = 0
    while i < n:
        x = pts[i, 0]
        width = x - base
        if width > 0.0:
            # every point in pts[i:] reaches at least x on axis 0 -> spans the slab
            hv += width * _hv_sweep(pts[i:, 1:], ref[1:])
            base = x
        # advance past all points tied at this axis-0 value
        j = i + 1
        while j < n and pts[j, 0] == x:
            j += 1
        i = j
    return hv