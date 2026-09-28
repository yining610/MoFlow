from dataclasses import dataclass

import numpy as np

from ..ccs.ccs import CCS
from ..workflow.graph import WorkflowGraph
from .trace import ConstructionTrace


@dataclass
class ArchiveEntry:
    vector: np.ndarray  # maximize-form terminal reward
    graph: WorkflowGraph
    trace: ConstructionTrace


class Archive:
    """A non-dominated set of discovered terminal workflows.

    - D: number of objective axes.
    - spec: optional ObjectiveSpec enabling preference-correct retrieval. When
      present, w is applied to the NORMALIZED vector (spec.scalarize).
    """

    def __init__(self, D: int, spec=None):
        self.D = D
        self.spec = spec
        self._entries: list[ArchiveEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[ArchiveEntry]:
        return list(self._entries)

    def add(self, entry: ArchiveEntry) -> bool:
        """Insert the entry if non-dominated, drop any entries it dominates, and return True if it was kept.

        - entry: the candidate ArchiveEntry (vector + graph + trace) to insert.
        """
        from ..ccs import geometry as geo

        v = entry.vector
        # reject if dominated by an existing entry
        for e in self._entries:
            if geo.dominates(e.vector, v) or (
                geo.weakly_dominates(e.vector, v)
                and np.allclose(e.vector, v)
                and e.graph.signature() == entry.graph.signature()
            ):
                return False
        # drop existing entries dominated by the newcomer
        self._entries = [e for e in self._entries if not geo.dominates(v, e.vector)]
        self._entries.append(entry)
        return True

    def ccs(self) -> CCS:
        if not self._entries:
            return CCS.empty(self.D)
        pts = np.vstack([e.vector for e in self._entries])
        labels = list(range(len(self._entries)))
        return CCS(points=pts, labels=labels).pruned()

    def retrieve(self, w: np.ndarray) -> ArchiveEntry | None:
        """Return the optimal policy for the given preference weights, or None if the archive is empty.
        """
        if not self._entries:
            return None
        if self.spec is not None:
            scores = np.array([self.spec.scalarize(e.vector, w) for e in self._entries])
        else:
            scores = np.array([float(np.dot(w, e.vector)) for e in self._entries])
        return self._entries[int(np.argmax(scores))]

    def hypervolume(self, ref: np.ndarray) -> float:
        return self.ccs().hypervolume(ref)

    def normalized_hypervolume(self) -> float:
        """Hypervolume of the front in normalized units, measured from the origin.
        """
        if self.spec is None:
            raise ValueError("normalized_hypervolume requires an ObjectiveSpec")
        norm = CCS(points=self.spec.normalize(self.ccs().points))
        return norm.hypervolume(np.zeros(self.D))