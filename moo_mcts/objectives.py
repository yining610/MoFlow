from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Sense(Enum):
    """Whether an objective is maximized or minimized in raw units.

    Internally we always maximize; MIN objectives are negated on the way in.
    """

    MAX = "max"
    MIN = "min"


class AxisKind(Enum):
    """How an axis's value is obtained, which decides whether a predictor must learn it.

      ANALYTIC : an exact function of the workflow STRUCTURE (cost, latency).
                 Never learned.
      PREDICTED: estimated from execution or a value head (accuracy, robustness,
                 consistency).
    """

    ANALYTIC = "analytic"
    PREDICTED = "predicted"


@dataclass(frozen=True)
class Objective:
    """A single objective axis.

    Attributes:
        name: short identifier, e.g. "accuracy", "cost", "latency".
        sense: MAX or MIN, in raw (human) units.
        unit: display unit ("frac", "tokens", "calls", "1-var").
        ref: maximize-form reference-point coord for hypervolume; a lower bound
             dominated by every achievable vector.
        span: maximize-form spread, used to normalize the axis to ~[0,1] before
             preference weights.
        kind: ANALYTIC (structural, never learned) or PREDICTED (learned).
    """

    name: str
    sense: Sense
    unit: str = ""
    ref: float = 0.0
    span: float = 1.0
    kind: "AxisKind" = AxisKind.PREDICTED

    def normalize(self, internal: float) -> float:
        """Map a maximize-form coordinate to ~[0,1] via (internal - ref) / span."""
        return (internal - self.ref) / self.span if self.span else 0.0

    def to_internal(self, raw: float) -> float:
        """Map a raw measurement to maximize-form, negating MIN objectives."""
        return raw if self.sense is Sense.MAX else -raw

    def to_raw(self, internal: float) -> float:
        """Invert `to_internal` for display."""
        return internal if self.sense is Sense.MAX else -internal


@dataclass(frozen=True)
class ObjectiveSpec:

    objectives: tuple[Objective, ...]

    def __post_init__(self) -> None:
        names = [o.name for o in self.objectives]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate objective names: {names}")

    @property
    def D(self) -> int:
        return len(self.objectives)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.objectives)

    def has(self, name: str) -> bool:
        """Whether an axis with this name is present in the chosen objective set.

        - name: objective name to look for.
        """
        return any(o.name == name for o in self.objectives)

    @property
    def predicted_names(self) -> tuple[str, ...]:
        """PREDICTED axes present in this spec, in vector order."""
        return tuple(o.name for o in self.objectives if o.kind is AxisKind.PREDICTED)

    @property
    def analytic_names(self) -> tuple[str, ...]:
        """ANALYTIC axes present in this spec, in vector order."""
        return tuple(o.name for o in self.objectives if o.kind is AxisKind.ANALYTIC)

    @property
    def predicted_indices(self) -> tuple[int, ...]:
        """Positions of the PREDICTED axes in the full vector, so a learned head's
        outputs can be scattered back into the right slots.
        """
        return tuple(i for i, o in enumerate(self.objectives) if o.kind is AxisKind.PREDICTED)

    @property
    def ref_point(self) -> np.ndarray:
        """Reference point in maximize-form, for hypervolume computations."""
        return np.array([o.ref for o in self.objectives], dtype=float)

    def vector(self, **raw: float) -> np.ndarray:
        """Build a maximize-form vector from named raw measurements, requiring
        exactly the axes in this spec (raising on any missing or unknown name).
        """
        missing = set(self.names) - set(raw)
        if missing:
            raise ValueError(f"missing objective values: {sorted(missing)}")
        extra = set(raw) - set(self.names)
        if extra:
            raise ValueError(f"unknown objective values: {sorted(extra)}")
        return np.array(
            [o.to_internal(float(raw[o.name])) for o in self.objectives], dtype=float
        )

    def assemble(self, **available: float) -> np.ndarray:
        """Build a maximize-form vector from named raw values, using only the axes
        in this spec and silently ignoring the rest (raises if a required axis is
        missing).

        - available: raw values keyed by objective name; extras beyond this spec
          are ignored.
        """
        missing = set(self.names) - set(available)
        if missing:
            raise ValueError(f"assemble missing required axes: {sorted(missing)}")
        return np.array(
            [o.to_internal(float(available[o.name])) for o in self.objectives], dtype=float
        )

    def display(self, v: np.ndarray) -> dict[str, float]:
        """Map a maximize-form vector back to a dict of raw, human-facing values.

        - v: maximize-form vector of shape (D,).
        """
        v = np.asarray(v, dtype=float)
        if v.shape != (self.D,):
            raise ValueError(f"expected shape ({self.D},), got {v.shape}")
        return {o.name: o.to_raw(float(v[i])) for i, o in enumerate(self.objectives)}

    @property
    def spans(self) -> np.ndarray:
        return np.array([o.span for o in self.objectives], dtype=float)

    def normalize(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        return (v - self.ref_point) / self.spans

    def scalarize(self, v: np.ndarray, w: np.ndarray) -> float:

        return float(np.dot(np.asarray(w, dtype=float), self.normalize(v)))


CANONICAL_ORDER: tuple[str, ...] = (
    "accuracy", "cost", "latency", "robustness", "consistency",
)


def _catalogue(cost_ref_tokens: float, latency_ref_calls: float) -> dict[str, Objective]:
    return {
        "accuracy": Objective("accuracy", Sense.MAX, "frac", ref=0.0, span=1.0,
                              kind=AxisKind.PREDICTED),
        "cost": Objective("cost", Sense.MIN, "tokens", ref=-cost_ref_tokens,
                          span=cost_ref_tokens, kind=AxisKind.ANALYTIC),
        "latency": Objective("latency", Sense.MIN, "calls", ref=-latency_ref_calls,
                            span=latency_ref_calls, kind=AxisKind.ANALYTIC),
        "robustness": Objective("robustness", Sense.MAX, "1-var", ref=0.0, span=1.0,
                               kind=AxisKind.PREDICTED),
        "consistency": Objective("consistency", Sense.MAX, "1-var", ref=0.0, span=1.0,
                                kind=AxisKind.PREDICTED),
    }


def available_objectives() -> tuple[str, ...]:
    """All selectable objective names, in canonical order."""
    return CANONICAL_ORDER

def default_spec(
    objectives: Sequence[str] = None,
    cost_ref_tokens: float = 20000.0,
    latency_ref_calls: float = 8.0,
) -> ObjectiveSpec:
    """Build an ObjectiveSpec over a chosen subset of the five objectives.
    """
    cat = _catalogue(cost_ref_tokens, latency_ref_calls)
    if objectives is None:
        chosen = list(CANONICAL_ORDER)
    else:
        names = list(objectives)
        unknown = [n for n in names if n not in cat]
        if unknown:
            raise ValueError(
                f"unknown objective(s) {unknown}; available: {list(CANONICAL_ORDER)}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate objectives: {names}")
        if len(names) < 2:
            raise ValueError(
                f"need at least 2 objectives for a multi-objective problem, got {names}"
            )
        chosen_set = set(names)
        chosen = [n for n in CANONICAL_ORDER if n in chosen_set]  # canonical order
    return ObjectiveSpec(objectives=tuple(cat[n] for n in chosen))
