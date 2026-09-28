from dataclasses import dataclass, field

import numpy as np


@dataclass
class TraceStep:
    label: str  # e.g. "add Ensemble(k=3) as 'voters'"
    rationale: str  # NL reason attached at proposal time
    delta: np.ndarray = None  # per-decision value-vector change (maximize-form)


@dataclass
class ConstructionTrace:
    steps: list[TraceStep] = field(default_factory=list)

    def add(self, label: str, rationale: str, delta: np.ndarray = None) -> None:
        self.steps.append(TraceStep(label=label, rationale=rationale, delta=delta))

    def extended(self, label: str, rationale: str, delta=None) -> "ConstructionTrace":
        t = ConstructionTrace(steps=list(self.steps))
        t.add(label, rationale, delta)
        return t

    def render(self, spec=None) -> str:
        """Render the construction trace as a human-readable multi-line string.

        - spec: optional ObjectiveSpec; when given, each step's value-vector delta
          is appended in display form.
        """
        out = []
        for i, s in enumerate(self.steps, 1):
            line = f"  {i}. {s.label}  --  {s.rationale}"
            if s.delta is not None and spec is not None:
                d = spec.display(s.delta)
                line += "  [" + ", ".join(f"{k}{'+' if v>=0 else ''}{v:.2g}" for k, v in d.items()) + "]"
            out.append(line)
        return "\n".join(out) if out else "  (empty workflow)"