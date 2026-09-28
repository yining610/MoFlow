from dataclasses import dataclass
from typing import Any

@dataclass
class ExecMeta:
    """Resource accounting accumulated during one workflow execution."""

    tokens: int = 0
    sequential_calls: int = 0

    def add(self, other: "ExecMeta") -> "ExecMeta":
        return ExecMeta(
            tokens=self.tokens + other.tokens,
            sequential_calls=self.sequential_calls + other.sequential_calls,
        )


@dataclass
class EditProposal:
    """One proposed atomic edit with its NL rationale and an optional prior score."""

    edit: Any
    rationale: str = ""
    prior: float = 0.0  # optional proposer confidence (used for RAVE-ish ordering)
