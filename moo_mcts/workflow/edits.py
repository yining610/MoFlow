"""Atomic typed edits: the MDP actions.

One atomic typed edit per decision: a single operator-add, role-set,
control-insertion, or terminate. Each edit:
  - is a frozen dataclass (hashable -> usable as an MCTS action key),
  - carries a natural-language `rationale` (the interpretable edge label, S3),
  - applies via `apply(edit, graph) -> new immutable graph`,
  - is checked for legality before being offered.
"""


from dataclasses import dataclass

from . import operators as ops
from .graph import OperatorNode, WorkflowGraph


@dataclass(frozen=True)
class AtomicEdit:
    """Base class for atomic edits.

    - rationale: natural-language reason for the edit; becomes the edge label.
    """

    rationale: str = ""

    def key(self) -> tuple:
        """Return a hashable identity for this edit (excludes the free-text rationale)."""
        raise NotImplementedError

    def apply(self, g: WorkflowGraph) -> WorkflowGraph:
        raise NotImplementedError

    def label(self) -> str:
        """Return a short human-facing label for traces."""
        raise NotImplementedError


@dataclass(frozen=True)
class AddOperator(AtomicEdit):
    operator: str = "Custom"
    width: int = 1
    role: str = "default"

    def key(self) -> tuple:
        return ("add_operator", self.operator, self.width, self.role)

    def apply(self, g: WorkflowGraph) -> WorkflowGraph:
        nid = g._next_id()
        node = OperatorNode(id=nid, operator=self.operator, width=self.width, role=self.role)
        return g.add_node(node)

    def label(self) -> str:
        w = f"(k={self.width})" if self.width > 1 else ""
        return f"add {self.operator}{w} as '{self.role}'"


@dataclass(frozen=True)
class SetRole(AtomicEdit):
    node_id: int = -1
    role: str = "default"

    def key(self) -> tuple:
        return ("set_role", self.node_id, self.role)

    def apply(self, g: WorkflowGraph) -> WorkflowGraph:
        return g.replace_node(self.node_id, role=self.role)

    def label(self) -> str:
        return f"set node {self.node_id} role -> '{self.role}'"


@dataclass(frozen=True)
class AddControl(AtomicEdit):
    control_kind: str = "branch"  # "branch" (parallel-width k) | "loop" (loop-until)
    width: int = 1  # parallel width for a branch / max iters for a loop
    operator: str = "Custom"  # the wrapped unit operator (any registered op)
    role: str = ""  # optional; defaults to control:<kind>

    def key(self) -> tuple:
        return ("add_control", self.control_kind, self.width, self.operator)

    def apply(self, g: WorkflowGraph) -> WorkflowGraph:
        nid = g._next_id()
        node = OperatorNode(
            id=nid,
            operator=self.operator,  # the wrapped unit the control repeats/parallelizes
            width=self.width,
            role=self.role or f"control:{self.control_kind}",
            is_control=True,
            control_kind=self.control_kind,
        )
        return g.add_node(node)

    def label(self) -> str:
        return f"add control[{self.control_kind}] over {self.operator} width={self.width}"


@dataclass(frozen=True)
class Terminate(AtomicEdit):
    def key(self) -> tuple:
        return ("terminate",)

    def apply(self, g: WorkflowGraph) -> WorkflowGraph:
        return g.terminate()

    def label(self) -> str:
        return "terminate (workflow complete)"


# Maximum ensemble / control width. Bounds untrusted LLM-proposed widths: an
# unbounded width inflates cost arbitrarily and exceeds the analytic layer's
# completion-bound assumption (valuation.analytic._cheapest_costliest max_width).
MAX_WIDTH = 5


def is_legal(edit: AtomicEdit, g: WorkflowGraph, max_depth: int) -> bool:
    """Return whether `edit` may be applied to workflow graph.

    - edit: the candidate atomic edit.
    - g: the current workflow graph the edit would apply to.
    - max_depth: search horizon H; beyond it only Terminate is legal.
    """
    if g.terminated:
        return False  # nothing follows a terminal state
    if isinstance(edit, Terminate):
        if g.is_empty():
            return False  # cannot terminate the empty template
        try:
            return ops.get(g.nodes[-1].operator).emits_answer
        except KeyError:
            return True
    if isinstance(edit, SetRole):
        try:
            g.node(edit.node_id)
        except KeyError:
            return False
        return True
    if isinstance(edit, (AddOperator, AddControl)):
        if g.depth >= max_depth:
            return False  # horizon H reached -> only terminate is legal
        if edit.width < 1 or edit.width > MAX_WIDTH:
            return False  # reject degenerate / unbounded (e.g. hallucinated) widths
        # both AddOperator and AddControl carry a wrapped operator -> must exist
        try:
            ops.get(edit.operator)
        except KeyError:
            return False
        if isinstance(edit, AddControl) and edit.control_kind not in ("branch", "loop"):
            return False
        return True
    return False
