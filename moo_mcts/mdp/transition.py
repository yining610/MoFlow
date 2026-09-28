from typing import Callable, Optional

from ..workflow.edits import AtomicEdit
from ..workflow.graph import WorkflowGraph
from .actions import apply
from .state import State

# A realizer maps (graph, nominal_edit, seed) -> a concrete edit to apply.
Realizer = Callable[[WorkflowGraph, AtomicEdit, int], AtomicEdit]


def instantiate(
    state: State,
    edit: AtomicEdit,
    sample_seed: int = 0,
    realizer: Optional[Realizer] = None,
) -> State:
    """Realize `edit` at `state`, returning the concrete successor state.

    - state: the current MDP state to transition from.
    - edit: the nominal (structural) edit chosen by the decision.
    - sample_seed: selects which concrete realization to draw; distinct seeds may
      yield distinct successor graphs, which is how a chance node accumulates
      multiple successors.
    - realizer: optional function mapping the nominal edit to a concrete edit for
      the given seed.
    """
    concrete = realizer(state.graph, edit, sample_seed) if realizer is not None else edit
    return apply(state, concrete)
