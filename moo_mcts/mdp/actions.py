from ..workflow.edits import AtomicEdit, is_legal
from .state import State


def apply(state: State, edit: AtomicEdit) -> State:
    """Apply an atomic edit to a state, returning the successor state.

    - state: the current MDP state whose graph the edit is applied to.
    - edit: the atomic edit to apply.
    """
    return State(graph=edit.apply(state.graph), max_depth=state.max_depth)


def is_action_legal(state: State, edit: AtomicEdit) -> bool:
    return is_legal(edit, state.graph, state.max_depth)


def must_terminate(state: State) -> bool:
    """At the horizon, the only legal continuation is terminate."""
    return state.at_horizon() and not state.is_terminal()
