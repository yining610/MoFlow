import numpy as np

from ..ccs.ccs import CCS
from ..concurrency import parallel_map
from ..mdp import actions as mdp_actions
from ..mdp.state import State
from ..search.trace import ConstructionTrace
from ..workflow.edits import Terminate


def _greedy_complete(
    state: State, proposer, max_depth: int, rng, trace: ConstructionTrace
) -> tuple[State, ConstructionTrace]:
    """Greedily fill open slots until terminal, picking one proposer suggestion per step.

    - state: the partial state to complete.
    - proposer: supplies candidate edits at each step.
    - max_depth: completion horizon (also bounds the step guard).
    - rng: varies which suggestion (and completion length) is taken.
    - trace: construction trace to extend.

    Returns the terminal state and the trace extended with each chosen edit, so the
    completion archives with its real decision history, not the prefix's trace.
    """
    s, t = state, trace
    guard = 0
    while not s.is_terminal() and guard < max_depth + 2:
        guard += 1
        if mdp_actions.must_terminate(s):
            e = Terminate()
            s, t = mdp_actions.apply(s, e), t.extended(e.label(), "horizon reached")
            break
        props = proposer.propose(s.graph, n=max(2, 3), max_depth=max_depth)
        if not props:
            e = Terminate()
            s, t = mdp_actions.apply(s, e), t.extended(e.label(), "no further proposals")
            break
        # weight toward terminate sometimes so completions vary in length
        i = int(rng.integers(0, len(props)))
        p = props[i]
        s, t = mdp_actions.apply(s, p.edit), t.extended(p.edit.label(), p.rationale)
    if not s.is_terminal():
        e = Terminate()
        s, t = mdp_actions.apply(s, e), t.extended(e.label(), "completion cutoff")
    return s, t


def completion_beam(
    state: State,
    proposer,
    evaluator,
    evalset,
    k: int,
    rng,
    base_trace: ConstructionTrace = None,
) -> tuple[CCS, list[np.ndarray], list[tuple[np.ndarray, State, ConstructionTrace]]]:
    """Estimate the accuracy/robustness front of a partial workflow via k diverse completions.
    """
    D = evaluator.spec.D
    base_trace = base_trace if base_trace is not None else ConstructionTrace()
    rewards: list[np.ndarray] = []
    completions: list[tuple[np.ndarray, State, ConstructionTrace]] = []

    # Diversify the first extension across distinct proposer suggestions, then
    # greedily complete each.
    first = proposer.propose(state.graph, n=max(k, 3), max_depth=state.max_depth)
    if not first:
        first = [None]

    seeds = rng.integers(0, 2**31 - 1, size=k)

    def _complete_one(j):
        sub_rng = np.random.default_rng(int(seeds[j]))
        choice = first[j % len(first)]
        if choice is not None and mdp_actions.is_action_legal(state, choice.edit):
            seed_state = mdp_actions.apply(state, choice.edit)
            seed_trace = base_trace.extended(choice.edit.label(), choice.rationale)
        else:
            seed_state, seed_trace = state, base_trace
        terminal, term_trace = _greedy_complete(
            seed_state, proposer, state.max_depth, sub_rng, seed_trace
        )
        res = evaluator.evaluate(terminal.graph, evalset)
        return (res.vector, terminal, term_trace)

    completions = parallel_map(_complete_one, list(range(k)), evaluator.max_workers)
    rewards = [vec for (vec, _t, _tr) in completions]

    if not rewards:
        return CCS.empty(D), [], []
    return CCS.of(rewards).pruned(), rewards, completions
