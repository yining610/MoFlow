import numpy as np

from ..ccs.ccs import CCS
from .nodes import ChanceNode, DecisionNode


def beta(n_s: float, half_life: float) -> float:
    """Evidence weight in [0,1)

    - n_s: accumulated evidence (executed descendant samples) at the state.
    - half_life: evidence at which beta reaches 0.5.
    """
    return float(n_s / (n_s + half_life))


def backup_chance(node: ChanceNode) -> CCS:
    """Q-hat(s,a) = prune(union_s' (E[R] (+) V-hat(s'))) over successors, set on node.Q.

    - node: the chance node whose successors are backed up into its coverage set.

    The agent observes which s' occurred and continues optimally, so the set keeps every 
    branch's front rather than the probability-weighted mean.
    """
    D = node.D
    parts: list[CCS] = []
    for succ in node.successors.values():
        child = succ.child
        if child is None:
            continue
        if child.state.is_terminal():
            if succ.sample_rewards:
                mean_r = np.mean(np.stack(succ.sample_rewards), axis=0)
                parts.append(CCS.of([mean_r]))
            elif not child.V.is_empty():
                parts.append(child.V.copy())
        elif not child.V.is_empty():
            parts.append(child.V.copy())

    node.Q = CCS.union_all(parts, D=D) if parts else CCS.empty(D)
    return node.Q


def backup_decision(
    node: DecisionNode,
    half_life: float,
) -> CCS:
    """V-hat(s) = prune(union_a Q-hat(s,a)), beta-annealed with the prior (f_theta / critic), set on node.V.

    - node: the decision node whose children's Q sets are unioned into its value set.
    - half_life: evidence at which beta reaches 0.5, controlling the prior-to-backup
      handover (prior kept as an optimistic candidate while beta < 0.5).
    """
    D = node.D
    backed = CCS.union_all((c.Q for c in node.children.values()), D=D)

    n_s = node.n_executed_descendants()
    b = beta(n_s, half_life)

    if node.prior is not None and b < 0.5:
        # cold start: include an OPTIMISTIC prior point so no reachable region is
        # pruned. The prior estimate is treated as one extra candidate
        # vector in the set; dominance pruning keeps it only if non-dominated.
        est, _unc = node.prior
        prior_set = CCS.of([est])
        node.V = backed.union(prior_set) if not backed.is_empty() else prior_set
    else:
        node.V = backed if not backed.is_empty() else (
            CCS.of([node.prior[0]]) if node.prior is not None else CCS.empty(D)
        )
    return node.V
