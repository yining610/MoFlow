"""Recursive conditional-policy extraction from a built search tree.

Given V-hat(s) at every decision node, inference for a preference w: at each
decision pick the action whose Q-hat supports the w-optimal vector, then -- since
transitions are stochastic -- BRANCH on the realized successor and recurse.
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from ..search.nodes import DecisionNode


@dataclass
class PolicyBranch:
    """One realized-successor branch under a chosen action."""

    prob: float          # empirical P(s'|s,a)
    succ_key: str        # successor state signature
    label: str           # the realized edit label (which realization occurred)
    child: "PolicyNode"


@dataclass
class PolicyNode:
    """A node of the conditional policy served for a fixed preference w."""

    state_key: str
    graph: object                       # WorkflowGraph at this state
    value: np.ndarray | None            # supporting (maximize-form) vector for w
    chosen_label: str | None            # the action chosen here (None at a leaf)
    rationale: str
    terminal: bool
    branches: list[PolicyBranch] = field(default_factory=list)

    def render(self, spec=None, indent: int = 0) -> str:
        pad = "  " * indent
        if self.terminal:
            n = len(getattr(self.graph, "nodes", ()))
            val = ""
            if self.value is not None and spec is not None:
                d = spec.display(self.value)
                val = "  [" + ", ".join(f"{k}={round(v, 4)}" for k, v in d.items()) + "]"
            return f"{pad}* terminal ({n} ops){val}"
        lines = [f"{pad}choose: {self.chosen_label}  --  {self.rationale}"]
        for b in self.branches:
            lines.append(f"{pad}  |- p={b.prob:.2f} [{b.label}]")
            lines.append(b.child.render(spec, indent + 2))
        return "\n".join(lines)


def _best(ccs, w: np.ndarray, spec) -> tuple[float, np.ndarray | None]:
    """Return (max scalarized value, supporting vector) over a CCS, spec-normalized.

    - ccs: the convex coverage set to scan (empty/None yields -inf, None vector).
    - w: preference weights to scalarize by.
    """
    if ccs is None or ccs.is_empty():
        return float("-inf"), None
    scores = np.array([spec.scalarize(ccs.points[i], w) for i in range(len(ccs))])
    i = int(np.argmax(scores))
    return float(scores[i]), ccs.points[i].copy()


def extract_policy(
    root: DecisionNode, w: np.ndarray, spec, max_depth: int = 256
) -> PolicyNode | None:
    """Extract the conditional policy optimal for `w` from the built tree.

    - root: the search tree's root decision node.
    - w: preference weights to optimize for.
    - max_depth: recursion bound on policy depth.

    Returns None if the root was never expanded (no actions tried).
    """
    w = np.asarray(w, dtype=float)
    if not root.children:
        return None

    def go(node: DecisionNode, depth: int) -> PolicyNode:
        leaf = node.state.is_terminal() or not node.children or depth <= 0
        if not leaf:
            # pick the action whose Q-hat supports the best w-scalarized value
            best_key, best_chance, best_score = None, None, float("-inf")
            for ck, chance in node.children.items():
                sc, _ = _best(chance.Q, w, spec)
                if sc > best_score:
                    best_score, best_key, best_chance = sc, ck, chance
            leaf = (
                best_chance is None
                or best_chance.Q.is_empty()
                or not best_chance.successors
            )
        if leaf:
            _, vec = _best(node.V, w, spec)
            return PolicyNode(
                state_key=node.state.key(), graph=node.state.graph, value=vec,
                chosen_label=None, rationale="", terminal=True, branches=[],
            )

        total = best_chance.visits
        branches: list[PolicyBranch] = []
        for sk, succ in best_chance.successors.items():
            if succ.child is None:
                continue
            steps = succ.child.trace.steps
            label = steps[-1].label if steps else best_chance.edit.label()
            branches.append(
                PolicyBranch(
                    prob=(succ.visits / total if total else 0.0),
                    succ_key=sk,
                    label=label,
                    child=go(succ.child, depth - 1),
                )
            )
        _, vec = _best(best_chance.Q, w, spec)
        return PolicyNode(
            state_key=node.state.key(), graph=node.state.graph, value=vec,
            chosen_label=best_chance.edit.label(), rationale=best_chance.rationale,
            terminal=False, branches=branches,
        )

    return go(root, max_depth)


def policy_dict_signature(d: dict | None) -> str | None:
    """Stable structural hash of a `policy_to_dict`-shaped policy dict.
    """
    if not d:
        return None

    def _strip(x):
        if isinstance(x, dict):
            return {k: _strip(v) for k, v in x.items() if k != "rationale"}
        if isinstance(x, list):
            return [_strip(v) for v in x]
        return x

    payload = json.dumps(_strip(d), sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


def policy_signature(policy: "PolicyNode | None") -> str | None:
    """Stable structural hash identifying the workflow a served policy encodes.
    """
    if policy is None:
        return None
    # Lazy import: serialize_policy imports from this module (avoid a cycle at import).
    from ..workflow.serialize_policy import policy_to_dict

    return policy_dict_signature(policy_to_dict(policy))
