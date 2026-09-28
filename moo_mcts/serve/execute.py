"""Execute a conditional policy.

A PolicyNode tree branches on the realized successor state, with each branch carrying its empirical P(s'|s,a).
"""

from ..workflow.interpret_yaml import run_graph


def sample_terminal(policy, rng):
    """Descend the policy, sampling branches by empirical P(s'|s,a), and return the terminal WorkflowGraph reached."""
    node = policy
    guard = 0
    while not node.terminal and node.branches and guard < 1024:
        guard += 1
        probs = [max(0.0, float(b.prob)) for b in node.branches]
        total = sum(probs)
        if total <= 0.0:
            node = node.branches[0].child
            continue
        r = float(rng.random()) * total
        cum, chosen = 0.0, node.branches[-1]
        for b, p in zip(node.branches, probs):
            cum += p
            if r <= cum:
                chosen = b
                break
        node = chosen.child
    return node.graph


async def run_policy(policy, task, ctx, rng):
    """Sample a terminal workflow from the policy and execute it on `task`."""
    graph = sample_terminal(policy, rng)
    return await run_graph(graph, task, ctx)
