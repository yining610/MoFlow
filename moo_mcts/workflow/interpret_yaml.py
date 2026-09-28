from . import operators as ops
from .graph import OperatorNode, WorkflowGraph
from .serialize_yaml import from_yaml


async def run_node(node: OperatorNode, state, ctx):
    """Execute one node, honoring control semantics.
    """
    if not node.is_control:
        return await ctx.run(operator=node.operator, width=node.width, role=node.role, payload=state)

    # Control nodes bill their cost + quality in one shot via account() (the per-unit
    # runs use meter=False so they don't double-charge latency); the numbers come from
    # node_tokens/node_calls/node_quality_gain.
    if node.control_kind == "branch":
        # branch = parallel best-of-K: run `width` INDEPENDENT attempts from the same
        # input and majority-vote them (self-consistency), matching branch's one-round
        # cost model and best-of-K quality bonus. Without the vote the attempts collapse
        # to a single kept-last sample, so the modeled benefit never materializes.
        state = await ctx.run_branch(
            operator=node.operator, role=node.role, payload=state, width=node.width
        )
    else:
        # loop = sequential refinement: thread the state through `width` rounds, keep last.
        for _ in range(max(1, node.width)):
            state = await ctx.run_unit(
                operator=node.operator, role=node.role, payload=state, meter=False
            )
    acc_gain, rob_gain, cons_gain = ops.node_quality_gain(
        node.operator, node.width, node.control_kind
    )
    ctx.account(
        tokens=ops.node_tokens(node.operator, node.width, node.control_kind),
        sequential_calls=ops.node_calls(node.operator, node.width, node.control_kind),
        acc_logit=acc_gain,
        rob=rob_gain,
        cons=cons_gain,
    )
    return state


async def run_graph(g: WorkflowGraph, task, ctx):
    """Interpret the DAG against an Executor ctx.
    """
    state = task
    for n in g.nodes:
        state = await run_node(n, state, ctx)
    return state


async def run_yaml(text: str, task, ctx):
    """Parse YAML into a graph and interpret it (convenience wrapper).
    """
    return await run_graph(from_yaml(text), task, ctx)
