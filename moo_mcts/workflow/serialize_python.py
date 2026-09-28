"""Serialize a WorkflowGraph to an executable async Python function (codegen).

The "Python" representation: a self-contained async `workflow(task, ctx)` where
`ctx` exposes the Executor backend (`ctx.run(operator, width, role, payload)`).
The source is human-readable so a discovered workflow can be inspected/exported.

Codegen and the YAML interpreter (interpret_yaml.py) share execution semantics
via the Executor protocol, so either path yields the same reward vector
(asserted in tests).
"""


from .graph import WorkflowGraph


def to_python_source(g: WorkflowGraph, func_name: str = "workflow") -> str:
    """Emit readable async Python source for the workflow.

    Threads a `state` through each node in construction order, executing each via
    the shared `run_node` expander (workflow.interpret_yaml) so interpreted and
    compiled forms have identical semantics and ExecMeta. `run_node` is imported
    at the top of the emitted module for standalone readability/export.

    - g: the WorkflowGraph to serialize.
    - func_name: name of the generated async function.
    """
    lines: list[str] = []
    lines.append("from moo_mcts.workflow.interpret_yaml import run_node")
    lines.append("from moo_mcts.workflow.graph import OperatorNode")
    lines.append("")
    lines.append(f"async def {func_name}(task, ctx):")
    lines.append('    """Auto-generated workflow. See WorkflowGraph.signature for identity."""')
    lines.append("    state = task")
    if g.is_empty():
        lines.append("    # empty template: pass the task through unchanged")
        lines.append("    return state")
        return "\n".join(lines) + "\n"

    for n in g.nodes:
        kind = f"control[{n.control_kind}] " if n.is_control else ""
        lines.append(f"    # node {n.id}: {kind}{n.operator} (role={n.role!r}, width={n.width})")
        node_repr = (
            f"OperatorNode(id={n.id}, operator={n.operator!r}, width={n.width}, "
            f"role={n.role!r}, is_control={n.is_control!r}, control_kind={n.control_kind!r})"
        )
        lines.append(f"    state = await run_node({node_repr}, state, ctx)")
    lines.append("    return state")
    return "\n".join(lines) + "\n"


def compile_python(g: WorkflowGraph, func_name: str = "workflow"):
    """Compile the generated source into a callable async function object.

    Returns the coroutine function. Only trusted codegen is exec'd (source
    interpolates just enum-like operator names, integer widths, role labels), so
    it is safe in-process. Shared `run_node` / `OperatorNode` are injected into
    the namespace so the compiled function runs the same expander as the interpreter.

    - g: the WorkflowGraph to compile.
    - func_name: name of the generated async function.
    """
    from .graph import OperatorNode
    from .interpret_yaml import run_node

    src = to_python_source(g, func_name)
    namespace: dict = {"run_node": run_node, "OperatorNode": OperatorNode}
    exec(compile(src, filename=f"<workflow:{g.signature()}>", mode="exec"), namespace)
    return namespace[func_name]
