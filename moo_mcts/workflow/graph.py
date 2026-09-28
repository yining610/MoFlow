"""WorkflowGraph: the typed DAG that is the single source of truth for a workflow.
"""


import hashlib
import json
from dataclasses import dataclass, replace

from . import operators as ops


@dataclass(frozen=True)
class OperatorNode:
    """One operator instance in the DAG."""

    id: int
    operator: str   # an OperatorSpec name
    width: int = 1  # ensemble width k (1 for non-ensemble ops)
    role: str = "default"  # prompt archetype / role label (interpretability)
    is_control: bool = False  # True for branch/loop marker nodes
    control_kind: str = ""  # "branch" | "loop" | ""

    def spec(self) -> ops.OperatorSpec:
        return ops.get(self.operator)


@dataclass(frozen=True)
class WorkflowGraph:
    """An immutable, partial-or-complete workflow DAG.

    - nodes: tuple of OperatorNode in topological (construction) order.
    - edges: tuple of (src_id, dst_id) dataflow edges.
    - frontier: id of the node whose output is the open slot, or None.
    - terminated: whether `terminate` has been applied (complete workflow).
    """

    nodes: tuple[OperatorNode, ...] = ()
    edges: tuple[tuple[int, int], ...] = ()
    frontier: int = None
    terminated: bool = False

    @classmethod
    def empty(cls) -> "WorkflowGraph":
        """Return the empty template W0 (AFlow's blank start)."""
        return cls()

    def _next_id(self) -> int:
        return (max((n.id for n in self.nodes), default=-1)) + 1

    def add_node(self, node: OperatorNode) -> "WorkflowGraph":
        """Return a new graph with `node` appended and an edge from the frontier.

        - node: the OperatorNode to append.
        """
        new_nodes = self.nodes + (node,)
        new_edges = self.edges
        if self.frontier is not None:
            new_edges = new_edges + ((self.frontier, node.id),)
        return replace(self, nodes=new_nodes, edges=new_edges, frontier=node.id)

    def replace_node(self, node_id: int, **changes) -> "WorkflowGraph":
        """Return a new graph with node `node_id` modified (e.g. set_role).

        - node_id: id of the node to modify.
        - changes: field=value overrides applied to that node.
        """
        new_nodes = tuple(
            (replace(n, **changes) if n.id == node_id else n) for n in self.nodes
        )
        return replace(self, nodes=new_nodes)

    def terminate(self) -> "WorkflowGraph":
        return replace(self, terminated=True, frontier=None)

    def node(self, node_id: int) -> OperatorNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"no node {node_id}")

    @property
    def depth(self) -> int:
        """Construction depth so far, i.e. the count of operator/control nodes."""
        return len(self.nodes)

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    def successors(self, node_id: int) -> list[int]:
        return [d for (s, d) in self.edges if s == node_id]

    def predecessors(self, node_id: int) -> list[int]:
        return [s for (s, d) in self.edges if d == node_id]

    def signature(self) -> str:
        """Return a stable content hash for caching and state identity.

        Two graphs with identical structure (nodes, widths, roles, edges,
        frontier, terminated) hash equal, enabling transposition reuse in the
        search tree.
        """
        payload = {
            "nodes": [
                [n.id, n.operator, n.width, n.role, n.is_control, n.control_kind]
                for n in self.nodes
            ],
            "edges": [list(e) for e in self.edges],
            "frontier": self.frontier,
            "terminated": self.terminated,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Return a plain-dict view of the graph (used by the YAML serializer)."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "operator": n.operator,
                    "width": n.width,
                    "role": n.role,
                    "is_control": n.is_control,
                    "control_kind": n.control_kind,
                }
                for n in self.nodes
            ],
            "edges": [list(e) for e in self.edges],
            "frontier": self.frontier,
            "terminated": self.terminated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowGraph":
        nodes = tuple(
            OperatorNode(
                id=int(nd["id"]),
                operator=str(nd["operator"]),
                width=int(nd.get("width", 1)),
                role=str(nd.get("role", "default")),
                is_control=bool(nd.get("is_control", False)),
                control_kind=str(nd.get("control_kind", "")),
            )
            for nd in d.get("nodes", [])
        )
        edges = tuple((int(a), int(b)) for a, b in d.get("edges", []))
        frontier = d.get("frontier", None)
        return cls(
            nodes=nodes,
            edges=edges,
            frontier=None if frontier is None else int(frontier),
            terminated=bool(d.get("terminated", False)),
        )
