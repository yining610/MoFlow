"""MDP state: a partial workflow under construction.
"""


from dataclasses import dataclass

from ..workflow.graph import WorkflowGraph


@dataclass(frozen=True)
class State:
    graph: WorkflowGraph
    max_depth: int

    @classmethod
    def root(cls, max_depth: int) -> "State":
        return cls(graph=WorkflowGraph.empty(), max_depth=max_depth)

    @property
    def depth(self) -> int:
        return self.graph.depth

    def is_terminal(self) -> bool:
        return self.graph.terminated

    def at_horizon(self) -> bool:
        return self.graph.depth >= self.max_depth

    def key(self) -> str:
        return self.graph.signature()