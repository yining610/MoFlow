"""Serialize a WorkflowGraph to/from a YAML spec (the "YAML" representation).

This is a faithful, round-trippable view of the DAG (S "both Python and YAML").
The YAML is also directly executable via interpret_yaml.py, so a workflow can be
shipped as YAML with no Python file written to disk (LEMON-style).
"""


import yaml
from .graph import WorkflowGraph


def to_yaml(g: WorkflowGraph) -> str:
    """Emit a human-readable YAML spec for the workflow.

    - g: the WorkflowGraph to serialize.
    """
    doc = {"workflow": g.to_dict()}
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def from_yaml(text: str) -> WorkflowGraph:
    """Parse a YAML spec back into a WorkflowGraph (inverse of `to_yaml`).

    - text: the YAML workflow spec.
    """
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or "workflow" not in doc:
        raise ValueError("YAML missing top-level 'workflow' key")
    return WorkflowGraph.from_dict(doc["workflow"])
