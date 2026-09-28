"""Serialize a conditional PolicyNode tree to/from YAML, and to executable Python."""

import numpy as np
import yaml

from ..serve.policy import PolicyBranch, PolicyNode
from .graph import WorkflowGraph


def _to_dict(p: PolicyNode) -> dict:
    if p.terminal:
        return {
            "terminal": True,
            "value": ([float(x) for x in p.value] if p.value is not None else None),
            "workflow": p.graph.to_dict(),
        }
    return {
        "terminal": False,
        "chosen": p.chosen_label,
        "rationale": p.rationale,
        "workflow": p.graph.to_dict(),
        "branches": [
            {
                "prob": float(b.prob),
                "label": b.label,
                "succ_key": b.succ_key,
                "policy": _to_dict(b.child),
            }
            for b in p.branches
        ],
    }


def policy_to_dict(policy: PolicyNode) -> dict:
    """Public, JSON-serializable view of a PolicyNode tree (see `_to_dict`).

    Reused by both `to_yaml_policy` and the policy-debug dump so the on-disk
    policy structure stays identical across the YAML and JSON artifacts.

    - policy: the root PolicyNode to serialize.
    """
    return _to_dict(policy)


def _from_dict(d: dict) -> PolicyNode:
    graph = WorkflowGraph.from_dict(d["workflow"])
    if d.get("terminal"):
        val = d.get("value")
        return PolicyNode(
            state_key=graph.signature(), graph=graph,
            value=(np.asarray(val, dtype=float) if val is not None else None),
            chosen_label=None, rationale="", terminal=True, branches=[],
        )
    branches = [
        PolicyBranch(
            prob=float(b.get("prob", 0.0)),
            succ_key=b.get("succ_key", ""),
            label=b.get("label", ""),
            child=_from_dict(b["policy"]),
        )
        for b in d.get("branches", [])
    ]
    return PolicyNode(
        state_key=graph.signature(), graph=graph, value=None,
        chosen_label=d.get("chosen"), rationale=d.get("rationale", ""),
        terminal=False, branches=branches,
    )


def to_yaml_policy(policy: PolicyNode) -> str:
    """Emit a human-readable, round-trippable YAML spec for the conditional policy.

    - policy: the root PolicyNode to serialize.
    """
    return yaml.safe_dump({"policy": _to_dict(policy)}, sort_keys=False, default_flow_style=False)


def from_yaml_policy(text: str) -> PolicyNode:
    """Parse a YAML policy spec back into a PolicyNode (inverse of `to_yaml_policy`).

    - text: the YAML policy spec.
    """
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or "policy" not in doc:
        raise ValueError("YAML missing top-level 'policy' key")
    return _from_dict(doc["policy"])


def to_python_policy(policy: PolicyNode, func_name: str = "policy") -> str:
    """Emit an executable async Python wrapper that runs the conditional policy."""
    y = to_yaml_policy(policy)
    lines = [
        "from moo_mcts.workflow.serialize_policy import from_yaml_policy",
        "from moo_mcts.serve.execute import run_policy",
        "",
        '_POLICY_YAML = """' + y + '"""',
        "",
        f"async def {func_name}(task, ctx, rng):",
        '    """Auto-generated conditional policy: sample a branch by P(s\'|s,a), run its workflow."""',
        "    policy = from_yaml_policy(_POLICY_YAML)",
        "    return await run_policy(policy, task, ctx, rng)",
    ]
    return "\n".join(lines) + "\n"
