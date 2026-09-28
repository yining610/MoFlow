import json

import numpy as np

from ..ccs.ccs import CCS
from ..objectives import ObjectiveSpec
from ..workflow.serialize_yaml import from_yaml, to_yaml
from .czt import CZTSelector
from .nodes import ChanceNode, DecisionNode, Successor
from .trace import ConstructionTrace, TraceStep


def _vec(spec: ObjectiveSpec | None, v) -> dict | list:
    """Render a single value vector either as named axes (with spec) or a raw list."""
    arr = np.asarray(v, dtype=float)
    if spec is not None:
        return {k: round(float(val), 6) for k, val in spec.display(arr).items()}
    return [round(float(x), 6) for x in arr]


def _ccs(spec: ObjectiveSpec | None, ccs) -> list:
    """Render a CCS front as a list of value vectors."""
    if ccs is None or ccs.is_empty():
        return []
    return [_vec(spec, ccs.points[i]) for i in range(len(ccs))]


def _trace(spec: ObjectiveSpec | None, trace) -> list:
    """Render a ConstructionTrace as a list of {label, rationale, delta} steps."""
    out = []
    for s in trace.steps:
        step = {"label": s.label, "rationale": s.rationale}
        if s.delta is not None:
            step["delta"] = _vec(spec, s.delta)
        out.append(step)
    return out


def tree_to_dict(root: DecisionNode, spec: ObjectiveSpec = None) -> dict:
    """Walk the search tree from `root` into a JSON-serializable dict.

    - root: the engine's root DecisionNode.
    - spec: optional ObjectiveSpec; when given, value vectors are rendered as
      named, display-form axes instead of raw maximize-form lists.
    """
    seen: set[str] = set()

    def node_dict(node: DecisionNode) -> dict:
        key = node.state.key()
        if key in seen:
            return {"ref": key}
        seen.add(key)
        d: dict = {
            "state_key": key,
            "depth": node.state.depth,
            "terminal": node.state.is_terminal(),
            "V_hat": _ccs(spec, node.V),
            "trace": _trace(spec, node.trace),
            "workflow_yaml": to_yaml(node.state.graph),
            "children": [],
        }
        if node.prior is not None:
            est, unc = node.prior
            d["prior"] = {"estimate": _vec(spec, est), "uncertainty": round(float(unc), 6)}
        if node._leaf_samples:
            d["leaf_samples"] = [_vec(spec, v) for v in node._leaf_samples]
        for ck, chance in node.children.items():
            cd: dict = {
                "action_key": list(ck) if isinstance(ck, tuple) else ck,
                "edit": chance.edit.label(),
                "rationale": chance.rationale,
                "visits": chance.visits,
                "n_samples": chance.n_samples,
                "Q_hat": _ccs(spec, chance.Q),
                "successors": [],
            }
            for sk, succ in chance.successors.items():
                sd: dict = {
                    "successor_key": sk,
                    "visits": succ.visits,
                    "prob": round(chance.prob(sk), 6),
                    "n_reward_draws": succ.n_samples,
                }
                if succ.child is not None:
                    sd["child"] = node_dict(succ.child)
                cd["successors"].append(sd)
            d["children"].append(cd)
        return d

    d = node_dict(root)
    if spec is not None:
        d["objectives"] = list(spec.names)
    return d


def render_tree(root: DecisionNode, spec: ObjectiveSpec = None) -> str:
    """Render the search tree from `root` as an indented, human-readable string.
    """
    seen: set[str] = set()
    lines: list[str] = []

    def fmt_front(ccs) -> str:
        pts = _ccs(spec, ccs)
        if not pts:
            return "(empty)"
        return "; ".join(str(p) for p in pts)

    def walk(node: DecisionNode, indent: str, label: str) -> None:
        key = node.state.key()
        tag = "TERMINAL" if node.state.is_terminal() else f"depth={node.state.depth}"
        head = f"{indent}{label}[{tag}] key={key[:12]}"
        if key in seen:
            lines.append(f"{head}  -> (seen)")
            return
        seen.add(key)
        lines.append(head)
        lines.append(f"{indent}  V_hat = {fmt_front(node.V)}")
        if node.prior is not None:
            est, unc = node.prior
            lines.append(f"{indent}  prior = {_vec(spec, est)} (unc={float(unc):.4g})")
        for ck, chance in node.children.items():
            lines.append(
                f"{indent}  |- action '{chance.edit.label()}' "
                f"visits={chance.visits} samples={chance.n_samples}"
            )
            lines.append(f"{indent}  |    rationale: {chance.rationale}")
            lines.append(f"{indent}  |    Q_hat = {fmt_front(chance.Q)}")
            for sk, succ in chance.successors.items():
                if succ.child is None:
                    lines.append(
                        f"{indent}  |    -> succ {sk[:12]} p={chance.prob(sk):.3g} (no child)"
                    )
                    continue
                walk(
                    succ.child,
                    indent + "  |    ",
                    f"-> p={chance.prob(sk):.3g} ",
                )

    walk(root, "", "ROOT ")
    return "\n".join(lines)


def dump_tree(root: DecisionNode, path: str, spec: ObjectiveSpec = None) -> dict:
    """Serialize the full tree to a JSON file at `path` and return the dict written.

    - root: the engine's root DecisionNode.
    - path: filesystem path for the JSON dump.
    - spec: optional ObjectiveSpec for named value-axis display.
    """
    d = tree_to_dict(root, spec)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)
    return d


class _LoadedState:
    """Minimal `State` stand-in rebuilt from a dump, exposing only what the policy
    extractor reads: key(), is_terminal(), the WorkflowGraph, and the depth.
    """

    def __init__(self, graph, key: str, terminal: bool, depth: int):
        self.graph = graph
        self._key = key
        self._terminal = terminal
        self.depth = depth

    def key(self) -> str:
        return self._key

    def is_terminal(self) -> bool:
        return self._terminal


class _LoadedEdit:
    """Minimal edit stand-in exposing only label() (the branch-label fallback)."""

    def __init__(self, label: str):
        self._label = label

    def label(self) -> str:
        return self._label


def _vec_in(entry, spec: ObjectiveSpec) -> np.ndarray:
    """Inverse of `_vec`: a named display dict -> maximize-form vector (signs flipped
    back via the spec), or a raw list -> vector as-is (dumps made with spec=None).
    """
    if isinstance(entry, dict):
        return spec.vector(**entry)
    return np.asarray(entry, dtype=float)


def _ccs_in(points, spec: ObjectiveSpec) -> CCS:
    """Rebuild a CCS front from a list of dumped value vectors (empty -> empty CCS)."""
    if not points:
        return CCS.empty(spec.D)
    return CCS.of([_vec_in(p, spec) for p in points])


def load_tree(data, spec: ObjectiveSpec) -> DecisionNode:
    """Reconstruct a search tree from a `dump_tree`/`tree_to_dict` artifact.
    """
    if isinstance(data, str):
        with open(data, "r", encoding="utf-8") as fh:
            data = json.load(fh)

    registry: dict[str, DecisionNode] = {}

    def trace_of(steps) -> ConstructionTrace:
        t = ConstructionTrace()
        for s in steps:
            t.steps.append(
                TraceStep(label=s.get("label", ""), rationale=s.get("rationale", ""))
            )
        return t

    def build(nd: dict) -> DecisionNode:
        if "ref" in nd:
            return registry[nd["ref"]]
        key = nd["state_key"]
        state = _LoadedState(
            from_yaml(nd["workflow_yaml"]), key,
            bool(nd.get("terminal", False)), int(nd.get("depth", 0)),
        )
        # placeholder selector: a loaded tree is rendered, never searched.
        node = DecisionNode(
            state, spec.D, CZTSelector(spec=spec), trace_of(nd.get("trace", []))
        )
        registry[key] = node  # register before recursing so refs back to it resolve
        node.V = _ccs_in(nd.get("V_hat", []), spec)
        for cd in nd.get("children", []):
            ak = cd.get("action_key")
            action_key = tuple(ak) if isinstance(ak, list) else ak
            chance = ChanceNode(
                action_key, _LoadedEdit(cd.get("edit", "")), cd.get("rationale", ""), spec.D
            )
            chance.Q = _ccs_in(cd.get("Q_hat", []), spec)
            chance.visits = int(cd.get("visits", 0))
            for sd in cd.get("successors", []):
                child = build(sd["child"]) if "child" in sd else None
                chance.successors[sd["successor_key"]] = Successor(
                    child=child, visits=int(sd.get("visits", 0))
                )
            node.children[action_key] = chance
        return node

    return build(data)
