from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorSpec:
    """Static metadata for one operator type."""

    name: str
    base_llm_calls: int  # sequential calls for the unit (width-1) operator
    base_tokens: int     # tokens per call (approx)
    parallel: bool = False  # if True, width>1 adds tokens but not latency

    accuracy_gain: float = 0.0
    robustness_gain: float = 0.0
    consistency_gain: float = 0.0
    description: str = ""
    # Whether this operator, as the LAST node, yields the task's final answer. Preparatory
    # operators (e.g. Localize outputs only analysis, no patch) set this False so a workflow
    # may not terminate on them (see edits.is_legal for Terminate).
    emits_answer: bool = True

    def llm_calls(self, width: int = 1) -> int:
        """Sequential LLM calls for this operator at the given ensemble width."""
        if self.parallel:
            return self.base_llm_calls  # parallel: latency does not grow with width
        return self.base_llm_calls * max(1, width)

    def tokens(self, width: int = 1) -> int:
        """Total tokens consumed at the given width (always grows with width)."""
        return self.base_tokens * self.base_llm_calls * max(1, width)


def node_calls(operator: str, width: int, control_kind: str = "") -> int:
    """Sequential LLM calls for a node, honoring control semantics.

      plain   : spec.llm_calls(width)   # static parallel flag applies.
      loop  L : base_llm_calls * L      # L sequential refinement rounds.
      branch K: base_llm_calls          # K parallel attempts, one round.

    - operator: registered operator name.
    - width: ensemble width / loop iterations / branch attempts.
    - control_kind: "loop", "branch", or "" for a plain node.
    """
    spec = get(operator)
    if control_kind == "loop":
        return spec.base_llm_calls * max(1, width)
    if control_kind == "branch":
        return spec.base_llm_calls
    return spec.llm_calls(width)


def node_tokens(operator: str, width: int, control_kind: str = "") -> int:
    """Total tokens for a node, honoring control semantics (see node_calls).

      plain   : spec.tokens(width).
      loop  L : spec.tokens(1) * L      # L sequential unit invocations.
      branch K: spec.tokens(1) * K      # K parallel unit invocations.

    - operator: registered operator name.
    - width: ensemble width / loop iterations / branch attempts.
    - control_kind: "loop", "branch", or "" for a plain node.
    """
    spec = get(operator)
    if control_kind in ("loop", "branch"):
        return spec.tokens(1) * max(1, width)
    return spec.tokens(width)


def node_quality_gain(
    operator: str, width: int, control_kind: str = ""
) -> tuple[float, float, float]:
    """Return (accuracy_logit_gain, robustness_gain, consistency_gain) for a node.

    Width helps with diminishing returns; the mapping per control kind:
      plain : acc *= 1 + 0.5*log1p(width-1)
              rob *= 1 + 0.3*(width-1)         # clamped by caller
              cons*= 1 + 0.5*log1p(width-1)    # clamped by caller
      loop L: all three *= 1 + 0.5*log1p(L-1) (sequential refinement).
      branch K: all three *= 1 + 0.6*log1p(K-1) (parallel best-of-K).
    """
    import numpy as np  # local import: never pass np as an arg ([[no-module-as-argument]])

    spec = get(operator)
    if control_kind == "loop":
        bonus = 1.0 + 0.5 * float(np.log1p(max(0, width - 1)))
        return (spec.accuracy_gain * bonus,
                spec.robustness_gain * bonus,
                spec.consistency_gain * bonus)
    if control_kind == "branch":
        bonus = 1.0 + 0.6 * float(np.log1p(max(0, width - 1)))
        return (spec.accuracy_gain * bonus,
                spec.robustness_gain * bonus,
                spec.consistency_gain * bonus)
    acc = spec.accuracy_gain * (1.0 + 0.5 * float(np.log1p(max(0, width - 1))))
    rob = spec.robustness_gain * (1.0 + 0.3 * (max(1, width) - 1))
    cons = spec.consistency_gain * (1.0 + 0.5 * float(np.log1p(max(0, width - 1))))
    return acc, rob, cons


_REGISTRY: dict[str, OperatorSpec] = {}


def register(spec: OperatorSpec) -> OperatorSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"operator already registered: {spec.name}")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> OperatorSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown operator {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_specs() -> list[OperatorSpec]:
    return list(_REGISTRY.values())


def names() -> list[str]:
    return list(_REGISTRY.keys())


def _load_defaults() -> None:
    if _REGISTRY:
        return

    register(OperatorSpec("Generate", 1, 2400, parallel=False,
                          accuracy_gain=0.20, robustness_gain=0.55, consistency_gain=0.45,
                          description="Single chain-of-thought generation."))
    register(OperatorSpec("ReviewRevise", 2, 2200, parallel=False,
                          accuracy_gain=0.08, robustness_gain=0.10, consistency_gain=0.10,
                          description="Generate then critique-and-revise (Madaan et al.)."))
    register(OperatorSpec("Ensemble", 1, 2200, parallel=True,
                          accuracy_gain=0.05, robustness_gain=0.25, consistency_gain=0.30,
                          description="k parallel samples + majority/self-consistency vote."))
    register(OperatorSpec("Test", 1, 1500, parallel=False,
                          accuracy_gain=0.05, robustness_gain=0.08, consistency_gain=0.10,
                          description="Generate unit tests and check the candidate."))
    register(OperatorSpec("Programmer", 1, 1800, parallel=False,
                          accuracy_gain=0.20, robustness_gain=0.10, consistency_gain=0.30,
                          description="Write-and-execute code to compute the answer."))
    register(OperatorSpec("TestCode", 2, 1800, parallel=False,
                          accuracy_gain=0.30, robustness_gain=0.10, consistency_gain=0.15,
                          description="Execute public tests; reflect on failures and repair the code (AFlow MBPP Test)."))
    register(OperatorSpec("Custom", 1, 2000, parallel=False,
                          accuracy_gain=0.15, robustness_gain=0.50, consistency_gain=0.45,
                          description="Generic single LLM node."))
    register(OperatorSpec("StepProgrammer", 1, 1800, parallel=False,
                          accuracy_gain=0.15, robustness_gain=0.15, consistency_gain=0.25,
                          description="Candidate-aware programmer: write-and-execute code that builds on the previous step's output (approach, "
                                      "subproblems, partial results, or a proposed answer) to compute the final answer."))
    register(OperatorSpec("Localize", 1, 2000, parallel=False,
                          accuracy_gain=0.10, robustness_gain=0.12, consistency_gain=0.15,
                          emits_answer=False,  # outputs analysis only; must be followed by a patch-producing op
                          description="SWE fault localization: name the responsible file(s)/function(s) before patching (no code yet)."))
    register(OperatorSpec("PatchRepair", 3, 2600, parallel=False,
                          accuracy_gain=0.32, robustness_gain=0.12, consistency_gain=0.15,
                          description="SWE apply/test loop: draft a unified-diff patch, grade it "
                                      "against the repo's tests via the harness, and repair on "
                                      "failure using the grader feedback."))
    register(OperatorSpec("Decompose", 2, 2200, parallel=False,
                          accuracy_gain=0.25, robustness_gain=0.15, consistency_gain=0.20,
                          description="Multi-hop question decomposition (self-ask / least-to-most): "
                                      "break the question into ordered single-hop sub-questions, "
                                      "answer each from the context, then compose the final answer."))
    register(OperatorSpec("EvidenceSelect", 2, 2200, parallel=False,
                          accuracy_gain=0.15, robustness_gain=0.30, consistency_gain=0.20,
                          description="Distractor filtering: identify the supporting passages/"
                                      "sentences among distractors, then answer using only that "
                                      "selected evidence."))
    register(OperatorSpec("GroundCheck", 2, 2000, parallel=False,
                          accuracy_gain=0.10, robustness_gain=0.25, consistency_gain=0.20,
                          description="Faithfulness verifier: cite the passage supporting the "
                                      "candidate answer; if unsupported, re-derive a corrected, "
                                      "context-grounded answer."))
    register(OperatorSpec("EliminateChoices", 2, 2400, parallel=False,
                          accuracy_gain=0.22, robustness_gain=0.15, consistency_gain=0.20,
                          description="Multiple-choice process-of-elimination: argue for/against "
                                      "each option A-D, rule out the distractors, then select the "
                                      "single surviving best answer."))


_load_defaults()
