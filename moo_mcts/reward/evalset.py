from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class EvalItem:
    """One unit to run the workflow on: one (problem, paraphrase) variant x one sample."""

    payload: Any  # the task/question text or structured input
    seed: int  # determinism handle for the executor's realization draws
    group: int = 0  # problem id (the robustness-aggregation unit)
    gold: Any = None  # gold answer
    variant: int = 0  # paraphrase id within the problem (0 = original wording)


@runtime_checkable
class EvalSetProvider(Protocol):
    def items(self) -> list[EvalItem]:
        ...

    @property
    def n_groups(self) -> int:
        """Number of robustness groups (problems)."""
        ...


def _variant_texts(original: Any, paraphrases: list[Any], m: int) -> list[Any]:
    """Return the original text and up to m-1 paraphrases (if available)."""

    variants = [original] + list(paraphrases)
    m = max(1, int(m))
    return variants[:m]


@dataclass
class TaskValidationSplit:
    """Per-task eval set: a fixed validation split, run over each problem's paraphrases.
    """

    examples: list[tuple]
    seeds: int = 1
    base_seed: int = 0
    n_paraphrases: int = 1

    @staticmethod
    def _unpack(ex: tuple) -> tuple[Any, int, Any, list[Any]]:
        if len(ex) >= 4:
            payload, ex_id, gold, paraphrases = ex[0], ex[1], ex[2], ex[3]
        elif len(ex) == 3:
            payload, ex_id, gold = ex
            paraphrases = []
        else:
            (payload, ex_id), gold, paraphrases = ex, None, []
        return payload, int(ex_id), gold, list(paraphrases or [])

    def items(self) -> list[EvalItem]:
        out: list[EvalItem] = []
        for ex in self.examples:
            payload, ex_id, gold, paraphrases = self._unpack(ex)
            variants = _variant_texts(payload, paraphrases, self.n_paraphrases)
            for v_id, text in enumerate(variants):
                for s in range(self.seeds):
                    out.append(
                        EvalItem(
                            payload=text,
                            seed=self.base_seed + ex_id * 100003 + v_id * 97 + s,
                            group=ex_id,
                            gold=gold,
                            variant=v_id,
                        )
                    )
        return out

    def accuracy_view(self) -> "TaskValidationSplit":

        return TaskValidationSplit(
            examples=self.examples, seeds=self.seeds,
            base_seed=self.base_seed, n_paraphrases=1,
        )

    @property
    def n_groups(self) -> int:
        return len({self._unpack(ex)[1] for ex in self.examples})


@dataclass
class QueryNeighborhood:
    """Per-query eval set: the m paraphrase variants of a single query (one problem).

    One EvalItem per variant (like `TaskValidationSplit` with seeds=1); the R repeat
    draws are supplied by the engine's `chance_samples` in `Evaluator.evaluate_terminal`,
    so there is no per-eval-set sample knob here (that used to double-count with R).
    """

    paraphrases: list[Any]
    base_seed: int = 0
    gold: Any = None  # one gold answer shared across paraphrases (real grading)

    def items(self) -> list[EvalItem]:
        out: list[EvalItem] = []
        for p_id, text in enumerate(self.paraphrases):
            out.append(
                EvalItem(
                    payload=text,
                    seed=self.base_seed + p_id * 9973,
                    group=0,
                    gold=self.gold,
                    variant=p_id,
                )
            )
        return out

    def accuracy_view(self) -> "QueryNeighborhood":

        return QueryNeighborhood(
            paraphrases=self.paraphrases[:1],
            base_seed=self.base_seed, gold=self.gold,
        )

    @property
    def n_groups(self) -> int:
        return 1
