from dataclasses import dataclass, field

from moo_mcts.reward.evalset import QueryNeighborhood, TaskValidationSplit


@dataclass
class Task:

    name: str
    examples: list[tuple]
    seeds: int = 1

    def validation_split(self, n_paraphrases: int = 1) -> TaskValidationSplit:
        return TaskValidationSplit(
            examples=self.examples, seeds=self.seeds, n_paraphrases=n_paraphrases
        )


@dataclass
class Query:
    """A single incoming query + its paraphrase neighborhood."""

    text: str
    paraphrases: list[str] = field(default_factory=list)
    gold: object = None  # gold answer for grading

    def neighborhood(self, m: int = None) -> QueryNeighborhood:
        # paraphrase 0 is always the original query
        para = [self.text] + list(self.paraphrases)
        if m is not None:
            para = para[:m] if m <= len(para) else para + [self.text] * (m - len(para))
        return QueryNeighborhood(paraphrases=para, gold=self.gold)
