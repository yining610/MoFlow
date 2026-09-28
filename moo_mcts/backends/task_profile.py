from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class TaskProfile:

    description: str = (
        "You are a careful, expert problem solver. Read the input and work toward "
        "the best possible answer."
    )
    answer_format: str = (
        "End your response with the final answer on its own last line, written as "
        "'Answer: <your answer>'."
    )
    answer_key: Callable[[str], str] = None
    operators: tuple[str, ...] = None
    cost_ref_tokens: float = None
    latency_ref_calls: float = None


DEFAULT_PROFILE = TaskProfile()
