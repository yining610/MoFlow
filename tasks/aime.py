import re

from moo_mcts.backends.task_profile import TaskProfile

from .hf_dataset import HFDatasetSpec

DEFAULT_PATH = "data/aime_2026"

_BOXED = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
_INT = re.compile(r"-?\d+")


def extract_int_answer(text: str) -> int | None:
    """Extract the model's final integer answer from free-form output.

    Return the last \\boxed{N}, else the last standalone integer in the
    text. Returns None if no integer is present.
    """
    if not text:
        return None
    boxed = _BOXED.findall(text)
    if boxed:
        return int(boxed[-1])
    nums = _INT.findall(text)
    if nums:
        return int(nums[-1])
    return None


def aime_checker(solution_text: str, item) -> bool:
    if item.gold is None:
        return False
    pred = extract_int_answer(solution_text)
    return pred is not None and int(pred) == int(item.gold)


def _math_vote_key(text: str) -> str:
    """Group answers by their final integer for self-consistency voting."""
    n = extract_int_answer(text)
    return str(n) if n is not None else ""


MATH_PROFILE = TaskProfile(
    description=(
        "You are an expert competition mathematician. Solve the problem carefully, "
        "reasoning step by step."
    ),
    answer_format=(
        "Give the final integer answer (0-999) on the last line as \\boxed{ANSWER}."
    ),
    answer_key=_math_vote_key,
    operators=("Generate", "Ensemble", "Programmer", "StepProgrammer"),
    cost_ref_tokens=20_000.0,
    latency_ref_calls=8.0,
)


AIME_2026 = HFDatasetSpec(
    name="aime_2026",
    path=DEFAULT_PATH,
    problem_col="problem",
    answer_col="answer",
    index_col="problem_idx",
    gold_cast=int,
    checker=aime_checker,
    task_profile=MATH_PROFILE,
    default_val_size=10,  # 30 problems -> 10 validate / 20 held-out test
    hf_id="MathArena/aime_2026",  # raw dataset source on the HF hub
    paraphrase_path="data/aime_2026_paraphrase",  # source the frozen split is built from
    split_dir="data/aime_2026_splits",  # canonical frozen val/test split (with paraphrases)
)
