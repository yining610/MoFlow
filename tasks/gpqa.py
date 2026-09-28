import re

from moo_mcts.backends.task_profile import TaskProfile

from .hf_dataset import HFDatasetSpec

DEFAULT_PATH = "data/gpqa_diamond"

# "Answer: C", "Answer - c", "final answer is (D)", "the answer is B", etc.
_ANSWER_LINE = re.compile(
    r"answer\b[\s:*\-]*"
    r"(?:(?:is|was|of|should\s+be|would\s+be)\b[\s:*\-]*)?"
    r"\(?\s*([A-Da-d])\b",
    re.IGNORECASE,
)
# "\boxed{C}", "\boxed{(D)}", "\boxed{\text{C}}".
_BOXED = re.compile(r"\\boxed\{\s*\(?\s*(?:\\text\{)?\s*([A-Da-d])\s*\}?\s*\)?\s*\}")
# A standalone choice letter, optionally parenthesized: "(C)", "C.", "C)".
_STANDALONE = re.compile(r"(?<![A-Za-z])\(?([A-Da-d])\)?(?![A-Za-z])")


def extract_choice_answer(text: str) -> str | None:
    """Extract the model's final multiple-choice letter (A-D) from free-form output.
    """
    if not text:
        return None
    for pat in (_ANSWER_LINE, _BOXED):
        m = pat.findall(text)
        if m:
            return m[-1].upper()
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    m = _STANDALONE.findall(tail) or _STANDALONE.findall(text)
    if m:
        return m[-1].upper()
    return None


def gpqa_checker(solution_text: str, item) -> bool:
    """Case-insensitive exact match of the predicted choice letter to the gold."""
    if item.gold is None:
        return False
    pred = extract_choice_answer(solution_text)
    return pred is not None and pred.upper() == str(item.gold).strip().upper()


def _gpqa_vote_key(text: str) -> str:
    """Group answers by their chosen letter for self-consistency voting."""
    letter = extract_choice_answer(text)
    return letter if letter is not None else ""


def _normalize_gold(answer) -> str:
    return str(answer).strip().upper()


GPQA_PROFILE = TaskProfile(
    description=(
        "You are an expert scientist answering a graduate-level multiple-choice "
        "question in physics, chemistry, or biology. Read the question and all of the "
        "options carefully, and reason step by step to determine the single best answer."
    ),
    answer_format=(
        "End your response with the final answer on its own last line, written as "
        "'Answer: <letter>', where <letter> is exactly one of A, B, C, or D."
    ),
    answer_key=_gpqa_vote_key,
    operators=("Generate", "Ensemble", "ReviewRevise", "EliminateChoices"),
    cost_ref_tokens=70_000.0,
    latency_ref_calls=10.0,
)


GPQA_DIAMOND = HFDatasetSpec(
    name="gpqa_diamond",
    path=DEFAULT_PATH,
    problem_col="question",       # question text with the A-D options embedded inline
    answer_col="answer",          # single gold letter (A/B/C/D); gold_cast normalizes case
    gold_cast=_normalize_gold,
    checker=gpqa_checker,
    task_profile=GPQA_PROFILE,
    default_val_size=20,          # 198 problems -> 20 validate / 50 held-out test
    default_test_size=50,
    hf_id="fingertap/GPQA-Diamond",  # raw dataset source on the HF hub
    hf_split="test",                 # the dataset's only split
    paraphrase_path="data/gpqa_diamond_paraphrase",  # source the frozen split is built from
    split_dir="data/gpqa_diamond_splits",            # canonical frozen val/test split
)