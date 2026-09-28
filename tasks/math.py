"""Adapted from AFlow's MATH evaluation code.
"""

import re
from math import isclose
from sympy import N, simplify
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex

from moo_mcts.backends.task_profile import TaskProfile

from .hf_dataset import HFDatasetSpec

DEFAULT_PATH = "data/math"

_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<!\d)[.!?]\s+")


def extract_boxed_answer(text: str) -> str | None:
    if not text:
        return None
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    return sentences[-1] if sentences else None


def gold_answer(solution: str) -> str:
    boxed = extract_boxed_answer(solution)
    return boxed if boxed is not None else str(solution).strip()


def _parse_digits(num):
    s = re.sub(",", "", str(num))
    try:
        return float(s)
    except ValueError:
        if s.endswith("%"):
            s = s[:-1].rstrip("\\")
            try:
                return float(s) / 100
            except ValueError:
                pass
    return None


def _is_digit(num) -> bool:
    return _parse_digits(num) is not None


def _symbolic_equal(a: str, b: str) -> bool:

    def _parse(s: str):
        for f in (parse_latex, parse_expr):
            try:
                return f(s)
            except Exception:
                continue
        return s

    pa, pb = _parse(a), _parse(b)
    try:
        if simplify(pa - pb) == 0:
            return True
    except Exception:
        pass
    try:
        if isclose(float(N(pa)), float(N(pb)), abs_tol=1e-3):
            return True
    except Exception:
        pass
    return False


def math_equal(prediction, reference) -> bool:
    if prediction is None or reference is None:
        return False
    if str(prediction) == str(reference):
        return True
    try:
        if _is_digit(prediction) and _is_digit(reference):
            return isclose(_parse_digits(prediction), _parse_digits(reference), abs_tol=1e-3)
    except Exception:
        pass
    try:
        return _symbolic_equal(str(prediction), str(reference))
    except Exception:
        pass
    return False


def math_checker(solution_text: str, item) -> bool:
    if item.gold is None:
        return False
    pred = extract_boxed_answer(solution_text)
    return math_equal(pred, item.gold)


def _math_vote_key(text: str) -> str:
    ans = extract_boxed_answer(text)
    return ans.strip() if ans is not None else ""


def _is_level5(row: dict) -> bool:
    return str(row.get("level", "")).strip() == "Level 5"


MATH_PROFILE = TaskProfile(
    description=(
        "You are an expert competition mathematician. Solve the problem carefully, "
        "reasoning step by step."
    ),
    answer_format=(
        "Put your final answer in \\boxed{...} on the last line. Give it in exact, "
        "simplest form (reduced fraction, exact radical, or integer), not a decimal "
        "approximation."
    ),
    answer_key=_math_vote_key,
    operators=("Generate", "Ensemble", "Programmer", "StepProgrammer"),
    cost_ref_tokens=70_000.0,
    latency_ref_calls=10.0,
)


MATH_L5 = HFDatasetSpec(
    name="math",
    path=DEFAULT_PATH,
    problem_col="problem",
    answer_col="solution",        # gold_cast reduces this to the boxed answer
    gold_cast=gold_answer,
    checker=math_checker,
    task_profile=MATH_PROFILE,
    default_val_size=20,          # 150 Level-5 problems -> 20 validate / 50 held-out test
    default_test_size=50,         # None = all remaining become held-out test
    hf_id="qwedsacf/competition_math",  # raw dataset source on the HF hub
    hf_split="train",
    row_filter=_is_level5,        # keep only Level 5 problems
    max_examples=150,             # deterministic cap on the Level-5 pool
    paraphrase_path="data/math_paraphrase",  # source the frozen split is built from
    split_dir="data/math_splits",            # canonical frozen val/test split (with paraphrases)
)
