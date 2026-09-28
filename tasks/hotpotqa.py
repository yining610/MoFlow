import re
import string
from collections import Counter

from moo_mcts.backends.task_profile import TaskProfile

from .hf_dataset import HFDatasetSpec

DEFAULT_PATH = "data/hotpotqa"

_ANSWER = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def extract_answer(text: str) -> str:
    """Pull the concise final answer from free-form output.
    """
    if not text:
        return ""
    matches = _ANSWER.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def hotpot_checker(solution_text: str, item) -> float:
    """Continuous token-F1 of the predicted answer against the gold answer."""
    if item.gold is None:
        return 0.0
    return f1_score(extract_answer(solution_text), str(item.gold))


def _hotpot_vote_key(text: str) -> str:
    """Group answers by their normalized final answer for self-consistency voting."""
    return normalize_answer(extract_answer(text))


HOTPOT_PROFILE = TaskProfile(
    description=(
        "You are an expert at multi-hop question answering. Read the provided context "
        "passages carefully and reason step by step to answer the question, using only "
        "information supported by the context."
    ),
    answer_format=(
        "End your response with the final answer on its own last line, written as "
        "'Answer: <answer>'. The answer must be concise and direct (a short phrase, "
        "name, entity, or yes/no) with no explanation."
    ),
    answer_key=_hotpot_vote_key,
    operators=("Generate", "Ensemble", "Custom", "Decompose", "EvidenceSelect", "GroundCheck"),
    cost_ref_tokens=20_000.0,
    latency_ref_calls=8.0,
)


HOTPOT_HARD = HFDatasetSpec(
    name="hotpotqa",
    path=DEFAULT_PATH,
    problem_col="question",
    answer_col="answer",
    index_col="id",
    gold_cast=str,
    checker=hotpot_checker,       # returns a float F1 in [0,1] (not a bool)
    task_profile=HOTPOT_PROFILE,
    default_val_size=20,          # 70 hard problems
    default_test_size=50,
    hf_id="hotpotqa/hotpot_qa",   # raw dataset source on the HF hub
    hf_config="distractor",       # distractor setting: gold + distractor context provided
    hf_split="validation",        # dev split (all hard); prep filters level == "hard"
    paraphrase_path="data/hotpotqa_paraphrase",  # unused (paraphrases live in the split jsonl)
    split_dir="data/hotpotqa_splits",            # canonical frozen val/test split
)
