import json
import os
import re

from datasets import DatasetDict, load_from_disk
from moo_mcts.concurrency import parallel_map
from moo_mcts.logging_util import get_logger

from .hf_dataset import HFDatasetSpec

log = get_logger("paraphrase")

_SEP = "===PARAPHRASE-SEPARATOR==="

_PARAPHRASE_PROMPT = (
    "You are rephrasing a problem statement to create paraphrases for robustness "
    "testing of a solver. Produce {n} DISTINCT paraphrases of the problem below.\n\n"
    "STRICT RULES:\n"
    "- Preserve the EXACT meaning. Keep every quantity, number, variable, name, and "
    "condition identical so the correct answer is unchanged.\n"
    "- Keep all mathematical notation (e.g. LaTeX such as \\frac or \\sqrt) verbatim.\n"
    "- Only vary wording, phrasing, and sentence order.\n"
    "- Do NOT solve the problem, add hints, or add/remove information.\n"
    "- Each paraphrase must stand alone and be answerable on its own.\n\n"
    "Problem:\n{problem}\n\n"
    "Output ONLY the {n} paraphrases. Separate consecutive paraphrases with a line "
    "containing exactly:\n" + _SEP + "\n"
    "Do not number them, add labels, quote them, or include any other text."
)


def _parse_paraphrase_list(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Strip a surrounding markdown code fence if the model added one.
    fence = re.match(r"^```[a-zA-Z]*\s*\n(.*)\n```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    if _SEP in text:
        parts = text.split(_SEP)
    else:
        # Fallback 1: a well-formed JSON list (model ignored the delimiter but
        # happened to escape correctly, or returned dicts).
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
            if isinstance(data, list):
                out: list[str] = []
                for d in data:
                    if isinstance(d, str):
                        out.append(d.strip())
                    elif isinstance(d, dict):
                        v = d.get("text") or d.get("paraphrase") or d.get("problem") or ""
                        if v:
                            out.append(str(v).strip())
                return [s for s in out if s]
        # Fallback 2: split on blank lines between paragraphs.
        parts = re.split(r"\n\s*\n", text)

    out = []
    for p in parts:
        s = p.strip()
        # Drop a leading list marker the model may have added ("1.", "2)", "- ").
        s = re.sub(r"^(?:\d+[.)]|[-*])\s+", "", s)
        if s:
            out.append(s)
    return out


class Paraphraser:

    def __init__(self, client, n: int = 3, temperature: float = 0.9):
        self.client = client
        self.n = int(n)
        self.temperature = temperature

    def paraphrase(self, text: str) -> list[str]:
        """Return up to `n` distinct paraphrases of `text` (never the original verbatim)."""
        if self.n <= 0 or not text.strip():
            return []
        prompt = _PARAPHRASE_PROMPT.format(n=self.n, problem=text)
        resp = self.client.complete(prompt, temperature=self.temperature)
        seen: set[str] = set()
        out: list[str] = []
        for v in _parse_paraphrase_list(resp):
            key = v.strip()
            if not key or key == text.strip() or key in seen:
                continue
            seen.add(key)
            out.append(v)
        log.debug("paraphrased -> %d variants (requested %d)", len(out), self.n)
        return out[: self.n]


def augment_dataset_with_paraphrases(
    spec: HFDatasetSpec, paraphraser: Paraphraser, in_path: str , out_path: str,
) -> tuple[str, list[list[str]]]:

    in_path = in_path or spec.path
    out_path = out_path or in_path.rstrip("/") + "_paraphrase"
    if os.path.abspath(out_path) == os.path.abspath(in_path):
        raise ValueError(
            f"out_path must differ from in_path ({in_path!r}) so the original dataset "
            "is not overwritten; pick a sibling dir like <dataset>_paraphrase"
        )

    ds = load_from_disk(in_path)
    if isinstance(ds, DatasetDict):
        ds = ds[next(iter(ds))]

    problems = [str(p) for p in ds[spec.problem_col]]
    log.info("paraphrasing %d problems (n=%d each)", len(problems), paraphraser.n)
    paras = parallel_map(paraphraser.paraphrase, problems)

    if spec.paraphrase_col in ds.column_names:
        ds = ds.remove_columns(spec.paraphrase_col)
    ds = ds.add_column(spec.paraphrase_col, paras)

    ds.save_to_disk(out_path)
    log.info("wrote paraphrase-augmented dataset (column %r) -> %s",
             spec.paraphrase_col, out_path)
    return out_path, paras
