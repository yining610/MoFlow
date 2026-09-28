"""Generic HuggingFace dataset loader shared across tasks.
"""

import json
import os
from dataclasses import dataclass
from datasets import DatasetDict, load_from_disk, load_dataset
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from .benchmark import Query, Task

if TYPE_CHECKING:
    from moo_mcts.backends.task_profile import TaskProfile


def _identity(x: Any) -> Any:
    return x


def _render(spec: "HFDatasetSpec", text: str, gold: Any) -> str:
    """Render the model-facing payload from stored problem text + gold.
    """
    if spec.render_payload is None:
        return text
    return spec.render_payload(text, gold)

@dataclass
class HFDatasetSpec:

    name: str
    path: str
    problem_col: str
    answer_col: str
    checker: Callable[[str, Any], bool]
    index_col: str = None # optional column to sort by before shuffling
    gold_cast: Callable[[Any], Any] = _identity # e.g. int for AIME; default no-op
    task_profile: "TaskProfile" = None
    paraphrase_col: str = "paraphrases"
    default_val_size: int = 6
    default_test_size: int = None
    hf_id: str = None
    hf_config: str = None
    hf_split: str = "train"
    row_filter: Callable[[dict], bool] = None  # keep rows where this is True
    max_examples: int = None
    preprocess: Callable[[Any], Any] = None
    render_payload: Callable[[str, Any], str] = None
    public_tester: Callable[[Any], Any] = None
    paraphrase_path: str = None
    split_dir: str = None


def _split_dir(path: str) -> str:
    return os.path.join(path.rstrip("/"), "splits")


def dataset_exists(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


def download_hf_dataset(spec: HFDatasetSpec, out_path: str = None, split: str = None) -> str:

    if not spec.hf_id:
        raise ValueError(
            f"dataset {spec.name!r} has no hf_id; set it on the HFDatasetSpec to "
            "enable download (or place the dataset on disk at spec.path)"
        )
    out_path = out_path or spec.path
    ds = load_dataset(spec.hf_id, name=spec.hf_config, split=split or spec.hf_split)
    if isinstance(ds, DatasetDict):
        ds = ds[next(iter(ds))]

    if spec.preprocess is not None:
        ds = spec.preprocess(ds)

    missing = [c for c in (spec.problem_col, spec.answer_col) if c not in ds.column_names]
    if missing:
        raise ValueError(
            f"{spec.hf_id!r} is missing required column(s) {missing}; "
            f"available columns: {ds.column_names}"
        )

    if spec.row_filter is not None:
        ds = ds.filter(spec.row_filter)
    if spec.max_examples is not None and len(ds) > spec.max_examples:
        ds = ds.shuffle(seed=0).select(range(spec.max_examples))
        
    keep = [c for c in (spec.index_col, spec.problem_col, spec.answer_col) if c and c in ds.column_names]
    ds = ds.select_columns(keep)
    ds.save_to_disk(out_path)
    return out_path


def make_default_split(
    spec: HFDatasetSpec,
    path: str = None,
    val_size: int = None,
    seed: int = 0,
    out_dir: str = None,
    test_size: int = None,
) -> dict:
    """Freeze a reproducible val/test split, carrying paraphrases when the source has them.
    """

    src = path or spec.paraphrase_path or spec.path
    ds = load_from_disk(src)
    if isinstance(ds, DatasetDict):
        ds = ds[next(iter(ds))]
    if spec.index_col:
        ds = ds.sort(spec.index_col)
    ds = ds.shuffle(seed=seed)
    has_paraphrases = spec.paraphrase_col in ds.column_names

    # Keep only the columns the loaders need, so the split is self-contained.
    keep = [c for c in (spec.problem_col, spec.answer_col, spec.index_col, spec.paraphrase_col) if c and c in ds.column_names]
    rows = [
        {**{c: r[c] for c in keep}, "example_id": ex_id}
        for ex_id, r in enumerate(ds)
    ]
    n = len(rows)
    vs = max(1, min(val_size if val_size is not None else spec.default_val_size, n))
    val_rows = rows[:vs]
    ts_req = test_size if test_size is not None else spec.default_test_size
    if ts_req is None:
        test_rows = rows[vs:]
    else:
        ts = max(0, min(ts_req, n - vs))
        test_rows = rows[vs:vs + ts]

    out_dir = out_dir or spec.split_dir or _split_dir(src)
    os.makedirs(out_dir, exist_ok=True)
    for fname, recs in (("validate.jsonl", val_rows), ("test.jsonl", test_rows)):
        with open(os.path.join(out_dir, fname), "w") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": spec.name,
        "source_path": src,
        "source_fingerprint": getattr(ds, "_fingerprint", None),
        "seed": seed,
        "total": n,
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "unused": n - len(val_rows) - len(test_rows),
        "problem_col": spec.problem_col,
        "answer_col": spec.answer_col,
        "index_col": spec.index_col,
        "paraphrase_col": spec.paraphrase_col if has_paraphrases else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "split.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _read_split_jsonl(
    spec: HFDatasetSpec, fpath: str
) -> list[tuple[str, int, Any, list[str]]]:
    """Read a split file into (payload, example_id, gold, paraphrases) tuples.
    """
    out: list[tuple[str, int, Any, list[str]]] = []
    with open(fpath) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ex_id = int(rec.get("example_id", i))
            gold = spec.gold_cast(rec[spec.answer_col])
            payload = _render(spec, str(rec[spec.problem_col]), gold)
            paras = [_render(spec, str(p), gold) for p in (rec.get(spec.paraphrase_col) or [])]
            out.append((payload, ex_id, gold, paras))
    return out


def _resolve_split_dir(spec: HFDatasetSpec, path: str | None, split_dir: str | None) -> str:
    return split_dir or spec.split_dir or _split_dir(path or spec.path)


def load_default_split(
    spec: HFDatasetSpec, path: str = None, split_dir: str = None
) -> tuple[Task, list[tuple[str, int, Any, list[str]]]]:

    out_dir = _resolve_split_dir(spec, path, split_dir)
    val_path = os.path.join(out_dir, "validate.jsonl")
    test_path = os.path.join(out_dir, "test.jsonl")
    if not (os.path.isfile(val_path) and os.path.isfile(test_path)):
        raise FileNotFoundError(
            f"no frozen default split in {out_dir!r}; create it first with "
            "`python -m moo_mcts.cli data --task <name>` "
            "(or pass an explicit --val-size to use an on-the-fly split)"
        )
    val = _read_split_jsonl(spec, val_path)
    heldout = _read_split_jsonl(spec, test_path)
    task = Task(name=f"{spec.name}[val{len(val)}]", examples=val, seeds=1)
    return task, heldout


def load_hf_split(
    spec: HFDatasetSpec,
    path: str = None,
    val_size: int = None,
    seed: int = 0,
    split_dir: str = None,
) -> tuple[Task, list[tuple[str, int, Any, list[str]]]]:
    """Load a default val/test split from disk or create one on-the-fly from a HuggingFace dataset for the per-task mode,
    """

    # NOTE: if validation size is not specificied, we use the frozen default split (if it exists) for reproducibility
    if val_size is None:
        return load_default_split(spec, path, split_dir=split_dir)

    ds = load_from_disk(path or spec.paraphrase_path or spec.path)
    if isinstance(ds, DatasetDict):
        ds = ds[next(iter(ds))]
    if spec.index_col:
        ds = ds.sort(spec.index_col)
    ds = ds.shuffle(seed=seed)

    examples = []
    for ex_id, r in enumerate(ds):
        gold = spec.gold_cast(r[spec.answer_col])
        payload = _render(spec, str(r[spec.problem_col]), gold)
        paras = [_render(spec, str(p), gold) for p in (r.get(spec.paraphrase_col) or [])]
        examples.append((payload, ex_id, gold, paras))
    val_size = max(1, min(val_size, len(examples)))
    val = examples[:val_size]
    heldout = examples[val_size:]
    task = Task(name=f"{spec.name}[val{val_size}]", examples=val, seeds=1)
    return task, heldout


def load_split_queries(
    spec: HFDatasetSpec, split_dir: str = None, subset: str = "val",
    n_queries: int = None,
) -> list[Query]:
    """Load a frozen val/test split from disk and return it as a list of Query objects for the per-query mode.
    """

    out_dir = _resolve_split_dir(spec, None, split_dir)
    files = {
        "val": ["validate.jsonl"],
        "test": ["test.jsonl"],
        "all": ["validate.jsonl", "test.jsonl"],
    }.get(subset)
    if files is None:
        raise ValueError(f"subset must be one of val/test/all, got {subset!r}")

    records: list[tuple[str, int, Any, list[str]]] = []
    for fname in files:
        fpath = os.path.join(out_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"no frozen split file {fpath!r}; create it first with "
                "`python -m moo_mcts.cli data --task <name>`"
            )
        records.extend(_read_split_jsonl(spec, fpath))

    if n_queries is not None:
        records = records[: max(0, n_queries)]

    return [
        Query(text=payload, paraphrases=list(paras), gold=gold)
        for payload, _ex_id, gold, paras in records
    ]
