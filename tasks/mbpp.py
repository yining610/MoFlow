"""Adapted from AFlow's MBPP benchmark code
"""
import ast
import json
import warnings
from typing import Optional, Tuple

warnings.filterwarnings("ignore", category=SyntaxWarning)

from moo_mcts.backends.sandbox_executor import run_grader
from moo_mcts.backends.task_profile import TaskProfile
from moo_mcts.logging_util import get_logger

from .hf_dataset import HFDatasetSpec

log = get_logger("mbpp")

DEFAULT_PATH = "data/mbpp"

PASS = "PASS"
FAIL = "FAIL"

def _syntax_check(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        return False


def code_extract(text: str) -> str:

    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i:j + 1])
            if _syntax_check(current_lines):
                current_length = sum(1 for line in lines[i:j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)
    return "\n".join(lines[longest_line_pair[0]:longest_line_pair[1] + 1])


def _sanitize_with_ast(code: str, entrypoint: Optional[str] = None) -> str:
    """A function that uses Python's built-in ast module"""

    tree = ast.parse(code)
    imports: list[str] = []
    definitions: list[tuple[str, str]] = []  # (name, source)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, ast.FunctionDef):
            definitions.append((node.name, ast.unparse(node)))
        elif isinstance(node, ast.ClassDef):
            definitions.append((node.name, ast.unparse(node)))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions.append((target.id, ast.unparse(node)))

    if entrypoint:
        dependencies: dict[str, set[str]] = {name: set() for name, _ in definitions}
        for name, code_str in definitions:
            for subnode in ast.walk(ast.parse(code_str)):
                if isinstance(subnode, ast.Name) and subnode.id in dependencies:
                    dependencies[name].add(subnode.id)

        reachable: set[str] = set()

        def _dfs(name: str) -> None:
            if name in reachable:
                return
            reachable.add(name)
            for dep in dependencies.get(name, []):
                _dfs(dep)

        if entrypoint in dependencies:
            _dfs(entrypoint)
        kept = [src for name, src in definitions if name in reachable]
    else:
        kept = [src for _, src in definitions]

    return "\n".join(imports + kept)


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    code = code_extract(code)
    try:
        return _sanitize_with_ast(code, entrypoint)
    except Exception:
        return code  # if parsing fails, return the extracted code unchanged

_CHECK_TIMEOUT_S = 15
_SANDBOX_WALL_TIMEOUT_S = _CHECK_TIMEOUT_S + 10  # covers exec(solution)+exec(test)+check()

# Grading harness run INSIDE the isolated subprocess (moo_mcts sandbox). It reads a JSON
# payload {solution, test, entry_point} from stdin and writes [status, message] to
# result.json.
_GRADER_HARNESS = r'''
import json
import os
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

CHECK_TIMEOUT_S = 15
PASS = "PASS"
FAIL = "FAIL"


def write_result(status, message):
    with open(os.path.join(os.getcwd(), "result.json"), "w", encoding="utf-8") as fh:
        json.dump([status, message], fh)


class _TimeoutError(Exception):
    pass


def run_with_timeout(func, timeout):
    result = []
    stop_event = threading.Event()

    def target():
        try:
            result.append(func())
        except BaseException as e:  # capture SystemExit too, else a bare exit -> false PASS
            result.append(e)
        finally:
            stop_event.set()

    # daemon so a runaway check dies with this (short-lived) subprocess
    threading.Thread(target=target, daemon=True).start()
    if not stop_event.wait(timeout):
        raise _TimeoutError("Function execution timed out")
    if not result:
        return None
    if isinstance(result[0], BaseException):
        raise result[0]
    return result[0]


def format_failure(exc, sol_path, test_path):
    """A FAIL message that names the failing assert / offending source line."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if f.filename in (sol_path, test_path)]
    label = "%s: %s" % (type(exc).__name__, exc) if str(exc) else type(exc).__name__
    if not frames:
        return "Error: %s" % label
    lines = []
    for f in frames:
        where = "solution" if f.filename == sol_path else "test"
        lines.append("  %s line %d: %s" % (where, f.lineno, (f.line or "").strip()))
    return "Error: %s\nFailing line:\n%s" % (label, "\n".join(lines))


def main():
    payload = json.loads(sys.stdin.read())
    solution = payload["solution"]
    test = payload["test"]
    entry_point = payload["entry_point"]

    # Write to real files so tracebacks resolve the offending source line via linecache.
    sol_path = os.path.join(os.getcwd(), "solution.py")
    test_path = os.path.join(os.getcwd(), "tests.py")
    with open(sol_path, "w", encoding="utf-8") as fh:
        fh.write(solution)
    with open(test_path, "w", encoding="utf-8") as fh:
        fh.write(test)

    global_dict = {
        "math": __import__("math"),
        "hashlib": __import__("hashlib"),
        "re": __import__("re"),
        "List": List,
        "Dict": Dict,
        "Tuple": Tuple,
        "Optional": Optional,
        "Any": Any,
    }

    try:
        exec(compile(solution, sol_path, "exec"), global_dict)
        if entry_point not in global_dict:
            write_result(FAIL, "Error: function `%s` is not defined in the solution." % entry_point)
            return
        exec(compile(test, test_path, "exec"), global_dict)
        check = global_dict["check"]
        run_with_timeout(check, CHECK_TIMEOUT_S)
        write_result(PASS, "The solution passed all test cases.")
    except _TimeoutError:
        write_result(FAIL, "Execution timed out. Please check if your solution contains "
                           "infinite loops or overly time-consuming operations.")
    except BaseException as exc:
        write_result(FAIL, format_failure(exc, sol_path, test_path))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # last resort: always leave a verdict behind
        try:
            write_result("FAIL", "Error: grader harness crashed: %s" % exc)
        except BaseException:
            pass
'''


def check_solution(solution: str, test: str, entry_point: str) -> Tuple[str, str]:
    """Run one MBPP grading round in an isolated subprocess; returns (PASS|FAIL, message).

    The candidate is sanitized here (pure AST, no code execution) and then executed and
    graded inside a fresh, isolated Python subprocess under a hard wall-clock timeout.
    The PASS/FAIL rule is identical to the reference in-process grader; only the execution
    is isolated and the timeout actually kills runaway code.
    """
    solution = sanitize(code=solution, entrypoint=entry_point)
    payload = json.dumps(
        {"solution": solution, "test": test, "entry_point": entry_point}
    )
    res = run_grader(_GRADER_HARNESS, payload, timeout_s=_SANDBOX_WALL_TIMEOUT_S)

    if res.verdict is not None:
        try:
            status, message = res.verdict[0], res.verdict[1]
        except Exception:
            status, message = FAIL, f"Error: malformed grader verdict: {res.verdict!r}"
    elif res.timed_out:
        status, message = (
            FAIL,
            "Execution timed out. Please check if your solution contains infinite "
            "loops or overly time-consuming operations.",
        )
    else:
        status, message = FAIL, f"Error: {res.error}"

    # Full candidate + verdict, so a DEBUG run shows exactly what was graded and why.
    log.debug(
        "grade=%s entry_point=%s\n----- candidate solution -----\n%s\n"
        "----- execution result -----\n%s",
        status, entry_point, solution, message,
    )
    return status, message

def _entry_point_and_signature(
    code: str, test_list: list[str]
) -> tuple[Optional[str], Optional[str]]:
    try:
        tree = ast.parse(code)
    except Exception:
        return None, None

    funcs = [n for n in ast.iter_child_nodes(tree) if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return None, None

    called: set[str] = set()
    for assert_src in (test_list or []):
        try:
            atree = ast.parse(str(assert_src))
        except Exception:
            continue
        for node in ast.walk(atree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)

    entry = next((fn for fn in funcs if fn.name in called), funcs[-1])
    signature = f"def {entry.name}({ast.unparse(entry.args)}):"
    return entry.name, signature


def _build_test(test_list: list[str]) -> str:

    lines = ["def check():"]
    body: list[str] = []
    for assert_stmt in (test_list or []):
        for ln in str(assert_stmt).splitlines():
            body.append("    " + ln)
    lines.extend(body or ["    pass"])
    return "\n".join(lines) + "\n"


def _mbpp_preprocess(ds):
    """Derive the JSON grading blob for each row."""

    def _row(row):
        entry_point, signature = _entry_point_and_signature(
            row.get("code") or "", row.get("test_list") or []
        )
        if not entry_point:
            return {"grading": ""}
        grading = json.dumps({
            "entry_point": entry_point,
            "test": _build_test(row.get("test_list") or []),
            "signature": signature,
        })
        return {"grading": grading}

    ds = ds.map(_row)
    return ds.filter(lambda r: bool(r.get("grading")))


def mbpp_checker(solution_text: str, item) -> bool:
    gold = item.gold
    if not gold:
        return False
    status, _msg = check_solution(solution_text, gold["test"], gold["entry_point"])
    return status == PASS


def run_public_tests(solution: str, gold: dict) -> tuple[bool, str]:
    """Execute the item's public tests on a candidate; return (passed, feedback).
    """
    status, msg = check_solution(solution, gold["test"], gold["entry_point"])
    return status == PASS, msg


def _mbpp_public_tester(gold):
    if not isinstance(gold, dict) or "test" not in gold or "entry_point" not in gold:
        return None
    return lambda solution: run_public_tests(solution, gold)


def _mbpp_vote_key(text: str) -> str:
    if not text:
        return ""
    try:
        return sanitize(text).strip()
    except Exception:
        return ""


def _mbpp_render_payload(problem: str, gold) -> str:
    signature = str(gold.get("signature") or "") if isinstance(gold, dict) else ""
    return f"{problem}\n\n{signature}" if signature else problem


MBPP_PROFILE = TaskProfile(
    description=(
        "You are an expert Python programmer. Read the task and implement a correct, "
        "efficient solution."
    ),
    answer_format=(
        "Return the complete function definition matching the signature shown at the "
        "end of the task, inside a single ```python code block. Include any imports the "
        "function needs. Do not include tests, example usage, or explanatory prose after "
        "the code."
    ),
    answer_key=_mbpp_vote_key,

    operators=("Generate", "Ensemble", "ReviewRevise", "TestCode"),
    cost_ref_tokens=20_000.0,
    latency_ref_calls=8.0,
)


MBPP_SANITIZED = HFDatasetSpec(
    name="mbpp",
    path=DEFAULT_PATH,
    problem_col="prompt",           # natural-language description (paraphrased as-is)
    answer_col="grading",           # JSON {entry_point, test, signature}; gold_cast decodes it
    index_col="task_id",
    gold_cast=json.loads,
    checker=mbpp_checker,
    task_profile=MBPP_PROFILE,
    default_val_size=20,            # 20 validate / 50 held-out test (seeded draws from all 427)
    default_test_size=50,
    hf_id="Muennighoff/mbpp",       # raw dataset source on the HF hub
    hf_config="sanitized",          # the sanitized subset (427 problems, one "test" split)
    hf_split="test",
    preprocess=_mbpp_preprocess,    # derive the grading blob from code/test_list/test_imports
    render_payload=_mbpp_render_payload,  # append the signature to every variant at load
    public_tester=_mbpp_public_tester,    # expose per-item public tests to the TestCode operator
    paraphrase_path="data/mbpp_paraphrase",  # source the frozen split is built from
    split_dir="data/mbpp_splits",
)
