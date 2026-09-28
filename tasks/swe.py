
import hashlib
import json
import os
import re
import threading
import time
from typing import Optional

from moo_mcts.backends.task_profile import TaskProfile
from moo_mcts.logging_util import get_logger

from .hf_dataset import HFDatasetSpec

log = get_logger("swe")

DEFAULT_PATH = "data/swe_lite"

SWE_DATASET = os.environ.get("SWE_DATASET", "SWE-bench/SWE-bench_Lite")

_EVAL_TIMEOUT_S = int(os.environ.get("SWE_EVAL_TIMEOUT", "1800"))
_DEFAULT_SERVER_URL = "http://127.0.0.1:8100"


_SUBMIT_MAX_ATTEMPTS = int(os.environ.get("SWE_SUBMIT_RETRIES", "6"))   # ~75s of backoff total
_SUBMIT_BACKOFF_S = float(os.environ.get("SWE_SUBMIT_BACKOFF", "5.0"))  # linear: 5s,10s,...
_POLL_MAX_CONN_ERRORS = int(os.environ.get("SWE_POLL_MAX_CONN_ERRORS", "6"))  # consecutive


class SWEServerUnreachable(RuntimeError):
    """The grading server was unreachable after retries.

    Raised (not returned as None) so a mid-run server death halts the run loudly instead of
    silently counting every subsequent patch as unresolved.
    """

_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*\n(?P<body>.*?)```", re.DOTALL)
_DIFF_START_RE = re.compile(r"^(diff --git |--- )", re.MULTILINE)
_HUNK_HDR_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _looks_like_diff(text: str) -> bool:
    return ("diff --git " in text) or ("@@ " in text) or text.lstrip().startswith("--- ")


def extract_patch(text: str) -> str:
    """Pull a unified git diff out of free-form model output.
    """
    if not text:
        return ""

    fenced = list(_FENCE_RE.finditer(text))
    # 1) explicitly-tagged diff/patch fence (last one wins, mirroring answer-at-the-end).
    for m in reversed(fenced):
        if m.group("lang").lower() in {"diff", "patch"}:
            return _recount_hunks(_ensure_nl(m.group("body")))
    # 2) any fence whose body looks like a diff.
    for m in reversed(fenced):
        if _looks_like_diff(m.group("body")):
            return _recount_hunks(_ensure_nl(m.group("body")))
    # 3) unfenced: take from the first diff header to the end.
    m = _DIFF_START_RE.search(text)
    if m:
        return _recount_hunks(_ensure_nl(text[m.start():]))
    # 4) give up and return the text as-is (server will report an empty/failed patch).
    return _ensure_nl(text)


def _ensure_nl(s: str) -> str:
    s = s.strip("\n")
    return (s + "\n") if s else ""


def _recount_hunks(patch: str) -> str:
    """Repair unified-diff hunk headers so a well-formed body can apply.

    LLMs reliably write plausible hunk *bodies* but botch the ``@@ -a,b +c,d @@`` line
    counts (and sometimes emit a bare ``@@`` with no numbers, or blank context lines with
    no leading space). ``git apply``/``patch`` then reject the whole file as unparseable
    ("malformed patch" / "unexpected end of file" / "Only garbage was found") before any
    context matching happens. We recompute every hunk's counts from its body (like
    ``git apply --recount``), normalize bare blank lines to context lines, and synthesize a
    syntactically valid header for a bare ``@@`` so the harness's ``patch --fuzz`` fallback
    can relocate it by context. Start offsets are left as written (fuzz repairs a wrong
    offset; a wrong *count* is fatal). Idempotent on a well-formed patch.

    Note: this only fixes diff *serialization*. Hunks whose context lines do not match the
    real file at the base commit still fail to apply -- that needs the model to see the
    actual source, which this cannot supply.
    """
    if "@@" not in patch:
        return patch
    trailing_nl = patch.endswith("\n")
    lines = patch.rstrip("\n").split("\n")
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.startswith("@@"):
            out.append(line)
            i += 1
            continue
        m = _HUNK_HDR_RE.match(line)
        old_start = int(m.group(1)) if m else 1
        new_start = int(m.group(3)) if m else 1
        section = m.group(5) if m else ""
        body, j = [], i + 1
        while j < n:
            lj = lines[j]
            if lj.startswith("@@") or lj.startswith("diff --git"):
                break
            if lj.startswith("--- ") and j + 1 < n and lines[j + 1].startswith("+++ "):
                break  # start of the next file's header (git-less multi-file diff)
            body.append(lj)
            j += 1
        old_count = new_count = 0
        norm = []
        for b in body:
            if b.startswith("\\"):          # "\ No newline at end of file" -- counts for neither side
                norm.append(b)
                continue
            head = b[:1]
            if head == "+":
                new_count += 1
            elif head == "-":
                old_count += 1
            elif head == " ":
                old_count += 1
                new_count += 1
            else:                            # bare blank / unprefixed line -> empty context line
                old_count += 1
                new_count += 1
                b = " " + b
            norm.append(b)
        out.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{section}")
        out.extend(norm)
        i = j
    result = "\n".join(out)
    return result + "\n" if trailing_nl else result


def _normalize_patch(patch: str) -> str:
    """Equivalence key for voting: keep only +/- content lines, drop volatile headers."""
    out = []
    for line in patch.split("\n"):
        line = line.rstrip()
        if line.startswith(("diff --git", "index ", "--- ", "+++ ", "@@")):
            continue
        if line.startswith(("+", "-")):
            out.append(line)
    return "\n".join(out).strip()


def _server_url() -> str:
    return (os.environ.get("SWE_SERVER_URL", "").strip() or _DEFAULT_SERVER_URL).rstrip("/")


def _parse_stdout(stdout: str) -> dict:
    """Extract resolved/total (and friends) from the harness stdout tail.
    """
    result = {"resolved": 0, "unresolved": 0, "error": 0, "total": 0, "empty": 0}
    patterns = {
        "resolved": r"Instances resolved:\s*(\d+)",
        "unresolved": r"Instances unresolved:\s*(\d+)",
        "total": r"Instances submitted:\s*(\d+)",
        "error": r"Instances with errors:\s*(\d+)",
        "empty": r"Instances with empty patches:\s*(\d+)",
    }
    for line in stdout.split("\n"):
        for key, pat in patterns.items():
            m = re.search(pat, line)
            if m:
                result[key] = int(m.group(1))
    return result


_cache: dict[str, Optional[dict]] = {}
_cache_lock = threading.Lock()
_thread_local = threading.local()

def _session():
    """A per-thread requests.Session with trust_env off (ignore proxy env for localhost)."""
    sess = getattr(_thread_local, "sess", None)
    if sess is None:
        import requests  # lazy: keep module import light + flask/swebench-free
        sess = requests.Session()
        sess.trust_env = False
        _thread_local.sess = sess
    return sess


def _submit(instance_id: str, patch: str) -> Optional[str]:
    import requests  # lazy: for the exception types below
    payload = {
        "predictions": [{
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": "moo-mcts",
        }],
        "max_workers": 1,
        "dataset": SWE_DATASET,
    }
    last_exc: Optional[Exception] = None
    for attempt in range(1, _SUBMIT_MAX_ATTEMPTS + 1):
        try:
            resp = _session().post(f"{_server_url()}/evaluate", json=payload, timeout=60)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # server unreachable (down / restarting) -- retry with backoff before giving up.
            last_exc = e
            log.warning("[swe] submit unreachable for %s (attempt %d/%d): %s",
                        instance_id, attempt, _SUBMIT_MAX_ATTEMPTS, e)
            if attempt < _SUBMIT_MAX_ATTEMPTS:
                time.sleep(_SUBMIT_BACKOFF_S * attempt)
            continue
        except Exception as e:  # noqa: BLE001  (unexpected, non-reachability error)
            log.warning("[swe] submit error for %s: %s", instance_id, e)
            return None
        if resp.status_code not in (200, 202):
            log.warning("[swe] submit HTTP %s: %s", resp.status_code, resp.text[:200])
            return None
        return (resp.json() or {}).get("task_id")
    raise SWEServerUnreachable(
        f"SWE grading server {_server_url()} unreachable after {_SUBMIT_MAX_ATTEMPTS} submit "
        f"attempts ({last_exc}). Restart scripts/swe/run_swe_server.sh and --resume."
    )


def _poll(task_id: str, timeout: int) -> Optional[dict]:
    """Long-poll /result until SUCCESS/FAILURE or the deadline; returns the result dict."""
    import requests  # lazy
    deadline = time.monotonic() + timeout
    poll_interval = 5
    conn_errors = 0  # consecutive connection failures (reset on any HTTP response)
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        server_wait = min(20, max(int(remaining), 2))
        try:
            resp = _session().get(
                f"{_server_url()}/result/{task_id}",
                params={"timeout": server_wait},
                timeout=server_wait + 10,
            )
        except requests.exceptions.ConnectionError as e:
            # server went away mid-poll: bound consecutive failures, then halt loudly rather
            # than spin to the deadline and silently return None (-> scored as unresolved).
            conn_errors += 1
            if conn_errors >= _POLL_MAX_CONN_ERRORS:
                raise SWEServerUnreachable(
                    f"SWE grading server {_server_url()} unreachable during poll "
                    f"({conn_errors} consecutive connection errors, task {task_id[:12]}). "
                    f"Restart scripts/swe/run_swe_server.sh and --resume."
                ) from e
            log.warning("[swe] poll connection error task=%s (%d/%d); retrying",
                        task_id[:12], conn_errors, _POLL_MAX_CONN_ERRORS)
            time.sleep(poll_interval)
            continue
        except requests.exceptions.Timeout as e:
            # long-poll timeout is normal while the harness is still grinding -- not a
            # reachability signal, so don't count it toward the unreachable threshold.
            log.debug("[swe] poll timeout task=%s: %s; retrying", task_id[:12], e)
            time.sleep(poll_interval)
            continue
        except Exception as e:  # noqa: BLE001
            log.warning("[swe] poll error task=%s: %s", task_id[:12], e)
            return None
        conn_errors = 0  # got an HTTP response -> server is alive
        if resp.status_code == 200:
            data = resp.json() or {}
            if data.get("state") == "FAILURE":
                log.warning("[swe] task %s FAILURE: %s", task_id[:12], str(data.get("error", ""))[:150])
                return None
            return data.get("result")
        if resp.status_code == 202:  # still running
            time.sleep(poll_interval)
            continue
        log.warning("[swe] result HTTP %s: %s", resp.status_code, resp.text[:150])
        return None
    log.warning("[swe] result polling timeout after %ss (task %s)", timeout, task_id[:12])
    return None


def _graded_verdict(instance_id: str, patch: str, timeout: int = _EVAL_TIMEOUT_S) -> Optional[dict]:
    """Submit a patch, poll the harness, and return the parsed verdict counts.
    """
    if not instance_id or not (patch or "").strip():
        return None

    key = f"{instance_id}_{hashlib.md5(patch.encode('utf-8', 'ignore')).hexdigest()[:10]}"
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    task_id = _submit(instance_id, patch)
    if not task_id:
        return None
    result = _poll(task_id, timeout=timeout)
    if result is None:
        return None

    parsed = _parse_stdout(result.get("stdout_tail", ""))
    if parsed["total"] == 0:
        log.warning("[swe] %s: total=0 (returncode=%s) -- harness produced no report",
                    instance_id, result.get("returncode"))
        return None
    with _cache_lock:
        _cache[key] = parsed
    log.info("[swe] %s: %s (%d/%d, task=%s)", instance_id,
             "RESOLVED" if parsed["resolved"] >= 1 else "unresolved",
             parsed["resolved"], parsed["total"], task_id[:12])
    return parsed


def grade_patch(instance_id: str, patch: str, timeout: int = _EVAL_TIMEOUT_S) -> Optional[float]:
    """Grade one instance's patch through the harness server.
    """
    verdict = _graded_verdict(instance_id, patch, timeout)
    if verdict is None:
        return None
    return float(verdict["resolved"]) / verdict["total"]


def _grade_feedback(instance_id: str, patch: str) -> tuple[bool, str]:
    """In-loop oracle for the PatchRepair operator: (resolved, actionable_feedback).

    Uses the SAME harness grade as the final checker -- SWE-bench has no separate public tests,
    so this IS the real pass/fail signal -- but categorizes the outcome (no patch / failed to
    apply / applies-but-unresolved / resolved) so the model can act on it during repair.
    """
    if not (patch or "").strip():
        return False, ("No unified git diff was found in your output. Produce exactly one "
                       "```diff code block containing a valid patch.")
    verdict = _graded_verdict(instance_id, patch)
    if verdict is None:
        return False, ("The patch could not be graded (unparseable diff, or the eval server was "
                       "unreachable). Ensure the output is a single valid unified git diff that "
                       "applies cleanly at the base commit.")
    if verdict["resolved"] >= 1:
        return True, "The patch resolves the issue: all required tests pass."
    if verdict.get("empty"):
        return False, ("The harness treated the patch as empty (no effective change after "
                       "normalization). Make sure the diff edits real source lines with correct "
                       "file paths.")
    if verdict.get("error"):
        return False, ("The patch FAILED TO APPLY at the base commit (git apply error). Re-derive "
                       "the diff: verify the file paths, the @@ hunk headers, and that the context "
                       "lines match the repository exactly at the base commit.")
    return False, ("The patch applies but does NOT resolve the issue -- the required tests still "
                   "fail. Re-examine the root cause and revise the fix; you may be editing the "
                   "wrong location or missing a case.")


def swe_checker(solution_text: str, item) -> bool:
    """True iff the model's patch resolves the instance (all FAIL_TO_PASS/PASS_TO_PASS pass)."""
    gold = item.gold
    if not isinstance(gold, dict):
        return False
    instance_id = gold.get("instance_id")
    if not instance_id:
        return False
    patch = extract_patch(solution_text)
    # grade_patch raises SWEServerUnreachable if the server is down -> let it propagate so the
    # run halts (checkpoint preserved) instead of scoring every ungraded patch as unresolved.
    score = grade_patch(instance_id, patch)
    if score is None:
        log.warning("[swe] %s: no verdict -> counted as unresolved (patch_empty=%s)",
                    instance_id, not patch.strip())
        return False
    return score >= 1.0


def _swe_vote_key(text: str) -> str:
    """Group identical patches for Ensemble self-consistency voting."""
    try:
        return _normalize_patch(extract_patch(text))
    except Exception:  # noqa: BLE001
        return ""


def _swe_make_tester(gold):
    """Build the per-item patch oracle consumed by apply/test operators (e.g. PatchRepair).
    """
    instance_id = gold.get("instance_id") if isinstance(gold, dict) else None

    def _tester(solution_text: str) -> tuple[bool, str]:
        if not instance_id:
            return False, "No instance id available to grade this patch."
        return _grade_feedback(instance_id, extract_patch(solution_text))

    return _tester


_SPLIT_DIR = "data/swe_lite_splits"
_file_context_cache = None


def _load_file_context() -> dict:
    """Lazy-load the offline oracle file-context cache ({instance_id -> source block}).
    """
    global _file_context_cache
    if _file_context_cache is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.environ.get("SWE_FILE_CONTEXT") or os.path.join(repo_root, _SPLIT_DIR, "file_context.json")
        try:
            with open(path) as fh:
                _file_context_cache = json.load(fh)
        except (OSError, ValueError):
            _file_context_cache = {}
    return _file_context_cache


def _swe_render_payload(problem: str, gold) -> str:
    """Prepend repo/commit context and append the relevant source at the base commit.
    """
    body = problem
    if isinstance(gold, dict) and gold.get("repo"):
        commit = str(gold.get("base_commit") or "")[:12]
        head = f"Repository: {gold['repo']}"
        if commit:
            head += f" (base commit {commit})"
        body = f"{head}\n\n{problem}"
    if isinstance(gold, dict):
        ctx = _load_file_context().get(gold.get("instance_id") or "")
        if ctx:
            body = f"{body}\n\n{ctx}"
    return body


def _swe_preprocess(ds):
    """Derive the JSON grading blob (swe_meta) for each row; drop empty problem statements."""

    def _row(row):
        return {"swe_meta": json.dumps({
            "instance_id": row.get("instance_id") or "",
            "repo": row.get("repo") or "",
            "base_commit": row.get("base_commit") or "",
            "version": str(row.get("version") or ""),
        })}

    ds = ds.map(_row)
    return ds.filter(lambda r: bool((r.get("problem_statement") or "").strip())
                     and bool(r.get("instance_id")))


SWE_PROFILE = TaskProfile(
    description=(
        "You are an expert software engineer resolving a real GitHub issue. Read the issue "
        "report for the named repository and produce a code patch that fixes the described "
        "bug or implements the requested behavior. Reason about which source file(s) and "
        "function(s) are responsible before writing the fix."
    ),
    answer_format=(
        "Return your fix as exactly one unified git diff inside a single ```diff code block, "
        "and nothing else after it. Use real headers: a `diff --git a/<path> b/<path>` line "
        "per file, `--- a/<path>` / `+++ b/<path>` lines, and `@@ ... @@` hunks with correct "
        "line context. Paths must be relative to the repository root. The patch must apply "
        "cleanly at the given base commit with `git apply`. Do not include tests, prose, or "
        "explanations."
    ),
    answer_key=_swe_vote_key,
    operators=("Generate", "Localize", "PatchRepair", "Ensemble", "Custom"),
    cost_ref_tokens=500_000.0,
    latency_ref_calls=40.0,
)


SWE_LITE = HFDatasetSpec(
    name="swe",
    path=DEFAULT_PATH,
    problem_col="problem_statement",   # the GitHub issue text (paraphrased as-is)
    answer_col="swe_meta",             # JSON {instance_id, repo, base_commit, version}; gold_cast decodes it
    index_col="instance_id",           # stable sort key before the seeded shuffle
    gold_cast=json.loads,
    checker=swe_checker,
    task_profile=SWE_PROFILE,
    default_val_size=20,               # 20 validate / 50 held-out test
    default_test_size=50,
    hf_id="SWE-bench/SWE-bench_Lite",    # raw dataset source on the HF hub (test split = 300 rows)
    hf_split="test",
    preprocess=_swe_preprocess,          # derive the swe_meta grading blob
    render_payload=_swe_render_payload,  # prepend repo/commit context at load
    public_tester=_swe_make_tester,      # patch oracle for PatchRepair (harness grade == the test)
    paraphrase_path="data/swe_lite_paraphrase",  # source the frozen split is built from
    split_dir="data/swe_lite_splits",
    max_examples=100,
)
