"""Local code-execution sandbox for code-producing operators (Programmer/Test).
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from ..logging_util import get_logger

log = get_logger("sandbox")

@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    wall_ms: float = 0.0


def run_code(source: str, stdin: str = "", timeout_s: float = 5.0) -> SandboxResult:
    """Execute `source` as a Python script in a fresh temp dir under a timeout."""
    start = time.monotonic()
    log.debug("run_code: source:\n%s", _preview(source))
    with tempfile.TemporaryDirectory(prefix="moo_sbx_") as workdir:
        script = f"{workdir}/prog.py"
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(source)
        rc, out, err, timed_out = _run_isolated(
            [sys.executable, "-I", script],  # -I: isolated, ignore env/user site
            stdin=stdin, timeout_s=timeout_s, cwd=workdir,
        )

    wall_ms = (time.monotonic() - start) * 1000.0
    if timed_out:
        err = err + "\n[sandbox] timeout"
        log.debug("run_code: TIMEOUT after %.0fms (limit=%ss); process group killed", wall_ms, timeout_s)
        return SandboxResult(ok=False, stdout=_trunc(out), stderr=_trunc(err), wall_ms=wall_ms)
    if rc is None:  # spawn/other sandbox failure
        log.debug("run_code: sandbox error after %.0fms: %s", wall_ms, err)
        return SandboxResult(ok=False, stdout=_trunc(out), stderr=_trunc(err), wall_ms=wall_ms)
    ok = rc == 0
    log.debug("run_code: %s (returncode=%s, wall=%.0fms)", "ran OK" if ok else "FAILED", rc, wall_ms)
    log.debug("run_code: stdout:\n%s", _preview(out))
    log.debug("run_code: stderr:\n%s", _preview(err))
    return SandboxResult(ok=ok, stdout=_trunc(out), stderr=_trunc(err), wall_ms=wall_ms)


@dataclass
class GraderResult:
    verdict: object = None
    timed_out: bool = False
    error: str = ""


def run_grader(harness_source: str, payload: str, timeout_s: float = 25.0) -> GraderResult:
    """Run a grading harness in a fresh, isolated Python subprocess.
    """
    start = time.monotonic()
    log.debug("run_grader: payload:\n%s", _preview(payload))
    with tempfile.TemporaryDirectory(prefix="moo_grade_") as workdir:
        script = os.path.join(workdir, "harness.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(harness_source)
        result_path = os.path.join(workdir, "result.json")
        rc, _out, err, timed_out = _run_isolated(
            [sys.executable, "-I", script],  # -I: isolated, ignore env/user site
            stdin=payload, timeout_s=timeout_s, cwd=workdir,
        )
        wall_ms = (time.monotonic() - start) * 1000.0
        if timed_out:
            log.debug("run_grader: hard timeout after %ss (%.0fms wall); process group killed",
                      timeout_s, wall_ms)
            return GraderResult(timed_out=True)
        if rc is None:
            log.debug("run_grader: spawn error: %s", err)
            return GraderResult(error=err or "[sandbox] spawn error")

        if os.path.exists(result_path):
            try:
                with open(result_path, encoding="utf-8") as fh:
                    verdict = json.load(fh)
                log.debug("run_grader: verdict=%r (rc=%s, wall=%.0fms)", verdict, rc, wall_ms)
                return GraderResult(verdict=verdict)
            except Exception as e:
                return GraderResult(error=f"[sandbox] malformed verdict: {e}")
        log.debug("run_grader: NO VERDICT (rc=%s, wall=%.0fms); stderr:\n%s",
                  rc, wall_ms, _preview(err))
        return GraderResult(
            error=(
                f"[sandbox] harness produced no verdict (rc={rc}); "
                f"stderr: {err.strip()[-500:]}"
            )
        )


def _killpg(pgid: int) -> None:
    """SIGKILL an entire process group; ignore if it's already gone."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _run_isolated(argv, *, stdin: str = "", timeout_s: float, cwd: str):
    """Run `argv` as a subprocess in its OWN process group; on timeout SIGKILL the whole
    group so forked grandchildren (e.g. multiprocessing workers) die with it instead of
    orphaning and pinning CPU. Returns (returncode|None, stdout, stderr, timed_out).
    """
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            start_new_session=True,  # child leads a new group; pgid == child pid
        )
    except Exception as e:  # pragma: no cover - defensive (spawn failure)
        return None, "", f"[sandbox] {e}", False

    try:
        out, err = proc.communicate(input=stdin, timeout=timeout_s)
        return proc.returncode, _as_text(out), _as_text(err), False
    except subprocess.TimeoutExpired:
        _killpg(proc.pid)  # child still alive -> pgid == pid; kills the whole tree
        try:
            out, err = proc.communicate(timeout=5)
        except Exception:
            out, err = "", ""
        return proc.returncode, _as_text(out), _as_text(err), True
    except Exception as e:  # pragma: no cover - defensive
        _killpg(proc.pid)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return None, "", f"[sandbox] {e}", False


def _as_text(s) -> str:
    if s is None:
        return ""
    if isinstance(s, bytes):
        return s.decode("utf-8", errors="replace")
    return s


def _preview(s, limit: int = 500) -> str:
    s = _as_text(s)
    return s if len(s) <= limit else s[:limit] + f"\n[...+{len(s) - limit} chars]"


def _trunc(s: str, limit: int = 8192) -> str:
    return s if len(s) <= limit else s[:limit] + "\n[...truncated]"
