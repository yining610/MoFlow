"""Real LLM backend (OpenAI-compatible endpoint, e.g. a LiteLLM proxy).
"""

import json
import os
import re
import threading
from contextlib import nullcontext
from dataclasses import replace

import backoff
import openai

from ..config import BackendConfig
from ..logging_util import get_logger
from ..objectives import ObjectiveSpec
from ..workflow import operators as ops
from ..workflow.edits import AddControl, AddOperator, Terminate, is_legal
from .base import EditProposal, ExecMeta
from .sandbox_executor import run_code
from .task_profile import DEFAULT_PROFILE, TaskProfile

log = get_logger("backend")


class MalformedResponseError(Exception):
    """The proxy returned something that isn't a usable chat completion -- e.g. a
    bare error string, or a response object with no `choices`.
    """


def _billed_total_tokens(resp) -> int:
    """Return the total tokens the proxy billed for this response, or 0 if unknown.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0
    total = getattr(usage, "total_tokens", None) or 0
    if total > 0:
        return int(total)
    pt = getattr(usage, "prompt_tokens", None) or 0
    ct = getattr(usage, "completion_tokens", None) or 0
    return int(pt + ct)  # 0 if both absent


def _estimate_tokens(*texts: str) -> int:
    """Rough token estimate (~4 chars/token) for responses the proxy didn't meter."""
    chars = sum(len(t) for t in texts if t)
    return max(1, chars // 4)


class OpenAICompatibleClient:
    def __init__(self, client, model: str, default_temperature: float = 0.7,
                 max_concurrency: int = None):
        self._client = client
        self.model = model
        self.default_temperature = default_temperature

        self._total_tokens = 0
        self._tok_lock = threading.Lock()

        self._sema = (
            threading.BoundedSemaphore(max_concurrency)
            if max_concurrency and max_concurrency > 0
            else None
        )

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @total_tokens.setter
    def total_tokens(self, value: int) -> None:
        with self._tok_lock:
            self._total_tokens = int(value)

    def complete(self, prompt: str, temperature: float = None) -> str:
        text, _ = self.complete_metered(prompt, temperature=temperature)
        return text

    @backoff.on_exception(backoff.expo, (openai.APIError, MalformedResponseError),
                          max_tries=10, max_time=600, max_value=60)
    def complete_metered(self, prompt: str, temperature: float = None, seed: int = None) -> tuple[str, int]:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.default_temperature if temperature is None else temperature,
        )
        if seed is not None:
            kwargs["seed"] = seed  # honored by OpenAI-compatible proxies for determinism

        log.debug("LLM call prompt: %s", prompt.replace("\n", " "))
        with self._sema if self._sema is not None else nullcontext():
            resp = self._client.chat.completions.create(**kwargs)

        choices = None if isinstance(resp, str) else getattr(resp, "choices", None)
        if not choices:
            snippet = (resp if isinstance(resp, str) else repr(resp))[:200]
            log.warning("malformed LLM response (retrying): %s", snippet)
            raise MalformedResponseError(snippet)

        text = choices[0].message.content or ""
        tokens = _billed_total_tokens(resp)
        if tokens <= 0:
            # Proxy returned a response without usage accounting
            tokens = _estimate_tokens(prompt, text)
            log.warning(
                "LLM response missing usage.total_tokens; estimating %d tokens "
                "(prompt+completion ~chars/4)", tokens,
            )
        with self._tok_lock:
            self._total_tokens += tokens
        log.debug("LLM call response: %s", text.replace("\n", " "))
        return text, tokens


def make_client(bcfg: BackendConfig) -> OpenAICompatibleClient:
    key = os.environ.get(bcfg.api_key_env)
    if not key:
        raise RuntimeError(
            f"missing API key: set the environment variable {bcfg.api_key_env!r} "
            "(or pass --api-key-env NAME)"
        )

    client = openai.OpenAI(api_key=key, base_url=bcfg.base_url, timeout=bcfg.timeout_s)
    log.info("backend client ready (model=%s, base_url=%s, timeout=%ss, max_concurrency=%s)",
             bcfg.model, bcfg.base_url, bcfg.timeout_s, bcfg.max_concurrency)
    return OpenAICompatibleClient(
        client, model=bcfg.model, default_temperature=bcfg.temperature,
        max_concurrency=bcfg.max_concurrency,
    )

class OpenAICompatibleExecutor:
    """Realizes operator nodes via real LLM calls, metering tokens and sequential calls.
    """

    def __init__(
        self,
        client: OpenAICompatibleClient,
        seed: int = 0,
        code_timeout_s: float = 8.0,
        profile: TaskProfile = None,
    ):
        self.client = client
        self.seed = seed
        self.code_timeout_s = code_timeout_s
        self.code_test_loops = 3
        # Each round is a full harness grade (Docker run), so keep it small
        try:
            self.swe_repair_loops = max(1, int(os.environ.get("SWE_REPAIR_LOOPS", "2")))
        except ValueError:
            self.swe_repair_loops = 2
        self.profile = profile or DEFAULT_PROFILE
        self._meta = ExecMeta()
        self._solution = ""
        self._problem = ""
        self._public_tester = None  # per-item (solution)->(passed, feedback); set by the Evaluator

    def reset(self) -> None:
        self._meta = ExecMeta()
        self._solution = ""
        self._problem = ""
        self._public_tester = None  # the Evaluator re-attaches the per-item tester after reset()

    def set_public_tester(self, tester) -> None:
        """Attach the current item's public-test oracle (solution -> (passed, feedback)).
        """
        self._public_tester = tester

    @property
    def meta(self) -> ExecMeta:
        return self._meta

    def solution(self) -> str:
        return self._solution

    # one metered chat round (counts as one sequential call)
    def _call(self, prompt: str, temp: float = None, seed_off: int = 0) -> str:
        text, tokens = self.client.complete_metered(
            prompt, temperature=temp, seed=self.seed + seed_off
        )
        self._meta = self._meta.add(ExecMeta(tokens=tokens, sequential_calls=1))
        return text

    def _prompt(self, body: str, *, want_answer: bool = True) -> str:
        """Assemble a prompt as profile.description + input + body (+ answer format).

        - body: the operator-specific instruction inserted after the input.
        - want_answer: when True, append profile.answer_format (for answer-producing steps).
        """
        parts = [self.profile.description, f"Input:\n{self._problem}", body]
        if want_answer:
            parts.append(self.profile.answer_format)
        return "\n\n".join(parts)

    def _vote_key(self, text: str) -> str:
        """Equivalence key grouping equal answers in self-consistency voting.

        Uses profile.answer_key if set, else the normalized last non-empty line.
        """
        if not text:
            return ""
        if self.profile.answer_key is not None:
            return self.profile.answer_key(text)
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        return lines[-1].lower() if lines else text.strip()[:64]

    def _vote(self, outputs: list[str]) -> str:
        """Majority vote over candidate outputs by answer-equivalence key.
        """
        outs = [o for o in outputs if o]
        if not outs:
            return self._solution
        keys = [self._vote_key(s) for s in outs]
        counts: dict[str, int] = {}
        for key in keys:  # insertion-ordered dict preserves first-seen order
            counts[key] = counts.get(key, 0) + 1
        winner = max(counts, key=lambda k: counts[k])  # ties -> first-inserted key
        for s, key in zip(outs, keys):
            if key == winner:
                return s
        return outs[0]

    async def run(self, operator: str, width: int, role: str, payload):
        # The first node sees the raw problem; subsequent nodes see the running
        # candidate but always have the original problem available as context.
        if not self._problem:
            self._problem = str(payload)
        candidate = self._solution or self._problem

        if operator in ("Generate", "Custom"):
            self._solution = self._op_generate(role)
        elif operator == "ReviewRevise":
            self._solution = self._op_review_revise(candidate)
        elif operator == "Ensemble":
            self._solution = self._op_ensemble(width)
        elif operator == "Localize":
            self._solution = self._op_localize()
        elif operator == "PatchRepair":
            self._solution = self._op_patch_repair()
        elif operator == "Programmer":
            self._solution = self._op_programmer()
        elif operator == "StepProgrammer":
            self._solution = self._op_step_programmer(candidate)
        elif operator == "TestCode":
            self._solution = self._op_test_code(candidate)
        elif operator == "Decompose":
            self._solution = self._op_decompose()
        elif operator == "EvidenceSelect":
            self._solution = self._op_evidence_select()
        elif operator == "GroundCheck":
            self._solution = self._op_ground_check(candidate)
        elif operator == "EliminateChoices":
            self._solution = self._op_eliminate_choices()
        elif operator == "Test":
            self._solution = self._op_test(candidate)
        else:
            # unknown operator: treat as a generic generation step
            self._solution = self._op_generate(role)
        return self._solution

    async def run_unit(self, operator: str, role: str, payload, *, meter: bool = True):
        """Run one width-1 invocation for the control-flow expander.

        - meter: when False, suppress only the sequential-call (latency) increment so
          the expander can later charge ONE latency round per branch via account()
          (parallel semantics). Real tokens stay measured either way.
        """
        before_calls = self._meta.sequential_calls
        out = await self.run(operator=operator, width=1, role=role, payload=payload)
        if not meter:
            # roll back the latency increment(s) this unit added; the expander's
            # account() supplies the control node's correct one-round/L-round calls.
            self._meta = ExecMeta(
                tokens=self._meta.tokens, sequential_calls=before_calls
            )
        return out

    async def run_branch(self, operator: str, role: str, payload, width: int):
        k = max(1, int(width))
        base_seed = self.seed
        entry_solution = self._solution  # each attempt restarts from the same candidate
        outputs: list[str] = []
        try:
            for i in range(k):
                self._solution = entry_solution
                self.seed = base_seed + 1_000_003 * (i + 1)  # distinct sample per attempt
                before_calls = self._meta.sequential_calls
                out = await self.run(operator=operator, width=1, role=role, payload=payload)
                # roll back the per-attempt latency; account() charges the node's rounds
                self._meta = ExecMeta(tokens=self._meta.tokens, sequential_calls=before_calls)
                outputs.append(out)
        finally:
            self.seed = base_seed
        winner = self._vote(outputs)
        self._solution = winner
        return winner

    def account(self, tokens: int, sequential_calls: int,
                acc_logit: float = 0.0, rob: float = 0.0, cons: float = 0.0) -> None:
        """Charge explicit latency (sequential calls) for a control node.

        - sequential_calls: latency rounds to add for this control node.
        - tokens: ignored; real tokens were already measured by unmetered run_unit calls.
        - acc_logit, rob, cons: ignored (the real backend grades via the checker).
        """
        self._meta = self._meta.add(ExecMeta(tokens=0, sequential_calls=int(sequential_calls)))

    def _op_generate(self, role: str) -> str:
        return self._call(
            self._prompt(f"Role: {role}. Reason step by step, then give your best answer.")
        )

    def _op_review_revise(self, candidate: str) -> str:
        critique = self._call(
            self._prompt(
                f"A candidate answer:\n{candidate}\n\n"
                "Critique it: point out any errors, gaps, or weaknesses. Be specific.",
                want_answer=False,
            ),
            seed_off=1,
        )
        revised = self._call(
            self._prompt(
                f"Previous attempt:\n{candidate}\n\nCritique:\n{critique}\n\n"
                "Produce a corrected, improved answer."
            ),
            seed_off=2,
        )
        return revised

    def _op_ensemble(self, width: int) -> str:
        # `width` parallel samples (tokens grow, sequential latency does not: the
        # OperatorSpec marks Ensemble parallel, so count it as ONE sequential round).
        k = max(1, int(width))
        prompt = self._prompt("Solve the task independently.")
        samples: list[str] = []
        total_tokens = 0
        for i in range(k):
            text, tokens = self.client.complete_metered(
                prompt, temperature=0.9, seed=self.seed + 100 + i
            )
            samples.append(text)
            total_tokens += tokens
        # parallel: add all tokens but only ONE sequential call (matches latency model)
        self._meta = self._meta.add(ExecMeta(tokens=total_tokens, sequential_calls=1))
        return self._vote(samples)

    def _op_programmer(self) -> str:
        code_resp = self._call(
            "Write a short, self-contained Python program that solves the task "
            "below and prints ONLY the final answer.\n\nInput:\n"
            f"{self._problem}\n\nReturn only a ```python code block.",
            seed_off=3,
        )
        source = _extract_code_block(code_resp)
        if source:
            res = run_code(source, timeout_s=self.code_timeout_s)
            if res.ok and res.stdout.strip():
                return res.stdout.strip()
        # fall back to the model's own reasoning if the program failed
        return self._op_generate("programmer-fallback")

    def _op_step_programmer(self, candidate: str = "") -> str:
        instruction = (
            "The workflow so far has produced the intermediate result shown below. "
            "Write a short, self-contained Python program that builds on it to "
            "compute and print ONLY the final answer to the task. The earlier step "
            "may have outlined an approach, decomposed the task into subproblems, "
            "produced partial results, or proposed an answer — use whatever is "
            "helpful and carry the work forward in code. Compute the answer "
            "programmatically rather than merely reprinting a value stated in the "
            "text.\n\nInput:\n"
            f"{self._problem}\n\nWork so far:\n{candidate}\n\n"
            "Return only a ```python code block."
        )
        code_resp = self._call(instruction, seed_off=6)
        source = _extract_code_block(code_resp)
        if source:
            res = run_code(source, timeout_s=self.code_timeout_s)
            if res.ok and res.stdout.strip():
                return res.stdout.strip()
        return self._op_test(candidate)

    def _op_test(self, candidate: str) -> str:
        return self._call(
            self._prompt(
                f"Proposed answer:\n{candidate}\n\n"
                "Verify it by re-deriving or sanity-checking. If it is wrong, give "
                "the corrected answer."
            ),
            seed_off=4,
        )

    def _op_test_code(self, candidate: str) -> str:
        """Execute the item's public tests and repair on failure.
        """
        tester = self._public_tester
        if tester is None:
            return candidate
        sol = candidate
        for _ in range(max(1, int(self.code_test_loops))):
            passed, feedback = tester(sol)
            if passed:
                return sol
            resp = self._call(
                self._prompt(
                    f"A Python solution that failed the public tests:\n{sol}\n\n"
                    f"Execution result / failed test case:\n{feedback}\n\n"
                    "Reflect on why it failed, then provide a better, corrected solution. "
                    "Return only the function definition."
                ),
                seed_off=5,
            )
            sol = _extract_code_block(resp)
        return sol

    def _op_decompose(self) -> str:
        chain = self._call(
            self._prompt(
                "Break this question into the minimal ordered sequence of single-hop "
                "sub-questions needed to answer it. Answer each sub-question using only "
                "the context, quoting the supporting passage. Later sub-questions may "
                "depend on earlier answers. Output the numbered sub-question/answer chain "
                "-- do not give the final answer yet.",
                want_answer=False,
            ),
            seed_off=10,
        )
        return self._call(
            self._prompt(
                f"Sub-question reasoning chain:\n{chain}\n\n"
                "Using this chain, state the final answer to the original question."
            ),
            seed_off=11,
        )

    def _op_evidence_select(self) -> str:
        evidence = self._call(
            self._prompt(
                "The context contains relevant passages mixed with distractors. Identify "
                "the passages actually relevant to the question: list their titles and the "
                "specific supporting sentences, and ignore everything else. Output only this "
                "selected evidence -- do not answer the question yet.",
                want_answer=False,
            ),
            seed_off=12,
        )
        return self._call(
            self._prompt(
                f"Selected supporting evidence:\n{evidence}\n\n"
                "Answer the question using only this selected evidence."
            ),
            seed_off=13,
        )

    def _op_ground_check(self, candidate: str) -> str:
        check = self._call(
            self._prompt(
                f"Proposed answer:\n{candidate}\n\n"
                "Check whether the context actually supports this answer: quote the exact "
                "passage that entails it. If no passage supports it, or it relies on a "
                "distractor, say so and explain what the context does support. Do not give "
                "a final answer yet.",
                want_answer=False,
            ),
            seed_off=14,
        )
        return self._call(
            self._prompt(
                f"Previous answer:\n{candidate}\n\nGrounding check:\n{check}\n\n"
                "Produce the final, context-grounded answer (correct it if the check found "
                "it unsupported)."
            ),
            seed_off=15,
        )

    def _op_eliminate_choices(self) -> str:
        analysis = self._call(
            self._prompt(
                "Go through the answer options one at a time. For each option, give the "
                "specific reason it is correct or incorrect, using domain reasoning. Rule "
                "out every distractor you can. Output this per-option analysis -- do not "
                "give the final answer yet.",
                want_answer=False,
            ),
            seed_off=16,
        )
        return self._call(
            self._prompt(
                f"Per-option elimination analysis:\n{analysis}\n\n"
                "Based on this elimination, select the single best surviving option."
            ),
            seed_off=17,
        )

    def _op_localize(self) -> str:
        """SWE fault localization: name the responsible file(s)/function(s) before patching.
        """
        return self._call(
            self._prompt(
                "Before writing any code, localize the bug: identify the source file path(s), "
                "the function or class, and the specific lines or logic most likely responsible "
                "for the issue, with a brief justification. Do NOT write a patch yet -- output "
                "only this localization analysis.",
                want_answer=False,
            ),
            seed_off=9,
        )

    def _op_patch_repair(self) -> str:
        """SWE apply/test loop: draft a unified-diff patch, grade it against the repo's tests,
        and repair on failure using the grader's feedback.
        """
        prior = self._solution  # localization notes / a prior patch from an earlier node, if any
        context = f"\n\nWork so far (use if helpful):\n{prior}" if prior else ""
        sol = self._call(
            self._prompt(
                "Write the fix for the issue above as a single unified git diff." + context
            ),
            seed_off=7,
        )
        tester = self._public_tester
        if tester is None:
            return sol
        for _ in range(max(1, int(self.swe_repair_loops))):
            passed, feedback = tester(sol)
            if passed:
                return sol
            sol = self._call(
                self._prompt(
                    f"Your previous patch did not resolve the issue.\n\n"
                    f"Previous patch:\n{sol}\n\n"
                    f"Grader feedback:\n{feedback}\n\n"
                    "Diagnose the failure and produce a corrected unified git diff."
                ),
                seed_off=8,
            )
        return sol


def _extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

_PROPOSE_PROMPT = (
    "You are designing an LLM workflow (a DAG of operators) to solve the following task:\n"
    "{task}\n\n"
    "The current partial workflow has these operators in order: {ops}.\n\n"
    "Available operators (choose only from these, by exact name):\n{catalogue}\n\n"
    "Propose up to {n} good NEXT atomic edits to extend or finish the workflow. "
    "An edit is one of: a base operator; \"loop\" (wrap an operator to refine it "
    "sequentially for `width` passes); \"branch\" (run `width` parallel attempts of "
    "an operator and keep the best); or \"Terminate\".\n"
    "Pick operators and roles that fit THIS task; the last operator before Terminate must "
    "produce the task's required answer/output (do not finish on an analysis-only step).\n"
    'Reply with ONLY a JSON list, each item {{"op": <operator|"loop"|"branch"|"Terminate">, '
    '"operator": <inner operator for loop/branch, else omit>, "width": <int>=1>, '
    '"role": <short label>, "why": <one-line rationale>}}.'
)

_REALIZE_PROMPT = (
    "You are instantiating one step of an LLM workflow for the following task:\n"
    "{task}\n\n"
    "The workflow so far has these operators in order: {ops}.\n\n"
    "The next step adds a `{operator}` operator (width {width}). That operator's job: "
    "{operator_desc}\n"
    "Propose {n} DISTINCT concrete ways to play that step -- each a different role/approach "
    "the operator could take FOR THIS TASK (a distinct prompt archetype, reasoning strategy, "
    "or specialization), so that executing them could plausibly lead to different outcomes. "
    "Each role must fit the operator's job and this task.\n"
    'Reply with ONLY a JSON list of {n} items, each {{"role": <short distinct '
    'approach label, <=40 chars>, "why": <one-line rationale>}}.'
)


class OpenAICompatibleProposer:
    """LLM-as-proposer of atomic edits, and (optionally) of their concrete realizations.
    """

    def __init__(
        self,
        client: OpenAICompatibleClient,
        spec: ObjectiveSpec,
        n_realizations: int = 3,
        operators: tuple[str, ...] = None,
        task_profile: TaskProfile = None,
    ):
        self.client = client
        self.spec = spec
        self.n_realizations = int(n_realizations)
        # task context grounds the proposer so realized roles fit the task.
        profile = task_profile or DEFAULT_PROFILE
        self.task_desc = (profile.description or "").strip() or "(unspecified task)"

        if operators is None:
            self.operators = tuple(ops.names())
        else:
            allowed = [o for o in operators if o in ops.names()]
            unknown = [o for o in operators if o not in ops.names()]
            if unknown:
                log.warning("ignoring unknown operators in pool: %s", unknown)
            if not allowed:
                raise ValueError(
                    f"operator pool {operators!r} has no registered operators; "
                    f"known: {sorted(ops.names())}"
                )
            self.operators = tuple(allowed)

        # cache of concrete realizations per (state signature, nominal action key),
        self._realizations: dict[tuple, list] = {}

    def _catalogue(self) -> str:
        """Operator pool rendered as `- name: description` so the proposer knows what each operator actually does."""
        lines = []
        for name in self.operators:
            try:
                desc = ops.get(name).description
            except KeyError:
                desc = ""
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        return "\n".join(lines)

    def _complete_json_list(self, prompt: str, *, tries: int = 3) -> list[dict]:
        items: list[dict] = []
        for attempt in range(1, tries + 1):
            items = _parse_json_list(self.client.complete(prompt, temperature=0.7))
            if items:
                return items
            log.warning("proposer: empty/garbled JSON reply (attempt %d/%d); retrying",
                        attempt, tries)
        return items

    def realize(self, graph, edit, seed: int):
        """Return one concrete realization of `edit` for this sample `seed`.

        - graph: the current workflow the edit applies to (its signature keys the cache).
        - edit: the nominal edit chosen by CZT at the decision node.
        - seed: per-sample seed selecting which cached realization to draw.
        """
        if self.n_realizations <= 1 or not isinstance(edit, (AddOperator, AddControl)):
            return edit
        cache_key = (graph.signature(), edit.key())
        variants = self._realizations.get(cache_key)
        if variants is None:
            variants = self._make_variants(graph, edit)  # one LLM call; may raise
            self._realizations[cache_key] = variants
        return variants[seed % len(variants)]

    def _make_variants(self, graph, edit) -> list:
        """Ask the LLM for distinct role specializations of `edit`; return concrete edits.
        """
        cur = ", ".join(f"{x.operator}(k={x.width})" for x in graph.nodes) or "(empty)"
        try:
            operator_desc = ops.get(edit.operator).description or "(no description)"
        except KeyError:
            operator_desc = "(no description)"
        prompt = _REALIZE_PROMPT.format(
            task=self.task_desc, ops=cur, operator=edit.operator,
            operator_desc=operator_desc, width=edit.width, n=self.n_realizations,
        )
        items = self._complete_json_list(prompt)
        seen: set[str] = set()
        variants: list = []
        for it in items:
            role = str(it.get("role", "")).strip()
            if not role or role in seen:
                continue
            seen.add(role)
            why = str(it.get("why", "")) or edit.rationale
            variants.append(replace(edit, role=role, rationale=why))
        if not variants:
            # LLM gave nothing usable even after retries: fall back to a single
            # unspecialized variant so realization proceeds instead of crashing the run.
            log.warning("no usable realization roles for %r; using a default variant",
                        edit.operator)
            fallback_role = getattr(edit, "role", None) or "step"
            variants.append(replace(edit, role=fallback_role, rationale=edit.rationale))
        return variants

    def propose(self, graph, n: int, max_depth: int) -> list[EditProposal]:
        cur = ", ".join(f"{x.operator}(k={x.width})" for x in graph.nodes) or "(empty)"
        prompt = _PROPOSE_PROMPT.format(
            task=self.task_desc, ops=cur, catalogue=self._catalogue(), n=n
        )
        items = self._complete_json_list(prompt)
        out: list[EditProposal] = []
        for it in items:
            edit = _edit_from_json(it, self.operators)
            # drop edits whose operator falls outside the task pool
            if edit is not None and getattr(edit, "operator", None) is not None \
                    and edit.operator not in self.operators:
                continue
            if edit is not None and is_legal(edit, graph, max_depth):
                why = str(it.get("why", ""))
                out.append(EditProposal(edit=edit, rationale=why, prior=0.5))

        if n:
            # reserve a slot for Terminate
            out = out[: max(1, n - 1)] if len(out) >= n else out[:n]
        term = Terminate(rationale="LLM proposer: stop here")
        if is_legal(term, graph, max_depth) and term.key() not in {
            p.edit.key() for p in out
        }:
            out.append(EditProposal(edit=term, rationale=term.rationale, prior=0.3))
        return out


def _parse_json_list(text: str) -> list[dict]:
    """Parse a JSON list of dicts from an LLM reply, tolerating code fences / prose and
    common malformations."""
    if not text:
        return []
    # tolerate code fences / prose around the JSON
    m = re.search(r"\[.*\]", text, re.DOTALL)
    blob = m.group(0) if m else text
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return _salvage_json_objects(blob)
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


def _salvage_json_objects(blob: str) -> list[dict]:
    """Best-effort recovery from a malformed JSON array."""
    out: list[dict] = []
    depth = 0
    start = None
    for i, ch in enumerate(blob):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                chunk = blob[start:i + 1]
                obj = None
                for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                    try:
                        obj = json.loads(candidate)
                        break
                    except json.JSONDecodeError:
                        continue
                if isinstance(obj, dict):
                    out.append(obj)
                start = None
    return out


def _edit_from_json(it: dict, allowed: tuple[str, ...] = None):

    pool = tuple(allowed) if allowed else tuple(ops.names())
    default_inner = "Custom" if "Custom" in pool else pool[0]
    op = str(it.get("op", "")).strip()
    if op.lower() == "terminate":
        return Terminate(rationale=str(it.get("why", "stop")))
    width = it.get("width", 1)
    try:
        width = max(1, int(width))
    except (TypeError, ValueError):
        width = 1
    role = str(it.get("role", "step"))[:40] or "step"
    why = str(it.get("why", ""))

    if op.lower() in ("loop", "branch"):
        inner = str(it.get("operator", default_inner)) or default_inner
        if inner not in pool:
            inner = default_inner
        return AddControl(control_kind=op.lower(), width=width, operator=inner,
                          role=role, rationale=why)
    if op not in pool:
        return None
    return AddOperator(operator=op, width=width, role=role, rationale=why)