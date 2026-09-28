"""The CHMCTS / THTS search loop.

Per trial:
  1. SAMPLE a preference w ~ W_D (the CZT context).
  2. SELECT: descend; CZT picks an atomic decision for w at each decision node.
  3. EXPAND: at an unexpanded action, the proposer mints candidates and a chance
     node is created.
  4. INSTANTIATE + EVALUATE: realize the decision; terminal -> execute for reward
     R, else value the interior node via the (uncertainty-gated) predictor /
     completion beam.
  5. BACKUP: CHVI chance (Q-hat) then decision (V-hat) backups up the path.
  6. ARCHIVE: terminal workflows + traces enter the global CCS archive.
"""

import csv
import os

import numpy as np
from tqdm.auto import tqdm

from ..ccs.ccs import CCS
from ..config import SearchConfig
from ..logging_util import get_logger
from ..mdp import actions as mdp_actions
from ..mdp.state import State
from ..mdp.transition import instantiate
from ..objectives import ObjectiveSpec
from ..serve.save_policy import save_bundle
from ..valuation.completion import completion_beam
from .archive import Archive, ArchiveEntry
from .backups import backup_chance, backup_decision
from .checkpoint import Checkpointer
from .nodes import ChanceNode, DecisionNode
from .selection import make_selector
from .trace import ConstructionTrace

log = get_logger("engine")


class Engine:
    def __init__(
        self,
        spec: ObjectiveSpec,
        config: SearchConfig,
        proposer,
        evaluator,
        predictor,
        evalset,
        checkpointer: Checkpointer = None,
        resume: bool = True,
        metrics_path: str = None,
        weights_path: str = None,
        bundle_path: str = None,
        offline: bool = False,
    ):
        self.spec = spec
        self.cfg = config
        self.proposer = proposer
        self.evaluator = evaluator
        self.predictor = predictor
        # offline: value every node from the predictor (a pre-trained GNN); never execute workflows
        self.offline = bool(offline)

        self._metrics_path = metrics_path
        self._weights_path = weights_path
        self._bundle_path = bundle_path

        self._realizer = getattr(proposer, "realize", None)

        self._evalset = evalset

        self.rng = np.random.default_rng(config.seed)
        self._acc_idx = spec.names.index("accuracy") if spec.has("accuracy") else None
        self._cons_idx = spec.names.index("consistency") if spec.has("consistency") else None
        self._rob_idx = spec.names.index("robustness") if spec.has("robustness") else None

        self.compute_consistency = self._acc_idx is not None and self._cons_idx is not None
        self.compute_robustness = self._rob_idx is not None

        self._total_trials: int = 0
        self.hv_log: list[dict] = []
        self.w_log: list[np.ndarray] = []  # preference sampled per completed trial

        self._make_selector = lambda: make_selector(config.selector, self.spec, config)

        self.archive = Archive(spec.D, spec=spec)
        self.root = DecisionNode(
            State.root(config.max_depth),
            spec.D,
            self._make_selector(),
            trace=ConstructionTrace(),
        )
        self._nodes: dict[str, DecisionNode] = {self.root.state.key(): self.root}

        self._ckpt = checkpointer
        if self._ckpt is not None and resume and self._ckpt.exists():
            self.restore(self._ckpt.load())
            log.info(
                "resumed from checkpoint: %d trials, archive=%d, nodes=%d",
                self._total_trials, len(self.archive), len(self._nodes),
            )

    @property
    def total_trials(self) -> int:
        """Number of completed trials (bundle/checkpoint metadata)."""
        return self._total_trials

    def _sample_preference(self) -> np.ndarray:
        """Sample a preference w ~ uniform Dirichlet"""
        w = self.rng.dirichlet(np.ones(self.spec.D))
        return w

    def _realize(self, graph, edit, seed: int):
        """Realize the atomic edit into a concrete edit for this sample seed.
        """
        if self._realizer is None:
            return edit
        return self._realizer(graph, edit, seed)

    def _decision_node(self, state: State, trace: ConstructionTrace) -> DecisionNode:
        key = state.key()
        node = self._nodes.get(key)
        if node is None:
            node = DecisionNode(
                state, self.spec.D, self._make_selector(), trace=trace
            )
            self._nodes[key] = node
        return node

    def run(self, n_trials: int = None, progress: bool = True, desc: str = "search") -> Archive:

        n = n_trials if n_trials is not None else self.cfg.n_trials
        if self._total_trials >= n:
            # a fully-completed checkpoint was restored; nothing left to do.
            log.info("resume: already at %d/%d trials, nothing to run", self._total_trials, n)
            return self.archive
        start = self._total_trials
        log.info("search start: %d trials (from %d, max_depth=%d, branching=%d, R=%d)",
                 n, start, self.cfg.max_depth, self.cfg.branching, self.cfg.chance_samples)
        bar = tqdm(total=n, initial=start, desc=desc, unit="trial", leave=False) if progress else None
        # loop on the running total (not a range) so a resumed run continues toward n.
        while self._total_trials < n:
            w = self._sample_preference()
            log.debug("trial %d/%d start (w=%s)", self._total_trials + 1, n, np.array2string(w, precision=3))
            self._trial(w)
            if not self.offline:
                self.predictor.maybe_refit()
            self._total_trials += 1
            self.w_log.append(np.asarray(w, dtype=float))
            log.debug("trial %d/%d done | archive=%d nodes=%d", self._total_trials, n, len(self.archive), len(self._nodes))

            stop = False
            if self._total_trials % self.cfg.checkpoint_every == 0 or self._total_trials == n:
                self._record_checkpoint()
                stop = self._should_early_stop()

            if self._ckpt is not None and (
                stop or self._total_trials == n or self._total_trials % self._ckpt.every == 0
            ):
                self._ckpt.save(self.snapshot())
                self._persist_outputs()

            if self._bundle_path is not None and not self.offline and (
                stop or self._total_trials == n or self._total_trials % self.cfg.checkpoint_every == 0
            ):
                # offline: the bundle is read-only (the trained GNN we loaded); never overwrite it.
                save_bundle(self._bundle_path, self.predictor, self.spec, trials=self._total_trials)

            if bar is not None:
                bar.update(1)
                post = {"archive": len(self.archive), "nodes": len(self._nodes)}
                if stop:
                    post["stop"] = "early"
                bar.set_postfix(**post)

            prog_every = max(1, (n - start) // 20)
            if self._total_trials % prog_every == 0 or self._total_trials == n or stop:
                hv = self.hv_log[-1]["hv"] if self.hv_log else 0.0
                log.info("%s: trial %d/%d | HV=%.4g | archive=%d", desc,
                         self._total_trials, n, hv, len(self.archive))
            if stop:
                break
        if bar is not None:
            bar.close()
        log.info("search done: %d/%d trials | archive=%d", self._total_trials, n, len(self.archive))
        return self.archive

    def _should_early_stop(self) -> bool:
        # Early stop if the relative hypervolume gain over the trailing window is below the tolerance.
        cfg = self.cfg
        if not cfg.early_stop:
            return False
        if self._total_trials < cfg.es_min_trials or len(self.hv_log) < 2:
            return False
        cur_hv = self.hv_log[-1]["hv"]
        if cur_hv <= 0.0:
            return False  # nothing above the reference point yet; keep searching
        cutoff = self._total_trials - cfg.es_patience
        past = next((r for r in reversed(self.hv_log[:-1]) if r["trial"] <= cutoff), None)
        if past is None:
            return False  # history doesn't yet span a full patience window
        prev_hv = past["hv"]
        denom = prev_hv if prev_hv > 0.0 else 1.0
        rel_gain = (cur_hv - prev_hv) / denom
        if rel_gain < cfg.es_tol:
            log.info(
                "early stop at trial %d: relative HV gain %.4g over last %d trials < tol %.4g",
                self._total_trials, rel_gain, self._total_trials - past["trial"], cfg.es_tol,
            )
            return True
        return False

    def _client_total_tokens(self) -> int:
        """Cumulative tokens through the shared LLM client (proposer + executors), or 0."""
        client = getattr(getattr(self, "proposer", None), "client", None)
        return int(getattr(client, "total_tokens", 0)) if client is not None else 0

    def _record_checkpoint(self) -> None:

        arc_ccs = self.archive.ccs()
        norm_ccs = CCS(points=self.spec.normalize(arc_ccs.points))
        ev = self.evaluator
        self.hv_log.append({
            "trial": self._total_trials,
            "n_evaluations": getattr(ev, "n_evaluations", 0),# full eval-set passes (R chance-samples per terminal workflow each count)
            "n_executions": getattr(ev, "n_executions", 0),  # item-level runs (n_evaluations * n_samples_per_eval)
            "total_tokens": getattr(ev, "total_tokens", 0),  # execution tokens consumed by the evaluator (0 in offline mode)
            "total_llm_tokens": self._client_total_tokens(), # ALL tokens through the shared client (generation + execution); the only nonzero LLM cost in offline mode
            "total_calls": getattr(ev, "total_calls", 0),    # total sequential LLM calls made by the evaluator
            "hv": norm_ccs.hypervolume(np.zeros(self.spec.D)), # normalized hypervolume of the archive front (per-axis (v-ref)/span
            "archive_size": len(self.archive),               # number of non-dominated workflows in the archive
            "n_nondominated": len(arc_ccs),                  # size of the Pareto front after pruning the archive
            "sparsity": norm_ccs.sparsity(np.zeros(self.spec.D)), # PD-MORL sparsity: per-axis mean squared gap between adjacent sorted coords, averaged over axes (lower = denser)
        })

    def _persist_outputs(self) -> None:
        if self._metrics_path:
            self.save_metrics(self._metrics_path)
        if self._weights_path:
            self.save_weights(self._weights_path)

    def save_metrics(self, path: str) -> None:
        """Write the HV-over-budget trajectory to a CSV file at `path`.
        """
        if not self.hv_log:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fieldnames = list(dict.fromkeys(k for row in self.hv_log for k in row))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(self.hv_log)

    def save_weights(self, path: str) -> None:
        """Write the per-trial sampled preference weights to a CSV at `path`.
        """
        if not self.w_log:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["trial"] + [f"w{i}" for i in range(self.spec.D)])
            for t, w in enumerate(self.w_log, 1):
                writer.writerow([t] + [f"{float(x):.6g}" for x in np.asarray(w).ravel()])

    def snapshot(self) -> dict:
        """Return the pure, picklable search state needed to resume run() faithfully.
        """
        pred_state = None
        if hasattr(self.predictor, "state_dict"):
            pred_state = self.predictor.state_dict()
        ev = self.evaluator
        return {
            "version": 3,
            "total_trials": self._total_trials,
            "hv_log": self.hv_log,
            "w_log": self.w_log,
            "rng": self.rng,
            "nodes": self._nodes,
            "root_key": self.root.state.key(),
            "archive_entries": self.archive.entries,
            "predictor_state": pred_state,
            "evaluator_meter": {
                "n_evaluations": int(getattr(ev, "n_evaluations", 0)),
                "n_executions": int(getattr(ev, "n_executions", 0)),
                "total_tokens": int(getattr(ev, "total_tokens", 0)),
                "total_calls": int(getattr(ev, "total_calls", 0)),
            },
            "client_meter": {"total_tokens": self._client_total_tokens()},
        }

    def restore(self, snap: dict) -> None:
        """Reload a `snapshot()` dict onto this engine, in place.
        """
        self._total_trials = int(snap["total_trials"])
        self.hv_log = list(snap.get("hv_log", []))
        self.w_log = list(snap.get("w_log", []))
        self.rng = snap["rng"]
        self._nodes = snap["nodes"]
        self.root = self._nodes[snap["root_key"]]
        self.archive = Archive(self.spec.D, spec=self.spec)
        for entry in snap.get("archive_entries", []):
            self.archive.add(entry)
        pred_state = snap.get("predictor_state")
        if pred_state is not None and hasattr(self.predictor, "load_state_dict"):
            self.predictor.load_state_dict(pred_state)
        # restore the evaluator's cumulative meter
        meter = snap.get("evaluator_meter")
        if meter and self.evaluator is not None:
            self.evaluator.n_evaluations = int(meter.get("n_evaluations", 0))
            self.evaluator.n_executions = int(meter.get("n_executions", 0))
            self.evaluator.total_tokens = int(meter.get("total_tokens", 0))
            self.evaluator.total_calls = int(meter.get("total_calls", 0))
        # restore the shared client's cumulative token meter
        cmeter = snap.get("client_meter")
        client = getattr(getattr(self, "proposer", None), "client", None)
        if client is not None:
            if cmeter:
                client.total_tokens = int(cmeter.get("total_tokens", 0))
            elif meter:
                client.total_tokens = int(meter.get("total_tokens", 0))

    def _trial(self, w: np.ndarray) -> None:
        path: list[tuple] = []  # (DecisionNode, ChanceNode, selection, succ_key)
        node = self.root

        while True:
            # terminal state: nothing to expand
            if node.state.is_terminal():
                break

            # propose candidate actions and register them with CZT
            log.debug("  depth=%d propose (n=%d) on %s", node.state.depth, self.cfg.branching, node.state.key()[:12])
            props = self.proposer.propose(
                node.state.graph, n=self.cfg.branching, max_depth=self.cfg.max_depth
            )
            if not props:
                log.debug("  depth=%d no proposals -> stop descent", node.state.depth)
                break
            log.debug("  depth=%d proposed: %s", node.state.depth, [p.edit.label() for p in props])
            node.selector.ensure_actions(props)
            action_keys = {p.edit.key() for p in props}

            # select an action for this preference via the node's action selection policy
            sel = node.selector.select(w, action_keys, node)
            if sel is None:
                break
            edit = sel.edit

            # get or create the chance node for this action
            ck = edit.key()
            chance = node.children.get(ck)
            if chance is None:
                chance = ChanceNode(ck, edit, sel.rationale, self.spec.D)
                node.children[ck] = chance

            log.debug("  depth=%d selected action '%s' -> realize", node.state.depth, edit.label())
            seed = int(self.rng.integers(1 << 30))
            realized = self._realize(node.state.graph, edit, seed)
            succ_state = instantiate(node.state, realized, sample_seed=seed)
            delta_trace = node.trace.extended(realized.label(), sel.rationale)
            succ = self._decision_node(succ_state, delta_trace)
            succ_key = succ_state.key()
            chance.observe(succ_key, succ)

            path.append((node, chance, sel, succ_key))

            if succ_state.is_terminal():
                node = succ
                break
            node = succ
            if node.state.at_horizon():
                # Horizon: terminate is the only legal move. Add it as a real on-path
                # transition (chance node + terminal successor) so the terminal value backs
                # up into the tree.
                term_edit = _terminate()
                tck = term_edit.key()
                tchance = node.children.get(tck)
                if tchance is None:
                    tchance = ChanceNode(tck, term_edit, "horizon reached", self.spec.D)
                    node.children[tck] = tchance
                term_state = mdp_actions.apply(node.state, term_edit)
                term_trace = node.trace.extended("terminate (horizon)", "horizon reached")
                term_node = self._decision_node(term_state, term_trace)
                tkey = term_state.key()
                tchance.observe(tkey, term_node)
                path.append((node, tchance, None, tkey))
                node = term_node
                break

        # ---- evaluate the reached node
        leaf_value = self._value_node(node)

        # ---- archive terminal workflows
        if node.state.is_terminal() and not leaf_value.is_empty():
            for i in range(len(leaf_value)):
                self.archive.add(
                    ArchiveEntry(
                        vector=leaf_value.points[i].copy(),
                        graph=node.state.graph,
                        trace=node.trace,
                    )
                )

        # raw leaf value vectors handed to each selector (it normalizes/scalarizes itself).
        value_vectors = [leaf_value.points[i].copy() for i in range(len(leaf_value))]
        leaf_samples = getattr(node, "_leaf_samples", None)
        if node.state.is_terminal() and leaf_samples and path:
            _, last_chance, _, last_key = path[-1]
            succ = last_chance.successors.get(last_key)
            if succ is not None and succ.child is node:
                succ.add_samples(leaf_samples)

        # ---- backup along the path (chance then decision), update the selector
        for dnode, chance, sel, _succ_key in reversed(path):
            backup_chance(chance)
            backup_decision(dnode, half_life=self.cfg.anneal_half_life)
            if sel is not None:  # forced horizon-terminate carries no selection to credit
                dnode.selector.record(sel, w, value_vectors)

    def _value_node(self, node: DecisionNode) -> CCS:
        """Value a node: CCS of executed outcomes if terminal, else predictor estimate (or completion-beam front).
        """
        if self.offline:
            return self._value_node_offline(node)
        if node.state.is_terminal():
            R = max(1, self.cfg.chance_samples)
            log.info("  evaluate TERMINAL %s (R=%d chance-samples)", node.state.key()[:12], R)

            # single deduplicated pass: the R accuracy/consistency draws and the
            # robustness pass share one thread pool.
            samples, cons, rob = self.evaluator.evaluate_terminal(
                node.state.graph, self._evalset, R,
                want_consistency=self.compute_consistency,
                want_robustness=self.compute_robustness,
            )
            if cons is not None:
                for v in samples:
                    v[self._cons_idx] = cons
            if rob is not None:
                for v in samples:
                    v[self._rob_idx] = rob

            for v in samples:
                self.predictor.observe(node.state.graph, v)
            node._leaf_samples = samples
            node.V = CCS.of(samples)
            return node.V

        # intermediate node: prior estimate (analytic axes exact, acc/rob predicted)
        est, unc = self.predictor.predict(node.state.graph)
        node.prior = (est, unc)
        log.debug("  value INTERIOR %s: predictor unc=%.3g (gate=%.3g)", node.state.key()[:12], unc, self.cfg.uncertainty_gate)

        # uncertainty gate: only run a real completion beam if unsure
        if unc >= self.cfg.uncertainty_gate:
            log.info("  uncertainty %.3g >= gate -> completion beam (k=%d) on %s", unc, self.cfg.beam_k, node.state.key()[:12])
            front, _, completions = completion_beam(
                node.state,
                proposer=self.proposer,
                evaluator=self.evaluator,
                evalset=self._evalset,
                k=self.cfg.beam_k,
                rng=self.rng,
                base_trace=node.trace,
            )
            if not front.is_empty():
                node.V = front
                # each beam point is a REAL executed terminal workflow; archive it with its own terminal graph + construction trace.
                for vec, term_state, term_trace in completions:
                    self.predictor.observe(term_state.graph, vec)
                    self.archive.add(
                        ArchiveEntry(
                            vector=np.asarray(vec, dtype=float).copy(),
                            graph=term_state.graph,
                            trace=term_trace,
                        )
                    )
                return front
        node.V = CCS.of([est])
        return node.V

    def _value_node_offline(self, node: DecisionNode) -> CCS:
        """Value a node purely from the offline (pre-trained) GNN -- no workflow execution.
        """
        # deterministic valuation: the offline GNN is fixed, so a graph gets one stable value.
        est, unc = self.predictor.predict(node.state.graph, stochastic=False)
        node.prior = (est, unc)
        if node.state.is_terminal():
            node._leaf_samples = [np.asarray(est, dtype=float)]
            log.info("  evaluate TERMINAL %s via offline GNN (no execution)", node.state.key()[:12])
        node.V = CCS.of([est])
        return node.V

def _terminate():
    from ..workflow.edits import Terminate

    return Terminate(rationale="horizon reached")
