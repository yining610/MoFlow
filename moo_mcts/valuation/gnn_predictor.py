import numpy as np
import torch
import wandb

from ..logging_util import get_logger
from ..objectives import ObjectiveSpec
from ..valuation.analytic import analytic_vector
from ..workflow.graph import WorkflowGraph
from . import graph_encoder as ge
from .distill import WARMUP_REFIT, ReplayBuffer
from .role_encoder import RoleEncoder


log = get_logger("gnn")


def _find_device(device: str | None) -> torch.device:

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _migrate_legacy_mp_keys(model_sd: dict) -> dict:
    """Map pre-ModuleList checkpoints onto the current parameter names.
    """
    if not any(k.startswith(("mp1.", "mp2.")) for k in model_sd):
        return model_sd
    remap = (("mp1.", "mp_layers.0."), ("mp2.", "mp_layers.1."))
    out = {}
    for k, v in model_sd.items():
        for old, new in remap:
            if k.startswith(old):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


class GNNValuePredictor:
    def __init__(
        self,
        spec: ObjectiveSpec,
        critic,
        max_depth: int,
        hidden: int = 64,
        role_dim: int = 8,
        num_layers: int = 2,
        refit_every: int = 32,
        epochs: int = 60,
        seed: int = 0,
        buffer: ReplayBuffer = None,
        device: str = None,
        evidence_scale: float = 16.0,
        uncertainty_gate: float = 0.15,
        role_encoder_model: str = None,
    ):
        self.spec = spec
        self.critic = critic
        self.max_depth = max_depth
        self.hidden = hidden
        self.role_dim = role_dim
        self.num_layers = num_layers
        self.role_encoder_model = role_encoder_model
        self.refit_every = refit_every
        self.epochs = epochs
        # larger scale = slower decay of the uncertainty floor
        self.evidence_scale = max(1e-6, float(evidence_scale))
        self.uncertainty_gate = float(uncertainty_gate)
        # (unc, disagreement, evidence_floor) per predict() since the last refit
        self._unc_window: list[tuple[float, float, float]] = []
        self.device = _find_device(device)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.buffer = buffer if buffer is not None else ReplayBuffer(spec)

        self._pred_names = spec.predicted_names
        self._degenerate = len(self._pred_names) == 0
        if self._degenerate:
            # nothing to learn (all chosen axes are analytic): no net, no optimizer, no encoder.
            self.encoder = None
            self.model = None
            self.opt = None
        else:
            self.encoder = RoleEncoder(role_encoder_model, device=self.device)
            self.model = ge.build_net(
                hidden=hidden,
                role_input_dim=self.encoder.dim,
                role_dim=role_dim,
                output_dim=len(self._pred_names),
                num_layers=num_layers,
            ).to(self.device)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self._trained = False
        self.frozen = False
        log.info("GNN value predictor on device=%s (degenerate=%s)", self.device, self._degenerate)

    def _prior_vector(self, graph: WorkflowGraph) -> tuple[np.ndarray, float]:
        """Return the critic prior with analytic axes re-pinned, plus the critic's uncertainty.
        """
        est, unc = self.critic.value(graph)
        raw = self.spec.display(est)
        predicted = {n: raw[n] for n in self._pred_names}
        vec = analytic_vector(graph, self.spec, max_depth=self.max_depth, **predicted)
        return vec, float(unc)

    def predict(self, graph: WorkflowGraph, *, stochastic: bool = True) -> tuple[np.ndarray, float]:
        # if GNN is not trained yet, fall back to the analytic prior
        if self._degenerate or not self._trained or graph.is_empty():
            return self._prior_vector(graph)

        sample = ge.graph_to_tensors(graph, self.encoder.encode)
        batch = ge.collate([sample])
        inp = self._to_torch(batch)

        if stochastic:
            # MC-dropout: several stochastic forward passes -> mean + disagreement
            self.model.train()  # keep dropout active
            with torch.no_grad():
                preds = torch.stack([self.model(**inp) for _ in range(8)], dim=0)  # (8,1,P)
            mean = preds.mean(0).squeeze(0).cpu().numpy()
            std = preds.std(0).squeeze(0).cpu().numpy()
            # uncertainty among 8 MC-dropout realizations
            disagreement = float(np.clip(std.mean() * 4.0, 0.0, 1.0))
        else:
            # deterministic single eval-mode pass (dropout off)
            self.model.eval()
            with torch.no_grad():
                out = self.model(**inp)  # (1,P)
            mean = out.squeeze(0).cpu().numpy()
            disagreement = 0.0
        # map the head outputs back to their axis names.
        predicted = {n: float(mean[i]) for i, n in enumerate(self._pred_names)}

        # floor the uncertainty to avoid overconfidence when the buffer is small.
        evidence_floor = float(1.0 / (1.0 + len(self.buffer) / self.evidence_scale))
        unc = max(disagreement, evidence_floor)
        if stochastic and wandb is not None and wandb.run is not None:
            self._unc_window.append((float(unc), disagreement, evidence_floor))
        vec = analytic_vector(graph, self.spec, max_depth=self.max_depth, **predicted)
        return vec, unc

    def observe(self, graph: WorkflowGraph, target_vector: np.ndarray) -> None:
        if self._degenerate or self.frozen:
            return  # no learning target / frozen: net stays fixed, don't grow the buffer
        self.buffer.add(graph, target_vector)

    def maybe_refit(self) -> bool:
        if self._degenerate or self.frozen:
            return False
        # Use a smaller warm-up set to train the GNN on the first few observations, 
        # then switch to the configured refit_every.
        every = WARMUP_REFIT if not self._trained else self.refit_every
        if not self.buffer.due(every):
            return False
        self._fit()
        self.buffer.mark_refit()
        return True

    def _to_torch(self, batch: dict) -> dict:
        return {
            "node_feats": torch.tensor(batch["node_feats"]).to(self.device),
            "role_vecs": torch.tensor(batch["role_vecs"], dtype=torch.float32).to(self.device),
            "edge_index": torch.tensor(batch["edge_index"]).to(self.device),
            "batch": torch.tensor(batch["batch"]).to(self.device),
            "frontier_pos": torch.tensor(batch["frontier_pos"]).to(self.device),
            "num_graphs": batch["num_graphs"],
        }

    def _fit(self) -> None:
        pairs = [(g, y) for (g, y) in self.buffer.graphs() if not g.is_empty()]
        if len(pairs) < 8:
            return
        samples = [ge.graph_to_tensors(g, self.encoder.encode) for (g, _) in pairs]
        batch = ge.collate(samples)
        inp = self._to_torch(batch)
        yt = torch.tensor(np.stack([y for (_, y) in pairs])).to(self.device)

        self.model.to(self.device)
        self._align_optimizer_state()
        self.model.train()
        loss_fn = torch.nn.MSELoss()
        for _ in range(self.epochs):
            self.opt.zero_grad()
            pred = self.model(**inp)
            loss = loss_fn(pred, yt)
            loss.backward()
            self.opt.step()
        self._trained = True

        self._log_eval(train_pairs=pairs)

    def _metrics(self, pairs: list[tuple[WorkflowGraph, np.ndarray]]) -> tuple[dict, np.ndarray] | None:

        pairs = [(g, y) for (g, y) in pairs if not g.is_empty()]
        if not pairs:
            return None
        samples = [ge.graph_to_tensors(g, self.encoder.encode) for (g, _) in pairs]
        inp = self._to_torch(ge.collate(samples))
        yt = torch.tensor(np.stack([y for (_, y) in pairs])).to(self.device)

        self.model.eval() 
        with torch.no_grad():
            pred = self.model(**inp)
        self.model.train()

        p = pred.cpu().numpy()
        y = yt.cpu().numpy()
        err = p - y  # signed error (pred - target), shape (N, P)
        out: dict = {
            "n": float(len(pairs)),
            "loss": float(np.mean(err ** 2)),  # MSE, matches the training objective
            "mae": float(np.mean(np.abs(err))),
        }
        # R^2 = 1 - SS_res / SS_tot; undefined when targets have no variance.
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # per-axis R^2, named by the predicted-axis order the head outputs in.
        for i, name in enumerate(self._pred_names):
            tot_i = float(np.sum((y[:, i] - y[:, i].mean()) ** 2))
            res_i = float(np.sum(err[:, i] ** 2))
            out[f"r2/{name}"] = 1.0 - res_i / tot_i if tot_i > 0 else float("nan")
        return out, err

    def _log_eval(self, train_pairs: list[tuple[WorkflowGraph, np.ndarray]]) -> None:
        """Compute train + held-out test metrics for this refit and ship them to wandb."""
        if wandb is None or wandb.run is None:
            return
        train = self._metrics(train_pairs)
        test = self._metrics(self.buffer.test_graphs())
        metrics: dict = {"buffer/n_train": self.buffer.n_train,
                         "buffer/n_test": self.buffer.n_test}
        for prefix, m in (("train", train), ("test", test)):
            if m is None:
                continue
            scalars, err = m
            for k, v in scalars.items():
                metrics[f"{prefix}/{k}"] = v

            metrics[f"{prefix}/err"] = wandb.Histogram(err.reshape(-1))
            for i, name in enumerate(self._pred_names):
                metrics[f"{prefix}/err/{name}"] = wandb.Histogram(err[:, i])

        if self._unc_window:
            arr = np.array(self._unc_window, dtype=np.float64)  # (M, 3)
            u, dis, floor = arr[:, 0], arr[:, 1], arr[:, 2]
            metrics.update({
                "unc/mean": float(u.mean()),
                "unc/p50": float(np.median(u)),
                "unc/disagreement_mean": float(dis.mean()),
                "unc/evidence_floor_mean": float(floor.mean()),
                "unc/floor_binds_frac": float((floor >= dis).mean()),
                "unc/trusted_frac": float((u < self.uncertainty_gate).mean()),
                "unc/n_predicts": float(len(u)),
            })
            metrics["unc/hist"] = wandb.Histogram(u)
        self._unc_window = []

        wandb.log(metrics, step=self.buffer._refit_total)

    def _align_optimizer_state(self) -> None:
        if self.opt is None:
            return
        for group in self.opt.param_groups:
            keep_step_on_cpu = not (group.get("capturable", False) or group.get("fused", False))
            for p in group["params"]:
                st = self.opt.state.get(p)
                if not st:
                    continue
                for k, v in st.items():
                    if not torch.is_tensor(v):
                        continue
                    if k == "step" and keep_step_on_cpu:
                        st[k] = v.detach().cpu()
                    else:
                        st[k] = v.to(p.device)

    def state_dict(self) -> dict:
        """Return a picklable snapshot of the predictor: replay buffer + trained model.
        """
        state: dict = {
            "buffer": self.buffer,
            "trained": self._trained,
            "degenerate": self._degenerate,
        }
        if not self._degenerate:
            state["model"] = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            opt_state = self.opt.state_dict()
            for st in opt_state.get("state", {}).values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.detach().cpu()
            state["opt"] = opt_state
            if self.encoder is not None:
                state["role_cache"] = self.encoder.cache_state()
        return state

    def reconfigure(self, *, hidden: int = None, role_dim: int = None,
                    num_layers: int = None) -> None:
        """Rebuild the net to a different architecture (width/depth), in place.

        Called before loading a bundle whose stored architecture differs from this
        predictor's construction defaults
        """
        if self._degenerate or self.model is None:
            return
        new_hidden = int(hidden) if hidden is not None else self.hidden
        new_role_dim = int(role_dim) if role_dim is not None else self.role_dim
        new_layers = int(num_layers) if num_layers is not None else self.num_layers
        if (new_hidden, new_role_dim, new_layers) == (self.hidden, self.role_dim, self.num_layers):
            return
        log.info("reconfigure GNN net: hidden %d->%d, role_dim %d->%d, layers %d->%d",
                 self.hidden, new_hidden, self.role_dim, new_role_dim, self.num_layers, new_layers)
        self.hidden, self.role_dim, self.num_layers = new_hidden, new_role_dim, new_layers
        self.model = ge.build_net(
            hidden=new_hidden,
            role_input_dim=self.encoder.dim,
            role_dim=new_role_dim,
            output_dim=len(self._pred_names),
            num_layers=new_layers,
        ).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def load_state_dict(self, state: dict) -> None:
        """Restore a `state_dict()` snapshot onto this predictor (in place).
        """
        self.buffer = state["buffer"]
        self._trained = bool(state.get("trained", False))
        if self._degenerate or "model" not in state:
            return
        if self.encoder is not None and "role_cache" in state:
            self.encoder.load_cache(state["role_cache"])
        try:
            self.model.load_state_dict(_migrate_legacy_mp_keys(state["model"]))
            self.model.to(self.device)
            self.opt.load_state_dict(state["opt"])
            self._align_optimizer_state()
        except (RuntimeError, ValueError, KeyError) as exc:
            log.warning(
                "GNN weights incompatible with current model (%s); keeping restored buffer, model stays fresh and will retrain", exc,
            )
            self._trained = False