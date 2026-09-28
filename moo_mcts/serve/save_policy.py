import json
import os
import pickle
import re

import numpy as np

from ..logging_util import get_logger
from ..objectives import ObjectiveSpec
from ..workflow.serialize_policy import policy_to_dict

log = get_logger("save_policy")

RESULTS_ROOT = "results"

BUNDLE_VERSION = 1

def _slug(name: str) -> str:
    """Filename-safe slug for a preference name (e.g. 'accuracy-max')."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return s or "policy"

def result_dir(dataset_name: str, mode: str, model: str = "") -> str:
    model_slug = _slug(model)
    path = os.path.join(RESULTS_ROOT, dataset_name, model_slug, mode)
    os.makedirs(path, exist_ok=True)
    return path

def policy_dump_dict(
    policy, spec: ObjectiveSpec, w, name, heldout_res=None, n_heldout: int = 0
) -> dict:

    w_arr = np.asarray(w, dtype=float)
    served_value = None
    if policy is not None and policy.value is not None:
        served_value = {
            k: round(float(v), 6) for k, v in spec.display(policy.value).items()
        }
    heldout = None
    if heldout_res is not None:
        heldout = {
            "n_problems": int(n_heldout),
            "raw": {
                k: round(float(v), 6)
                for k, v in heldout_res.raw.items()
                if k in spec.names
            },
        }
    return {
        "preference": {
            "name": str(name),
            "w": [round(float(x), 6) for x in w_arr],
            "axes": list(spec.names),
        },
        "served_value": served_value,
        "heldout": heldout,
        "policy": (policy_to_dict(policy) if policy is not None else None),
    }


def dump_policy(
    policy, spec: ObjectiveSpec, w, name, heldout_res=None, n_heldout: int = 0,
    path: str = None, out_dir: str = RESULTS_ROOT,
) -> str:

    d = policy_dump_dict(
        policy, spec, w, name, heldout_res=heldout_res, n_heldout=n_heldout
    )
    out = path or os.path.join(out_dir, f"policy_{_slug(name)}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)
    return out

def default_bundle_path(model: str, spec: ObjectiveSpec) -> str:
    """Shared GNN-bundle path (model weights + replay buffer)
    """
    model_slug = _slug(model)
    axes = "-".join(spec.predicted_names) or "none"
    path = os.path.join(RESULTS_ROOT, "gnn_bundles", model_slug)
    return os.path.join(path, f"{axes}.pkl")


def _bundle_meta(predictor, spec: ObjectiveSpec, trials: int | None) -> dict:
    buf = getattr(predictor, "buffer", None)
    return {
        "bundle_version": BUNDLE_VERSION,
        "predictor": type(predictor).__name__,
        "predicted_names": list(spec.predicted_names),
        "objective_names": list(spec.names),
        "hidden": getattr(predictor, "hidden", None),
        "role_dim": getattr(predictor, "role_dim", None),
        "num_layers": getattr(predictor, "num_layers", None),
        "role_encoder_model": getattr(predictor, "role_encoder_model", None),
        "role_input_dim": getattr(getattr(predictor, "encoder", None), "dim", None),
        "trained": bool(getattr(predictor, "_trained", False)),
        "refit_total": int(getattr(buf, "_refit_total", 0)) if buf is not None else 0,
        "n_train": int(buf.n_train) if buf is not None else 0,
        "n_test": int(buf.n_test) if buf is not None else 0,
        "trials": int(trials) if trials is not None else None,
    }


def save_bundle(path: str | None, predictor, spec: ObjectiveSpec, *, trials: int = None) -> None:

    if not path or not hasattr(predictor, "state_dict"):
        return
    bundle = {"meta": _bundle_meta(predictor, spec, trials), "state": predictor.state_dict()}
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    m = bundle["meta"]
    log.info(
        "saved GNN bundle %s (trained=%s n_train=%d n_test=%d refit_total=%d trials=%s)",
        path, m["trained"], m["n_train"], m["n_test"], m["refit_total"], m["trials"],
    )


def load_bundle(path: str | None, predictor, spec: ObjectiveSpec) -> dict | None:
    """Restore a bundle onto `predictor` in place; return its meta (or None if skipped).
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
    except Exception as exc:  # corrupt / partial / incompatible pickle
        log.warning("could not load GNN bundle %s (%s); ignoring", path, exc)
        return None
    meta = bundle.get("meta", {}) if isinstance(bundle, dict) else {}
    if list(meta.get("predicted_names", [])) != list(spec.predicted_names):
        log.warning(
            "bundle axes %s != current %s; ignoring bundle",
            meta.get("predicted_names"), list(spec.predicted_names),
        )
        return None
    
    if hasattr(predictor, "reconfigure"):
        predictor.reconfigure(
            hidden=meta.get("hidden"),
            role_dim=meta.get("role_dim"),
            num_layers=meta.get("num_layers"),
        )
    if hasattr(predictor, "load_state_dict"):
        predictor.load_state_dict(bundle["state"])
    buf = getattr(predictor, "buffer", None)

    # force an immediate refit for a fresh predictor that inherited a non-empty buffer
    if buf is not None:
        buf.request_refit()

    log.info(
        "loaded GNN bundle %s (trained=%s n_train=%d n_test=%d refit_total=%d)",
        path,
        getattr(predictor, "_trained", meta.get("trained")),
        buf.n_train if buf is not None else 0,
        buf.n_test if buf is not None else 0,
        getattr(buf, "_refit_total", 0) if buf is not None else 0,
    )
    return meta
