import numpy as np

from ..logging_util import get_logger

log = get_logger("role_encoder")

# edge characters stripped from both ends of a role label (quotes + punctuation + space)
_EDGE_CHARS = " \t\r\n\"'`.,;:!?-_()[]{}"


def canonicalize_role(role: str) -> str:
    """Normalize a free-text role label so surface variants collapse to one key.
    """
    if role is None:
        return "default"
    s = " ".join(str(role).strip().lower().split())    # trim + collapse whitespace
    s = s.strip(_EDGE_CHARS)                           # drop edge quotes/punctuation (any order)
    s = " ".join(s.split())
    return s or "default"


class RoleEncoder:
    """Map a role string to a fixed-length semantic vector, cached per canonical role."""

    def __init__(self, model_name: str | None = None, device=None):
        self._cache: dict[str, np.ndarray] = {}
        self.model_name = model_name
        self.device = device
        if not model_name:
            raise ValueError("RoleEncoder requires a sentence-transformers model name")
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device=self.device)
        except Exception as exc:  # missing dep, no network, bad model id, etc.
            raise RuntimeError(
                f"could not load role encoder '{model_name}' ({exc}); install the 'learned' "
                "extra (sentence-transformers) and ensure the model is available"
            ) from exc

        dim_fn = getattr(self._model, "get_embedding_dimension")
        self.dim = int(dim_fn())
        log.info("role encoder: sentence-transformers '%s' on device=%s (dim=%d)",
                 model_name, self._model.device, self.dim)

    def encode(self, role: str) -> np.ndarray:
        key = canonicalize_role(role)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        vec = np.asarray(self._model.encode(key, normalize_embeddings=True), dtype=np.float32)
        self._cache[key] = vec
        return vec

    def cache_state(self) -> dict:
        return {"dim": self.dim, "vectors": dict(self._cache)}

    def load_cache(self, state: dict) -> None:
        if not state:
            return
        if int(state.get("dim", -1)) != self.dim:
            log.warning("role cache dim %s != encoder dim %d; ignoring persisted role cache",
                        state.get("dim"), self.dim)
            return
        for k, v in (state.get("vectors") or {}).items():
            self._cache[k] = np.asarray(v, dtype=np.float32)
