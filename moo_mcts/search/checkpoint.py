import os
import pickle

from ..logging_util import get_logger

log = get_logger("checkpoint")


class Checkpointer:
    """Atomic pickle-based snapshot store for one search run.

    - path: file the snapshot is written to (parent dirs created on first save).
    - every: write a checkpoint every N trials (1 = after every trial).
    """

    def __init__(self, path: str, every: int = 1):
        self.path = path
        self.every = max(1, int(every))

    def exists(self) -> bool:
        """True if a committed checkpoint is present to resume from."""
        return bool(self.path) and os.path.exists(self.path)

    def save(self, snapshot: dict) -> None:
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(snapshot, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        log.debug("checkpoint written -> %s (trial=%s)", self.path, snapshot.get("total_trials"))

    def load(self) -> dict:
        """Read back the committed snapshot dict."""
        with open(self.path, "rb") as fh:
            return pickle.load(fh)
