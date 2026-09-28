
import logging
import sys

_FORMAT = "%(levelname)-5s %(name)s | %(message)s"

def _ensure_handler() -> logging.Logger:
    """Attach the stderr handler to the `moo_mcts` logger exactly once.
    """
    logger = logging.getLogger("moo_mcts")
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel("INFO")  # default until configure() sets an explicit level
    return logger


def configure(level: str = "INFO") -> None:
    """Set the log level for the whole `moo_mcts` namespace (CLI entry point).
    """
    _ensure_handler().setLevel(level)


def get_logger(name: str) -> logging.Logger:
    _ensure_handler()
    return logging.getLogger(f"moo_mcts.{name}")
