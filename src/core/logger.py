import json
import logging
import os
import sys
import time
from contextlib import contextmanager


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_metrics"):
            payload["metrics"] = record.extra_metrics
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _StdoutHandler(logging.StreamHandler):
    """StreamHandler that always writes to the *current* ``sys.stdout``.

    A plain ``StreamHandler(sys.stdout)`` keeps the stream that existed when the logger was
    created, which breaks when stdout is later replaced (pytest capture, notebooks, ...).
    """

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:  # StreamHandler.__init__ assigns it; the value is ignored
        pass


def get_logger(name: str = "wikimatrix_nmt") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = _StdoutHandler()
        handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z"))
        logger.addHandler(handler)
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        logger.propagate = False  # Prevent duplicate logs from root logger
    return logger


@contextmanager
def track_latency(logger: logging.Logger, operation: str):
    start = time.perf_counter()
    metrics = {"operation": operation, "latency_ms": 0.0}
    try:
        yield metrics
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        metrics["latency_ms"] = elapsed_ms
        logger.info(
            "Completed %s in %s ms",
            operation,
            elapsed_ms,
            extra={"extra_metrics": metrics},
        )
