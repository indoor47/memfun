from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", json_output: bool = False) -> logging.Logger:
    """Configure and return the root memfun logger."""
    logger = logging.getLogger("memfun")

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stderr)

    if json_output:
        import json

        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps({
                    "ts": record.created,
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                    **({"exc": self.formatException(record.exc_info)} if record.exc_info else {}),
                })

        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the memfun namespace."""
    return logging.getLogger(f"memfun.{name}")
