from __future__ import annotations

import structlog


def get_logger(name: str):
    """Structured logger, drop-in for ProjectClipper's get_logger.
    Returns a structlog BoundLogger supporting `log.info("event", key=value)`."""
    return structlog.get_logger(name)
