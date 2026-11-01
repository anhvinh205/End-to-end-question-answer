"""Logging helpers shared by the CLI scripts and the API."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once (idempotent)."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
