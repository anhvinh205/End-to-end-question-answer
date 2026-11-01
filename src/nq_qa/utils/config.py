"""YAML configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "configs/config.yaml"
CONFIG_ENV_VAR = "NQ_QA_CONFIG"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config.

    Resolution order: explicit argument, ``$NQ_QA_CONFIG``, ``configs/config.yaml``.
    """
    path = Path(config_path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return cfg
