"""
Configuration loader for Analytics Copilot.

Loads settings from config.yaml and applies deterministic seed.
"""

import os
import random
import yaml
from pathlib import Path

_config = None
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config(config_path: str = None) -> dict:
    """Load and cache configuration from config.yaml."""
    global _config
    if _config is not None:
        return _config

    if config_path is None:
        config_path = _PROJECT_ROOT / "config.yaml"

    with open(config_path, "r") as f:
        _config = yaml.safe_load(f)

    # Apply deterministic seed
    seed = _config.get("seed", 42)
    random.seed(seed)

    return _config


def get_config() -> dict:
    """Return cached config, loading if needed."""
    if _config is None:
        return load_config()
    return _config
