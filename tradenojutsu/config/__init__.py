"""Configuration management for TradeNoJutsu."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


_CONFIG_DIR = Path(__file__).parent
_settings_cache: dict[str, Any] | None = None


def load_settings(config_path: Path | None = None) -> dict[str, Any]:
    """Load settings from YAML config file."""
    global _settings_cache
    if _settings_cache is not None and config_path is None:
        return _settings_cache

    path = config_path or _CONFIG_DIR / "settings.yaml"
    with open(path) as f:
        settings = yaml.safe_load(f)

    # Load env vars
    env_path = _CONFIG_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Override with environment variables
    if api_key := os.getenv("ANTHROPIC_API_KEY"):
        settings.setdefault("secrets", {})["anthropic_api_key"] = api_key
    if mode := os.getenv("TRADENOJUTSU_MODE"):
        settings["agent"]["mode"] = mode

    if config_path is None:
        _settings_cache = settings
    return settings


def save_settings(settings: dict[str, Any], config_path: Path | None = None) -> None:
    """Save settings back to YAML (used by self-learning module)."""
    path = config_path or _CONFIG_DIR / "settings.yaml"
    with open(path, "w") as f:
        yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
    global _settings_cache
    _settings_cache = None
