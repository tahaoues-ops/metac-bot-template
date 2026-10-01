"""
Config loading. One YAML file holds every setting; experiment configs inherit
from it with `extends:` and override only what they change.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


class Config(dict):
    """A dict with dotted-path access: cfg.get_path("critique.rounds")."""

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def name(self) -> str:
        return str(self.get("name", "unnamed"))

    def fingerprint(self) -> str:
        """Short hash of everything that affects forecasts (used in logs)."""
        relevant = {k: v for k, v in self.items() if k not in ("name", "eval", "output")}
        blob = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def forecaster_fingerprint(self) -> str:
        """
        Hash of what affects each individual forecaster's LLM calls. Excludes aggregation
        (recomputed from stored member predictions) and the NUMBER of forecasters (each
        forecaster is independent, so a 5-forecaster run contains the 1- and 3-forecaster runs).
        Eval runs are stored by this hash, so such configs share LLM work.
        """
        relevant = {k: copy.deepcopy(v) for k, v in self.items()
                    if k not in ("name", "eval", "output", "aggregation", "publish_to_metaculus")}
        relevant.get("ensemble", {}).pop("total_forecasters", None)
        relevant.get("models", {}).pop("leak_judge", None)
        relevant.get("summary", {}).pop("enabled", None)
        blob = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict) and value:
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_config_path(name_or_path: str) -> Path:
    """Accepts a path, or a bare name looked up in configs/ and configs/ablations/."""
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    for folder in (CONFIG_DIR, CONFIG_DIR / "ablations"):
        for suffix in ("", ".yaml", ".yml"):
            p = folder / f"{name_or_path}{suffix}"
            if p.exists():
                return p
    raise FileNotFoundError(f"Config not found: {name_or_path}")


def load_config(name_or_path: str | Path | None = None, _depth: int = 0) -> Config:
    if _depth > 10:
        raise ValueError("Config `extends` chain is too deep (cycle?)")
    path = resolve_config_path(str(name_or_path)) if name_or_path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    own_name = data.get("name") or path.stem  # a child config never inherits its parent's name
    parent = data.pop("extends", None)
    if parent:
        parent_path = path.parent / parent
        if not parent_path.exists():
            parent_path = resolve_config_path(parent)
        base = load_config(parent_path, _depth + 1)
        data = _deep_merge(dict(base), data)
    data["name"] = own_name
    cfg = Config(data)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    r = cfg.get_path
    lo, hi = r("research.min_iterations"), r("research.max_iterations")
    if not (1 <= lo <= hi):
        raise ValueError(f"research.min_iterations ({lo}) must be >=1 and <= max_iterations ({hi})")
    if r("ensemble.total_forecasters", 0) < 1 or not r("ensemble.members"):
        raise ValueError("ensemble needs total_forecasters >= 1 and at least one member")
    allowed = {
        "aggregation.binary": {"median", "mean", "weighted_logodds"},
        "aggregation.multiple_choice": {"median", "mean", "weighted_logodds"},
        "aggregation.numeric": {"weighted_quantile_mean", "median_quantile"},
        "prompt_style": {"structured", "simple"},
        "checks.verification": {"off", "flag", "verify"},
    }
    for path, options in allowed.items():
        if r(path) not in options:
            raise ValueError(f"{path} must be one of {sorted(options)}, got {r(path)!r}")
    names = [m["name"] for m in r("ensemble.members")]
    if len(set(names)) != len(names):
        raise ValueError("ensemble member names must be unique")
