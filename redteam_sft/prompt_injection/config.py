"""CLI argument parsing and config file loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt injection benchmark")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML or JSON config file",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Optional override for the benchmark JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of cases to run",
    )
    parser.add_argument(
        "--attacker",
        type=str,
        action="append",
        default=None,
        help="Optional attacker id filter; can be passed multiple times",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    with open(config_path, "r", encoding="utf-8") as f:
        if suffix == ".json":
            return json.load(f)
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "PyYAML is required for YAML configs. Install pyyaml or use JSON."
                ) from exc
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported config format: {path}")
