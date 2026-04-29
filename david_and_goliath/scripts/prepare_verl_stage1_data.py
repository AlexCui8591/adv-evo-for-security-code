"""Prepare veRL-format prompt data for offline Stage-1 GRPO.

veRL expects RLHF data as parquet/json rows with at least:
  data_source, prompt, ability, reward_model, extra_info
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_PACKAGE_ROOT = _HERE.parent.parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent
for _path in (_PROJECT_ROOT, _PACKAGE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from core.injection_engine import load_coding_tasks
from core.strategy_db import MAPElitesDB
from core.types import Carrier
from red_team.prompt_builder import RedPromptBuilder


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "total_rounds": 1,
    "coding_tasks_path": "david_and_goliath/data/coding_tasks/tasks.jsonl",
    "output_dir": "outputs/verl_grpo_8b",
    "stage1": {
        "num_prompts": 512,
        "val_num_prompts": 64,
        "disable_qwen_thinking": True,
        "blue_behavior_summary": "No previous offline Blue Team data is available.",
        "carriers": ["NATURAL_LANGUAGE", "CODE_COMMENT", "DOCSTRING", "MARKDOWN"],
    },
    "red_team": {
        "prompt_builder": {},
    },
}


def _load_yaml(path: str) -> dict[str, Any]:
    if not _YAML_AVAILABLE:
        raise RuntimeError("PyYAML is required to load --config. Install: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _coerce_carriers(names: list[str]) -> list[Carrier]:
    carriers: list[Carrier] = []
    for name in names:
        try:
            carriers.append(Carrier[str(name)])
        except KeyError as exc:
            valid = ", ".join(c.name for c in Carrier)
            raise ValueError(f"Unknown carrier {name!r}. Valid values: {valid}") from exc
    return carriers


def _make_rows(config: dict[str, Any], split: str, count: int) -> list[dict[str, Any]]:
    rng = random.Random(int(config.get("seed", 42)) + (0 if split == "train" else 10_000))
    tasks = load_coding_tasks(config["coding_tasks_path"])
    if not tasks:
        raise ValueError(f"No coding tasks found in {config['coding_tasks_path']}")

    stage1_cfg = config.get("stage1", {})
    total_rounds = max(1, int(config.get("total_rounds") or 1))
    carriers = _coerce_carriers(stage1_cfg.get("carriers") or [c.name for c in Carrier])
    strategy_db = MAPElitesDB(
        niche_capacity=int(config.get("niche_capacity", 5)),
        seed=int(config.get("seed", 42)),
    )
    prompt_builder = RedPromptBuilder(
        config=config.get("red_team", {}).get("prompt_builder", {}),
    )
    blue_summary = str(stage1_cfg.get("blue_behavior_summary") or "")

    rows: list[dict[str, Any]] = []
    for index in range(count):
        round_num = (index % total_rounds) + 1
        parents, inspirations = strategy_db.sample()
        messages = prompt_builder.build(
            parents=parents,
            inspirations=inspirations,
            blue_summary=blue_summary,
            round_num=round_num,
        )
        if stage1_cfg.get("disable_qwen_thinking", True):
            messages[-1]["content"] = f"{messages[-1]['content']}\n\n/no_think"
        task = rng.choice(tasks)
        carrier = rng.choice(carriers)
        rows.append(
            {
                "data_source": "david_and_goliath_stage1",
                "prompt": messages,
                "ability": "prompt_injection_payload",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "",
                },
                "extra_info": {
                    "split": split,
                    "index": index,
                    "round": round_num,
                    "task_id": task.id,
                    "injection_position": carrier.name,
                },
            }
        )
    return rows


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported data output suffix: {path.suffix}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and pyarrow are required for parquet output") from exc

    pd.DataFrame(rows).to_parquet(path, index=False)


def prepare_datasets(
    config: dict[str, Any],
    train_path: Path,
    val_path: Path,
) -> tuple[int, int]:
    train_rows = _make_rows(
        config=config,
        split="train",
        count=int(config.get("stage1", {}).get("num_prompts", 512)),
    )
    val_rows = _make_rows(
        config=config,
        split="val",
        count=int(config.get("stage1", {}).get("val_num_prompts", 64)),
    )
    _write_rows(train_rows, train_path)
    _write_rows(val_rows, val_path)
    return len(train_rows), len(val_rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare veRL Stage-1 GRPO parquet/jsonl prompt data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.config:
        config = _deep_merge(config, _load_yaml(args.config))
    if args.output_dir:
        config["output_dir"] = args.output_dir

    data_dir = Path(config["output_dir"]) / "verl_data"
    train_path = Path(args.train_path) if args.train_path else data_dir / "train.parquet"
    val_path = Path(args.val_path) if args.val_path else data_dir / "val.parquet"
    train_n, val_n = prepare_datasets(config, train_path, val_path)
    logging.info("Wrote veRL data: train=%s (%d), val=%s (%d)", train_path, train_n, val_path, val_n)


if __name__ == "__main__":
    main()
