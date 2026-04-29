"""Single-entry orchestration for the offline David & Goliath pipeline.

Default flow:
  Stage 1: veRL GRPO payload generation
  Stage 2: offline Blue Team batch run
  Stage 3: offline judging + memory writeback
  Stage 4: defense-memory distillation for future Blue Team runs
  Stage 5: unified offline analysis / visualization

This script is intentionally thin: it only resolves paths / configs and
dispatches the existing stage scripts in order.
"""

from __future__ import annotations

import argparse
import copy
import logging
import subprocess
import sys
import time
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


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_id": "coevo_default",
    "output_dir": None,
    "config": None,
    "blue_config": "david_and_goliath/configs/blue_team/full_tools.yaml",
    "oracle_config": "david_and_goliath/configs/oracle/hybrid_oracle.yaml",
    "tasks_path": None,
    "ood_tasks_path": None,
    "total_rounds": None,
    "seed": None,
    "concurrency": 4,
    "limit_stage2": None,
    "limit_stage3": None,
    "blue_defense_memory_path": None,
    "defense_retrieval_top_k": None,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="David & Goliath - Stage 1/2/3 pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional Stage-1 training config YAML.",
    )
    parser.add_argument(
        "--blue-config",
        type=str,
        default=None,
        help="Optional Stage-2 blue-team config YAML.",
    )
    parser.add_argument(
        "--oracle-config",
        type=str,
        default=None,
        help="Optional Stage-3 oracle config YAML.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment id used by Stage-1 and default output paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root output directory shared by all stages.",
    )
    parser.add_argument(
        "--tasks-path",
        type=str,
        default=None,
        help="Override coding_tasks_path for all stages.",
    )
    parser.add_argument(
        "--ood-tasks-path",
        type=str,
        default=None,
        help="Optional OOD coding_tasks_path used by the final analysis stage.",
    )
    parser.add_argument(
        "--total-rounds",
        type=int,
        default=None,
        help="Override Stage-1 total_rounds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override shared random seed.",
    )
    parser.add_argument(
        "--rollouts-path",
        type=str,
        default=None,
        help="Explicit Stage-1 rollout path.",
    )
    parser.add_argument(
        "--blue-responses-path",
        type=str,
        default=None,
        help="Explicit Stage-2 output path / Stage-3 input path.",
    )
    parser.add_argument(
        "--episodes-path",
        type=str,
        default=None,
        help="Explicit Stage-3 memory episodes path.",
    )
    parser.add_argument(
        "--blue-defense-memory-path",
        type=str,
        default=None,
        help="Output path for distilled Blue defense memory.",
    )
    parser.add_argument(
        "--defense-retrieval-top-k",
        type=int,
        default=None,
        help="Override how many historical defense memories Stage-2 injects.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override Stage-2 concurrency.",
    )
    parser.add_argument(
        "--limit-stage2",
        type=int,
        default=None,
        help="Debug knob: process at most N Stage-2 pending rows.",
    )
    parser.add_argument(
        "--limit-stage3",
        type=int,
        default=None,
        help="Debug knob: process at most N Stage-3 pending rows.",
    )
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")
    parser.add_argument("--skip-stage4", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands and validate inputs without executing stages.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


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


def _infer_output_dir(config: dict[str, Any], stage1_cfg: dict[str, Any]) -> str:
    if config.get("output_dir"):
        return str(config["output_dir"])
    if stage1_cfg.get("output_dir"):
        return str(stage1_cfg["output_dir"])
    if config.get("experiment_id"):
        return str(Path("outputs") / str(config["experiment_id"]))
    return str(Path("outputs") / "coevo_default")


def _build_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(DEFAULT_CONFIG)

    stage1_cfg: dict[str, Any] = {}
    if args.config:
        stage1_cfg = _load_yaml(args.config)
        config = _deep_merge(config, {"config": args.config})
        for key in ("experiment_id", "output_dir", "total_rounds", "seed"):
            if key in stage1_cfg:
                config[key] = stage1_cfg[key]
        if "coding_tasks_path" in stage1_cfg:
            config["tasks_path"] = stage1_cfg["coding_tasks_path"]

    if args.blue_config is not None:
        config["blue_config"] = args.blue_config
    if args.oracle_config is not None:
        config["oracle_config"] = args.oracle_config
    if args.experiment_id is not None:
        config["experiment_id"] = args.experiment_id
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.tasks_path is not None:
        config["tasks_path"] = args.tasks_path
    if args.ood_tasks_path is not None:
        config["ood_tasks_path"] = args.ood_tasks_path
    if args.total_rounds is not None:
        config["total_rounds"] = args.total_rounds
    if args.seed is not None:
        config["seed"] = args.seed
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency
    if args.blue_defense_memory_path is not None:
        config["blue_defense_memory_path"] = args.blue_defense_memory_path
    if args.defense_retrieval_top_k is not None:
        config["defense_retrieval_top_k"] = args.defense_retrieval_top_k
    if args.limit_stage2 is not None:
        config["limit_stage2"] = args.limit_stage2
    if args.limit_stage3 is not None:
        config["limit_stage3"] = args.limit_stage3

    output_dir = Path(_infer_output_dir(config, stage1_cfg))
    config["output_dir"] = str(output_dir)
    config["rollouts_path"] = args.rollouts_path or str(output_dir / "rollouts" / "rollouts.jsonl")
    config["blue_responses_path"] = args.blue_responses_path or str(
        output_dir / "blue_team" / "blue_responses.jsonl"
    )
    config["episodes_path"] = args.episodes_path or str(
        output_dir / "memory" / "episodes.jsonl"
    )
    config["blue_defense_memory_path"] = args.blue_defense_memory_path or str(
        output_dir / "memory" / "blue_defense_memory.jsonl"
    )
    return config, stage1_cfg


def _setup_logging(level: str, output_dir: str) -> None:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=getattr(logging, level), format=fmt, handlers=handlers)


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def _require_exists(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _build_stage1_cmd(config: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "-m", "david_and_goliath.scripts.run_stage1_verl"]
    if config.get("config"):
        cmd += ["--config", str(config["config"])]
    if config.get("experiment_id"):
        cmd += ["--experiment-id", str(config["experiment_id"])]
    if config.get("output_dir"):
        cmd += ["--output-dir", str(config["output_dir"])]
    if config.get("tasks_path"):
        cmd += ["--tasks-path", str(config["tasks_path"])]
    if config.get("total_rounds") is not None:
        cmd += ["--total-rounds", str(config["total_rounds"])]
    if config.get("seed") is not None:
        cmd += ["--seed", str(config["seed"])]
    if config.get("rollouts_path"):
        cmd += ["--rollouts-path", str(config["rollouts_path"])]
    cmd += ["--log-level", str(config["log_level"])]
    return cmd


def _build_stage2_cmd(config: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "-m", "david_and_goliath.scripts.run_offline_blue_team"]
    cmd += ["--rollouts-path", str(config["rollouts_path"])]
    cmd += ["--output-path", str(config["blue_responses_path"])]
    if config.get("blue_config"):
        cmd += ["--config", str(config["blue_config"])]
    if config.get("tasks_path"):
        cmd += ["--tasks-path", str(config["tasks_path"])]
    if config.get("concurrency") is not None:
        cmd += ["--concurrency", str(config["concurrency"])]
    if config.get("blue_defense_memory_path"):
        cmd += ["--defense-memory-path", str(config["blue_defense_memory_path"])]
    if config.get("defense_retrieval_top_k") is not None:
        cmd += [
            "--defense-retrieval-top-k",
            str(config["defense_retrieval_top_k"]),
        ]
    if config.get("limit_stage2") is not None:
        cmd += ["--limit", str(config["limit_stage2"])]
    cmd += ["--log-level", str(config["log_level"])]
    return cmd


def _build_stage3_cmd(config: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "-m", "david_and_goliath.scripts.run_offline_judging"]
    cmd += ["--rollouts-path", str(config["rollouts_path"])]
    cmd += ["--blue-responses-path", str(config["blue_responses_path"])]
    cmd += ["--output-path", str(config["episodes_path"])]
    if config.get("oracle_config"):
        cmd += ["--config", str(config["oracle_config"])]
    if config.get("tasks_path"):
        cmd += ["--tasks-path", str(config["tasks_path"])]
    if config.get("limit_stage3") is not None:
        cmd += ["--limit", str(config["limit_stage3"])]
    cmd += ["--log-level", str(config["log_level"])]
    return cmd


def _build_stage4_cmd(config: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "-m", "david_and_goliath.scripts.run_offline_defense_memory"]
    cmd += ["--episodes-path", str(config["episodes_path"])]
    cmd += ["--output-path", str(config["blue_defense_memory_path"])]
    cmd += ["--log-level", str(config["log_level"])]
    return cmd


def _build_analysis_cmd(config: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "-m", "david_and_goliath.scripts.run_offline_analysis"]
    cmd += ["--experiment-dir", str(config["output_dir"])]
    cmd += ["--output-dir", str(Path(config["output_dir"]) / "analysis")]
    if config.get("blue_config"):
        cmd += ["--blue-config", str(config["blue_config"])]
    if config.get("oracle_config"):
        cmd += ["--oracle-config", str(config["oracle_config"])]
    if config.get("tasks_path"):
        cmd += ["--tasks-path", str(config["tasks_path"])]
    if config.get("ood_tasks_path"):
        cmd += ["--ood-tasks-path", str(config["ood_tasks_path"])]
    if config.get("blue_defense_memory_path"):
        cmd += ["--blue-defense-memory-path", str(config["blue_defense_memory_path"])]
    cmd += ["--log-level", str(config["log_level"])]
    return cmd


def _run_stage(name: str, cmd: list[str], dry_run: bool) -> None:
    logger = logging.getLogger(__name__)
    logger.info("%s command: %s", name, _quote_cmd(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    config, _stage1_cfg = _build_config(args)
    config["log_level"] = args.log_level
    _setup_logging(args.log_level, config["output_dir"])
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("David & Goliath - Pipeline")
    logger.info("=" * 60)
    logger.info("Experiment      : %s", config["experiment_id"])
    logger.info("Output dir      : %s", config["output_dir"])
    logger.info("Stage-1 backend : veRL GRPO")
    logger.info("Rollouts path   : %s", config["rollouts_path"])
    logger.info("Blue responses  : %s", config["blue_responses_path"])
    logger.info("Episodes path   : %s", config["episodes_path"])
    logger.info("Defense memory  : %s", config["blue_defense_memory_path"])
    logger.info("Defense top-k   : %s", config.get("defense_retrieval_top_k"))
    logger.info(
        "Skip flags      : stage1=%s stage2=%s stage3=%s stage4=%s",
        args.skip_stage1,
        args.skip_stage2,
        args.skip_stage3,
        args.skip_stage4,
    )
    logger.info("Skip analysis   : %s", args.skip_analysis)
    logger.info("Dry run         : %s", args.dry_run)

    if not args.skip_stage1:
        _run_stage("Stage 1", _build_stage1_cmd(config), args.dry_run)
    elif not args.dry_run:
        _require_exists(config["rollouts_path"], "Stage-1 rollouts")

    if not args.skip_stage2:
        if not args.dry_run:
            _require_exists(config["rollouts_path"], "Stage-1 rollouts")
        _run_stage("Stage 2", _build_stage2_cmd(config), args.dry_run)
    elif not args.dry_run and not args.skip_stage3:
        _require_exists(config["blue_responses_path"], "Stage-2 blue responses")

    if not args.skip_stage3:
        if not args.dry_run:
            _require_exists(config["rollouts_path"], "Stage-1 rollouts")
            _require_exists(config["blue_responses_path"], "Stage-2 blue responses")
        _run_stage("Stage 3", _build_stage3_cmd(config), args.dry_run)

    if not args.skip_stage4:
        if not args.dry_run:
            _require_exists(config["episodes_path"], "Stage-3 episodes")
        _run_stage("Stage 4", _build_stage4_cmd(config), args.dry_run)

    if not args.skip_analysis:
        if not args.dry_run:
            _require_exists(config["rollouts_path"], "Stage-1 rollouts")
            _require_exists(config["episodes_path"], "Stage-3 episodes")
        _run_stage("Analysis", _build_analysis_cmd(config), args.dry_run)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
