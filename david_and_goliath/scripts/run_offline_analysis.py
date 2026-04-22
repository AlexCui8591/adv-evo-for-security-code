"""Unified offline evaluation and visualization for Red/Blue runs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from core.injection_engine import InjectionEngine, load_coding_tasks
from evaluation.cross_evaluator import CrossEvaluator, load_payloads_from_checkpoint
from evaluation.metrics import aggregate_metrics
from evaluation.ood_evaluator import OODEvaluator
from scripts import run_cross_eval as cross_eval_script
from scripts import run_stage1_eval as stage1_eval_script


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_dir": None,
    "output_dir": None,
    "blue_config": "david_and_goliath/configs/blue_team/full_tools.yaml",
    "oracle_config": "david_and_goliath/configs/oracle/hybrid_oracle.yaml",
    "tasks_path": "david_and_goliath/data/coding_tasks/tasks.jsonl",
    "ood_tasks_path": None,
    "n_tasks_per_payload": 3,
    "top_k_payloads": None,
    "fpr_samples": 20,
    "seed": 42,
    "stage1_top_k": 20,
    "stage1_topk_buckets": [10, 50, 100],
    "blue_defense_memory_path": None,
    "force_recompute": False,
    "skip_plots": False,
}

BLUE_VARIANTS: list[dict[str, Any]] = [
    {
        "name": "llm_no_tool_calls_no_memory",
        "label": "LLM only",
        "use_tools": False,
        "enable_static_memory_scan": False,
        "enable_defense_memory_retrieval": False,
        "use_defense_memory_path": False,
    },
    {
        "name": "tool_calls_no_memory",
        "label": "Tools only",
        "use_tools": True,
        "enable_static_memory_scan": False,
        "enable_defense_memory_retrieval": False,
        "use_defense_memory_path": False,
    },
    {
        "name": "tool_calls_static_memory",
        "label": "Tools + static memory",
        "use_tools": True,
        "enable_static_memory_scan": True,
        "enable_defense_memory_retrieval": False,
        "use_defense_memory_path": False,
    },
    {
        "name": "tool_calls_static_plus_defense_memory",
        "label": "Tools + static + defense",
        "use_tools": True,
        "enable_static_memory_scan": True,
        "enable_defense_memory_retrieval": True,
        "use_defense_memory_path": True,
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="David & Goliath - unified offline analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--blue-config", type=str, default=None)
    parser.add_argument("--oracle-config", type=str, default=None)
    parser.add_argument("--tasks-path", type=str, default=None)
    parser.add_argument("--ood-tasks-path", type=str, default=None)
    parser.add_argument("--n-tasks-per-payload", type=int, default=None)
    parser.add_argument("--top-k-payloads", type=int, default=None)
    parser.add_argument("--fpr-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stage1-top-k", type=int, default=None)
    parser.add_argument("--stage1-topk-buckets", type=str, default=None)
    parser.add_argument("--blue-defense-memory-path", type=str, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def _parse_buckets(value: Any) -> list[int]:
    if value is None:
        return [10, 50, 100]
    if isinstance(value, list):
        return sorted(set(int(v) for v in value if int(v) > 0))
    buckets: list[int] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k = int(chunk)
        if k > 0:
            buckets.append(k)
    return sorted(set(buckets)) or [10, 50, 100]


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.config:
        config = _deep_merge(config, _load_yaml(args.config))

    if args.experiment_dir is not None:
        config["experiment_dir"] = args.experiment_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.blue_config is not None:
        config["blue_config"] = args.blue_config
    if args.oracle_config is not None:
        config["oracle_config"] = args.oracle_config
    if args.tasks_path is not None:
        config["tasks_path"] = args.tasks_path
    if args.ood_tasks_path is not None:
        config["ood_tasks_path"] = args.ood_tasks_path
    if args.n_tasks_per_payload is not None:
        config["n_tasks_per_payload"] = args.n_tasks_per_payload
    if args.top_k_payloads is not None:
        config["top_k_payloads"] = args.top_k_payloads
    if args.fpr_samples is not None:
        config["fpr_samples"] = args.fpr_samples
    if args.seed is not None:
        config["seed"] = args.seed
    if args.stage1_top_k is not None:
        config["stage1_top_k"] = args.stage1_top_k
    if args.stage1_topk_buckets is not None:
        config["stage1_topk_buckets"] = args.stage1_topk_buckets
    if args.blue_defense_memory_path is not None:
        config["blue_defense_memory_path"] = args.blue_defense_memory_path
    if args.force_recompute:
        config["force_recompute"] = True
    if args.skip_plots:
        config["skip_plots"] = True

    if not config.get("experiment_dir"):
        raise ValueError("experiment_dir is required. Pass --experiment-dir or set it in YAML.")

    experiment_dir = Path(config["experiment_dir"]).resolve()
    config["experiment_dir"] = str(experiment_dir)
    if not config.get("output_dir"):
        config["output_dir"] = str(experiment_dir / "analysis")
    if not config.get("blue_defense_memory_path"):
        config["blue_defense_memory_path"] = str(
            experiment_dir / "memory" / "blue_defense_memory.jsonl"
        )
    config["stage1_topk_buckets"] = _parse_buckets(config.get("stage1_topk_buckets"))
    config["stage1_top_k"] = max(int(config.get("stage1_top_k") or 20), 1)
    return config


def _setup_logging(level: str, output_dir: str) -> None:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=getattr(logging, level), format=fmt, handlers=handlers)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _round_num(path: Path) -> int:
    try:
        return int(path.name.split("_")[-1])
    except ValueError:
        return -1


def _discover_checkpoints(experiment_dir: Path) -> list[Path]:
    checkpoints_dir = experiment_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    return sorted(
        [path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.startswith("round_")],
        key=_round_num,
    )


def _load_blue_team_section(path: str | None) -> dict[str, Any]:
    if not path:
        return copy.deepcopy(cross_eval_script.DEFAULT_CONFIG["blue_team"])
    payload = _load_yaml(path)
    if "blue_team" in payload and isinstance(payload["blue_team"], dict):
        return payload["blue_team"]
    return payload


def _load_oracle_section(path: str | None) -> dict[str, Any]:
    if not path:
        return copy.deepcopy(cross_eval_script.DEFAULT_CONFIG["oracle"])
    payload = _load_yaml(path)
    if "oracle" in payload and isinstance(payload["oracle"], dict):
        return payload["oracle"]
    return payload


def _stage_paths(experiment_dir: Path) -> dict[str, Path]:
    return {
        "rollouts": experiment_dir / "rollouts" / "rollouts.jsonl",
        "episodes": experiment_dir / "memory" / "episodes.jsonl",
        "defense_memory": experiment_dir / "memory" / "blue_defense_memory.jsonl",
    }


def _build_stage1_proxy(experiment_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = _stage_paths(experiment_dir)
    rollouts_path = paths["rollouts"]
    if not rollouts_path.exists():
        return {
            "available": False,
            "reason": f"missing rollouts: {rollouts_path}",
            "summary": None,
            "round_metrics": [],
            "joined_rows": [],
            "top_rows": [],
        }

    rollout_rows, duplicate_rollouts = stage1_eval_script._load_rollouts(rollouts_path)
    round_metrics = stage1_eval_script._build_round_metrics(rollout_rows)
    stage1_summary = stage1_eval_script._summarize_rollouts(rollout_rows)
    top_rows = stage1_eval_script._top_stage1_rows(rollout_rows, int(config["stage1_top_k"]))

    alignment_summary = None
    joined_rows: list[dict[str, Any]] = []
    duplicate_episodes = 0
    missing_stage3 = 0
    episodes_path = paths["episodes"]
    if episodes_path.exists():
        episodes_by_key, duplicate_episodes = stage1_eval_script._load_stage3_episodes(
            episodes_path
        )
        joined_rows, missing_stage3 = stage1_eval_script._join_stage3(
            rollout_rows, episodes_by_key
        )
        alignment_summary = stage1_eval_script._compute_alignment(
            joined_rows, config["stage1_topk_buckets"]
        )

    return {
        "available": True,
        "duplicate_rollouts": duplicate_rollouts,
        "duplicate_stage3_rows": duplicate_episodes,
        "missing_stage3_rows": missing_stage3,
        "summary": stage1_summary,
        "round_metrics": round_metrics,
        "alignment": alignment_summary,
        "joined_rows": joined_rows,
        "top_rows": top_rows,
        "paths": {
            "rollouts": str(rollouts_path),
            "episodes": str(episodes_path) if episodes_path.exists() else None,
        },
    }


def _build_eval_config(
    config: dict[str, Any],
    *,
    blue_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    blue_cfg = copy.deepcopy(_load_blue_team_section(config.get("blue_config")))
    if blue_overrides:
        blue_cfg.update(blue_overrides)
    oracle_cfg = copy.deepcopy(_load_oracle_section(config.get("oracle_config")))
    return {
        "seed": int(config["seed"]),
        "coding_tasks_path": str(config["tasks_path"]),
        "n_tasks_per_payload": int(config["n_tasks_per_payload"]),
        "top_k_payloads": config.get("top_k_payloads"),
        "fpr_samples": int(config.get("fpr_samples") or 0),
        "blue_team": blue_cfg,
        "oracle": oracle_cfg,
    }


def _save_episode_rows(path: Path, episodes: list[Any]) -> None:
    rows = [cross_eval_script._episode_to_row(ep) for ep in episodes]
    _dump_jsonl(path, rows)


def _run_id_eval(
    checkpoint_dir: Path,
    config: dict[str, Any],
    cache_dir: Path,
    *,
    blue_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    result_path = cache_dir / "cross_eval_results.json"
    episodes_path = cache_dir / "episodes.jsonl"
    if result_path.exists() and not config["force_recompute"]:
        return _load_json(result_path)

    eval_config = _build_eval_config(config, blue_overrides=blue_overrides)
    id_tasks = load_coding_tasks(eval_config["coding_tasks_path"])
    injection_engine = InjectionEngine(coding_tasks=id_tasks, seed=eval_config["seed"])
    oracle = cross_eval_script._build_oracle(eval_config)
    blue_team = cross_eval_script._build_blue_team(eval_config)
    evaluator = CrossEvaluator(
        oracle=oracle,
        blue_team=blue_team,
        injection_engine=injection_engine,
        seed=eval_config["seed"],
    )

    payloads = load_payloads_from_checkpoint(checkpoint_dir)
    if eval_config.get("top_k_payloads"):
        payloads = payloads[: int(eval_config["top_k_payloads"])]

    logger.info("Running ID eval for %s (%d payloads)", checkpoint_dir.name, len(payloads))
    episodes = evaluator.run_episodes(
        payloads=payloads,
        tasks=None,
        n_tasks_per_payload=eval_config["n_tasks_per_payload"],
    )
    results: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "target_blue_model": eval_config["blue_team"].get("model"),
        "judge_model": eval_config["oracle"].get("judge_model"),
        "n_payloads": len(payloads),
        "n_tasks_per_payload": eval_config["n_tasks_per_payload"],
        "seed": eval_config["seed"],
        "blue_team_config": eval_config["blue_team"],
        "id_metrics": aggregate_metrics(episodes, prefix="id"),
        "id_n_episodes": len(episodes),
    }
    fpr_n = int(eval_config.get("fpr_samples") or 0)
    if fpr_n > 0:
        results["fpr_check"] = evaluator.false_positive_check(n_samples=fpr_n)
        results["id_metrics"]["id/blue_false_positive_rate"] = results["fpr_check"].get(
            "false_positive_rate", 0.0
        )
    else:
        results["fpr_check"] = None

    db_path = checkpoint_dir / "strategy_db.json"
    if db_path.exists():
        from core.strategy_db import MAPElitesDB

        db = MAPElitesDB()
        db.load(db_path)
        results["checkpoint_strategy_db"] = db.niche_stats()
    else:
        results["checkpoint_strategy_db"] = {}

    cache_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(result_path, results)
    _save_episode_rows(episodes_path, episodes)
    return results


def _run_final_ood(
    checkpoint_dir: Path,
    config: dict[str, Any],
    cache_dir: Path,
) -> Optional[dict[str, Any]]:
    logger = logging.getLogger(__name__)
    ood_tasks_path = config.get("ood_tasks_path")
    if not ood_tasks_path:
        return None

    result_path = cache_dir / "ood_results.json"
    if result_path.exists() and not config["force_recompute"]:
        return _load_json(result_path)

    eval_config = _build_eval_config(config)
    id_tasks = load_coding_tasks(eval_config["coding_tasks_path"])
    ood_tasks = load_coding_tasks(str(ood_tasks_path))
    if not ood_tasks:
        logger.warning("No OOD tasks loaded from %s", ood_tasks_path)
        return None

    injection_engine = InjectionEngine(coding_tasks=id_tasks, seed=eval_config["seed"])
    oracle = cross_eval_script._build_oracle(eval_config)
    blue_team = cross_eval_script._build_blue_team(eval_config)
    evaluator = CrossEvaluator(
        oracle=oracle,
        blue_team=blue_team,
        injection_engine=injection_engine,
        seed=eval_config["seed"],
    )
    ood_eval = OODEvaluator(cross_evaluator=evaluator, ood_tasks=ood_tasks)

    payloads = load_payloads_from_checkpoint(checkpoint_dir)
    if eval_config.get("top_k_payloads"):
        payloads = payloads[: int(eval_config["top_k_payloads"])]

    logger.info("Running final OOD eval for %s", checkpoint_dir.name)
    results = ood_eval.compare(
        payloads=payloads,
        n_tasks_per_payload=eval_config["n_tasks_per_payload"],
    )
    if int(eval_config.get("fpr_samples") or 0) > 0:
        results["fpr_check"] = evaluator.false_positive_check(
            n_samples=int(eval_config["fpr_samples"])
        )
        fpr_rate = results["fpr_check"].get("false_positive_rate", 0.0)
        results["id_metrics"]["id/blue_false_positive_rate"] = fpr_rate
        results["ood_metrics"]["ood/blue_false_positive_rate"] = fpr_rate
    else:
        results["fpr_check"] = None
    results["checkpoint_dir"] = str(checkpoint_dir)
    results["blue_team_config"] = eval_config["blue_team"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(result_path, results)
    return results


def _run_blue_ablation(
    checkpoint_dir: Path,
    config: dict[str, Any],
    cache_root: Path,
) -> list[dict[str, Any]]:
    logger = logging.getLogger(__name__)
    defense_memory_path = Path(str(config["blue_defense_memory_path"]))
    rows: list[dict[str, Any]] = []

    for variant in BLUE_VARIANTS:
        overrides = {
            "use_tools": variant["use_tools"],
            "enable_static_memory_scan": variant["enable_static_memory_scan"],
            "enable_defense_memory_retrieval": variant["enable_defense_memory_retrieval"],
        }
        if variant["use_defense_memory_path"]:
            overrides["defense_memory_path"] = (
                str(defense_memory_path) if defense_memory_path.exists() else None
            )
        else:
            overrides["defense_memory_path"] = None

        cache_dir = cache_root / variant["name"]
        logger.info("Running blue ablation variant: %s", variant["name"])
        results = _run_id_eval(
            checkpoint_dir=checkpoint_dir,
            config=config,
            cache_dir=cache_dir,
            blue_overrides=overrides,
        )
        id_metrics = results["id_metrics"]
        fpr_rate = 0.0
        if results.get("fpr_check"):
            fpr_rate = float(results["fpr_check"].get("false_positive_rate", 0.0))
        rows.append(
            {
                "variant": variant["name"],
                "label": variant["label"],
                "use_tools": variant["use_tools"],
                "enable_static_memory_scan": variant["enable_static_memory_scan"],
                "enable_defense_memory_retrieval": variant["enable_defense_memory_retrieval"],
                "defense_memory_available": defense_memory_path.exists(),
                "id_asr": float(id_metrics.get("id/asr", 0.0)),
                "id_avg_reward": float(id_metrics.get("id/avg_reward", 0.0)),
                "judge_a_trigger_rate": float(id_metrics.get("id/judge_a_trigger_rate", 0.0)),
                "judge_b_trigger_rate": float(id_metrics.get("id/judge_b_trigger_rate", 0.0)),
                "blue_detection_rate": float(id_metrics.get("id/blue_detection_rate", 0.0)),
                "blue_false_positive_rate": fpr_rate,
                "blue_verification_pass_rate": float(
                    id_metrics.get("id/blue_verification_pass_rate", 0.0)
                ),
                "blue_tests_pass_rate": float(
                    id_metrics.get("id/blue_tests_pass_rate", 0.0)
                ),
                "blue_static_clean_rate": float(
                    id_metrics.get("id/blue_static_clean_rate", 0.0)
                ),
                "blue_tool_call_rate": float(id_metrics.get("id/blue_tool_call_rate", 0.0)),
                "blue_avg_num_tool_calls": float(
                    id_metrics.get("id/blue_avg_num_tool_calls", 0.0)
                ),
                "blue_memory_static_hit_rate": float(
                    id_metrics.get("id/blue_memory_static_hit_rate", 0.0)
                ),
                "blue_memory_dynamic_hit_rate": float(
                    id_metrics.get("id/blue_memory_dynamic_hit_rate", 0.0)
                ),
                "blue_high_risk_memory_hit_rate": float(
                    id_metrics.get("id/blue_high_risk_memory_hit_rate", 0.0)
                ),
                "blue_defense_context_applied_rate": float(
                    id_metrics.get("id/blue_defense_context_applied_rate", 0.0)
                ),
                "blue_avg_retrieved_memory_count": float(
                    id_metrics.get("id/blue_avg_retrieved_memory_count", 0.0)
                ),
            }
        )

    return rows


def _build_checkpoint_sweep_rows(results_by_checkpoint: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_asr = None
    baseline_reward = None
    for checkpoint_dir, results in results_by_checkpoint:
        metrics = results.get("id_metrics", {})
        row = {
            "round": _round_num(checkpoint_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "n_payloads": results.get("n_payloads", 0),
            "id_n_episodes": results.get("id_n_episodes", 0),
            "id_asr": float(metrics.get("id/asr", 0.0)),
            "id_avg_reward": float(metrics.get("id/avg_reward", 0.0)),
            "id_judge_a_trigger_rate": float(metrics.get("id/judge_a_trigger_rate", 0.0)),
            "id_judge_b_trigger_rate": float(metrics.get("id/judge_b_trigger_rate", 0.0)),
            "id_avg_payload_quality": float(metrics.get("id/avg_payload_quality", 0.0)),
            "id_avg_stealth_score": float(metrics.get("id/avg_stealth_score", 0.0)),
            "id_blue_detection_rate": float(metrics.get("id/blue_detection_rate", 0.0)),
            "id_blue_false_positive_rate": float(
                metrics.get("id/blue_false_positive_rate", 0.0)
            ),
            "strategy_db_coverage": float(
                (results.get("checkpoint_strategy_db") or {}).get("strategy_db/coverage", 0.0)
            ),
        }
        if baseline_asr is None:
            baseline_asr = row["id_asr"]
            baseline_reward = row["id_avg_reward"]
        row["delta_asr_vs_first"] = row["id_asr"] - float(baseline_asr or 0.0)
        row["delta_reward_vs_first"] = row["id_avg_reward"] - float(baseline_reward or 0.0)
        rows.append(row)
    return rows


def _summarize_checkpoint_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_checkpoints": 0}
    best_asr_row = max(rows, key=lambda row: row["id_asr"])
    best_reward_row = max(rows, key=lambda row: row["id_avg_reward"])
    return {
        "n_checkpoints": len(rows),
        "first_round": rows[0]["round"],
        "last_round": rows[-1]["round"],
        "best_checkpoint_by_asr": {
            "round": best_asr_row["round"],
            "checkpoint_dir": best_asr_row["checkpoint_dir"],
            "id_asr": best_asr_row["id_asr"],
        },
        "best_checkpoint_by_reward": {
            "round": best_reward_row["round"],
            "checkpoint_dir": best_reward_row["checkpoint_dir"],
            "id_avg_reward": best_reward_row["id_avg_reward"],
        },
    }


def _summarize_ood(results: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not results:
        return None
    id_metrics = results["id_metrics"]
    ood_metrics = results["ood_metrics"]
    return {
        "id_asr": float(id_metrics.get("id/asr", 0.0)),
        "ood_asr": float(ood_metrics.get("ood/asr", 0.0)),
        "id_avg_reward": float(id_metrics.get("id/avg_reward", 0.0)),
        "ood_avg_reward": float(ood_metrics.get("ood/avg_reward", 0.0)),
        "transfer_ratio": float(results.get("transfer_ratio", 0.0)),
        "asr_gap": float(id_metrics.get("id/asr", 0.0)) - float(ood_metrics.get("ood/asr", 0.0)),
        "reward_gap": float(id_metrics.get("id/avg_reward", 0.0))
        - float(ood_metrics.get("ood/avg_reward", 0.0)),
        "blue_false_positive_rate": float(
            (results.get("fpr_check") or {}).get("false_positive_rate", 0.0)
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def _fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _fmt_num(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _build_markdown_report(
    summary: dict[str, Any],
    checkpoint_rows: list[dict[str, Any]],
    blue_rows: list[dict[str, Any]],
) -> str:
    stage1 = summary.get("stage1_proxy") or {}
    stage1_summary = stage1.get("summary") or {}
    alignment = stage1.get("alignment") or {}
    sweep = summary.get("checkpoint_sweep") or {}
    ood = summary.get("ood_generalization")

    lines = [
        "# Unified Red/Blue Evaluation Report",
        "",
        f"- Generated: {summary['timestamp']}",
        f"- Experiment: `{summary['experiment_dir']}`",
        "",
        "## Stage 1 Proxy",
    ]
    if not stage1.get("available"):
        lines += ["Stage-1 proxy analysis unavailable.", ""]
    else:
        corr = (alignment.get("correlations") or {})
        lines += [
            f"- Rollouts: {stage1_summary.get('n_rollouts', 0)}",
            f"- Reward mean/std: {_fmt_num((stage1_summary.get('reward') or {}).get('mean'))} / {_fmt_num((stage1_summary.get('reward') or {}).get('std'))}",
            f"- Judge C quality mean: {_fmt_num((stage1_summary.get('judge_c_quality') or {}).get('mean'))}",
            f"- Niche coverage: {_fmt_rate(stage1_summary.get('niche_coverage'))}",
            f"- Pearson(stage1 reward, stage3 reward): {_fmt_num(corr.get('pearson_stage1_reward_vs_stage3_reward'))}",
            f"- Spearman(stage1 reward, stage3 reward): {_fmt_num(corr.get('spearman_stage1_reward_vs_stage3_reward'))}",
            "",
        ]

    lines += ["## Red Checkpoint Sweep"]
    if checkpoint_rows:
        lines += [
            f"- Checkpoints evaluated: {sweep.get('n_checkpoints', 0)}",
            f"- Best ASR checkpoint: round {((sweep.get('best_checkpoint_by_asr') or {}).get('round', 'n/a'))}",
            f"- Best reward checkpoint: round {((sweep.get('best_checkpoint_by_reward') or {}).get('round', 'n/a'))}",
            "",
            _markdown_table(
                checkpoint_rows,
                [
                    "round",
                    "id_asr",
                    "id_avg_reward",
                    "id_judge_a_trigger_rate",
                    "id_judge_b_trigger_rate",
                    "id_blue_detection_rate",
                    "id_blue_false_positive_rate",
                ],
            ),
            "",
        ]
    else:
        lines += ["No checkpoint results available.", ""]

    lines += ["## OOD Generalization"]
    if ood:
        lines += [
            f"- ID ASR: {_fmt_rate(ood.get('id_asr'))}",
            f"- OOD ASR: {_fmt_rate(ood.get('ood_asr'))}",
            f"- Transfer ratio: {_fmt_num(ood.get('transfer_ratio'))}",
            f"- ASR gap: {_fmt_rate(ood.get('asr_gap'))}",
            f"- Reward gap: {_fmt_num(ood.get('reward_gap'))}",
            "",
        ]
    else:
        lines += ["OOD analysis skipped or unavailable.", ""]

    lines += ["## Blue Ablation"]
    if blue_rows:
        lines += [
            _markdown_table(
                blue_rows,
                [
                    "variant",
                    "id_asr",
                    "judge_b_trigger_rate",
                    "judge_a_trigger_rate",
                    "blue_detection_rate",
                    "blue_false_positive_rate",
                    "blue_avg_num_tool_calls",
                ],
            ),
            "",
        ]
    else:
        lines += ["Blue ablation unavailable.", ""]

    lines += ["## Artifacts", ""]
    for label, rel_path in sorted((summary.get("artifacts") or {}).items()):
        lines.append(f"- `{label}`: `{rel_path}`")
    lines.append("")
    return "\n".join(lines)


def _safe_plot(path: Path, func) -> bool:
    if not _MPL_AVAILABLE:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    func(path)
    return True


def _plot_stage1_proxy_trends(path: Path, round_metrics: list[dict[str, Any]]) -> None:
    if not round_metrics:
        return
    rounds = [row["round"] for row in round_metrics]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(rounds, [row["reward_mean"] for row in round_metrics], marker="o")
    axes[0].set_ylabel("Stage1 reward")
    axes[0].set_title("Stage-1 proxy trends")

    axes[1].plot(
        rounds,
        [row["judge_c_quality_mean"] for row in round_metrics],
        marker="o",
        label="Judge C quality",
    )
    axes[1].plot(
        rounds,
        [row["judge_c_stealth_mean"] for row in round_metrics],
        marker="s",
        label="Judge C stealth",
    )
    axes[1].legend()
    axes[1].set_ylabel("Judge C")

    axes[2].plot(
        rounds,
        [row["cumulative_niche_coverage"] for row in round_metrics],
        marker="o",
        color="black",
    )
    axes[2].set_ylabel("Coverage")
    axes[2].set_xlabel("Round")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_stage1_proxy_alignment(path: Path, joined_rows: list[dict[str, Any]]) -> None:
    if not joined_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        [row["stage1_reward"] for row in joined_rows],
        [row["stage3_total_reward"] for row in joined_rows],
        alpha=0.5,
    )
    axes[0].set_xlabel("Stage-1 reward")
    axes[0].set_ylabel("Stage-3 reward")
    axes[0].set_title("Stage1 reward vs Stage3 reward")

    axes[1].scatter(
        [row["judge_c_quality"] for row in joined_rows],
        [row["stage3_total_reward"] for row in joined_rows],
        alpha=0.5,
        color="tab:green",
    )
    axes[1].set_xlabel("Judge C quality")
    axes[1].set_ylabel("Stage-3 reward")
    axes[1].set_title("Judge C quality vs Stage3 reward")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_red_checkpoint_progress(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rounds = [row["round"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(rounds, [row["id_asr"] for row in rows], marker="o", label="ASR")
    axes[0].plot(
        rounds,
        [row["id_judge_a_trigger_rate"] for row in rows],
        marker="s",
        label="Judge A",
    )
    axes[0].plot(
        rounds,
        [row["id_judge_b_trigger_rate"] for row in rows],
        marker="^",
        label="Judge B",
    )
    axes[0].legend()
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Red checkpoint progression")

    axes[1].plot(rounds, [row["id_avg_reward"] for row in rows], marker="o", label="Reward")
    axes[1].plot(
        rounds,
        [row["id_blue_detection_rate"] for row in rows],
        marker="s",
        label="Blue detection",
    )
    axes[1].legend()
    axes[1].set_ylabel("Value")
    axes[1].set_xlabel("Round")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ood_generalization(path: Path, ood_summary: Optional[dict[str, Any]]) -> None:
    if not ood_summary:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(["ID ASR", "OOD ASR"], [ood_summary["id_asr"], ood_summary["ood_asr"]])
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_title("Attack success")

    axes[1].bar(
        ["Transfer", "ASR gap", "Reward gap"],
        [
            ood_summary["transfer_ratio"],
            ood_summary["asr_gap"],
            ood_summary["reward_gap"],
        ],
    )
    axes[1].set_title("Generalization summary")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_blue_ablation_attack_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [row["label"] for row in rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.14
    series = [
        ("ASR", "id_asr"),
        ("Judge A", "judge_a_trigger_rate"),
        ("Judge B", "judge_b_trigger_rate"),
        ("Detection", "blue_detection_rate"),
        ("FPR", "blue_false_positive_rate"),
    ]
    for idx, (name, key) in enumerate(series):
        offset = [value + width * (idx - 2) for value in x]
        ax.bar(offset, [row[key] for row in rows], width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.set_title("Blue ablation - attack metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_blue_ablation_runtime_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [row["label"] for row in rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.12
    series = [
        ("Verify", "blue_verification_pass_rate"),
        ("Tests", "blue_tests_pass_rate"),
        ("Tool rate", "blue_tool_call_rate"),
        ("Avg tool calls", "blue_avg_num_tool_calls"),
        ("Static hit", "blue_memory_static_hit_rate"),
        ("Dynamic hit", "blue_memory_dynamic_hit_rate"),
    ]
    for idx, (name, key) in enumerate(series):
        offset = [value + width * (idx - 2.5) for value in x]
        ax.bar(offset, [row[key] for row in rows], width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.set_title("Blue ablation - runtime and memory metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_blue_detection_vs_fpr(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for row in rows:
        ax.scatter(row["blue_false_positive_rate"], row["blue_detection_rate"], s=70)
        ax.annotate(row["label"], (row["blue_false_positive_rate"], row["blue_detection_rate"]))
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("Detection rate")
    ax.set_title("Detection vs FPR")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_analysis(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    experiment_dir = Path(config["experiment_dir"])
    output_dir = Path(config["output_dir"])
    plot_dir = output_dir / "plots"
    cache_dir = output_dir / "cache"
    checkpoints = _discover_checkpoints(experiment_dir)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {experiment_dir / 'checkpoints'}")

    logger.info("=" * 60)
    logger.info("Unified Offline Analysis")
    logger.info("=" * 60)
    logger.info("Experiment dir : %s", experiment_dir)
    logger.info("Output dir     : %s", output_dir)
    logger.info("Checkpoints    : %d", len(checkpoints))
    logger.info("Tasks path     : %s", config["tasks_path"])
    logger.info("OOD tasks      : %s", config.get("ood_tasks_path") or "(none)")
    logger.info("Blue config    : %s", config["blue_config"])
    logger.info("Oracle config  : %s", config["oracle_config"])
    logger.info("Force recompute: %s", config["force_recompute"])
    logger.info("Skip plots     : %s", config["skip_plots"])

    stage1_proxy = _build_stage1_proxy(experiment_dir, config)

    checkpoint_results: list[tuple[Path, dict[str, Any]]] = []
    for checkpoint_dir in checkpoints:
        checkpoint_results.append(
            (
                checkpoint_dir,
                _run_id_eval(
                    checkpoint_dir=checkpoint_dir,
                    config=config,
                    cache_dir=cache_dir / "checkpoint_sweep" / checkpoint_dir.name,
                ),
            )
        )

    checkpoint_rows = _build_checkpoint_sweep_rows(checkpoint_results)
    checkpoint_summary = _summarize_checkpoint_sweep(checkpoint_rows)

    final_checkpoint = checkpoints[-1]
    ood_results = _run_final_ood(
        checkpoint_dir=final_checkpoint,
        config=config,
        cache_dir=cache_dir / "final_ood",
    )
    ood_summary = _summarize_ood(ood_results)

    blue_rows = _run_blue_ablation(
        checkpoint_dir=final_checkpoint,
        config=config,
        cache_root=cache_dir / "blue_ablation",
    )

    stage1_public = {
        "available": stage1_proxy.get("available", False),
        "duplicate_rollouts": stage1_proxy.get("duplicate_rollouts", 0),
        "duplicate_stage3_rows": stage1_proxy.get("duplicate_stage3_rows", 0),
        "missing_stage3_rows": stage1_proxy.get("missing_stage3_rows", 0),
        "summary": stage1_proxy.get("summary"),
        "alignment": stage1_proxy.get("alignment"),
        "top_rows": stage1_proxy.get("top_rows"),
        "paths": stage1_proxy.get("paths"),
    }

    summary: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "stage1_proxy": stage1_public,
        "checkpoint_sweep": checkpoint_summary,
        "ood_generalization": ood_summary,
        "blue_ablation": {
            "n_variants": len(blue_rows),
            "rows": blue_rows,
        },
        "artifacts": {},
    }

    artifact_paths = {
        "summary_json": output_dir / "summary.json",
        "report_md": output_dir / "report.md",
        "checkpoint_sweep_csv": output_dir / "checkpoint_sweep.csv",
        "blue_ablation_csv": output_dir / "blue_ablation.csv",
        "stage1_proxy_summary_json": output_dir / "stage1_proxy_summary.json",
        "stage1_proxy_trends_png": plot_dir / "stage1_proxy_trends.png",
        "stage1_proxy_alignment_png": plot_dir / "stage1_proxy_alignment.png",
        "red_checkpoint_progress_png": plot_dir / "red_checkpoint_progress.png",
        "ood_generalization_png": plot_dir / "ood_generalization.png",
        "blue_ablation_attack_metrics_png": plot_dir / "blue_ablation_attack_metrics.png",
        "blue_ablation_runtime_metrics_png": plot_dir / "blue_ablation_runtime_metrics.png",
        "blue_detection_vs_fpr_png": plot_dir / "blue_detection_vs_fpr.png",
    }

    if dry_run:
        for label, path in artifact_paths.items():
            summary["artifacts"][label] = str(path)
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(
        artifact_paths["stage1_proxy_summary_json"],
        {
            "timestamp": summary["timestamp"],
            "stage1_proxy": stage1_public,
            "round_metrics": stage1_proxy.get("round_metrics", []),
        },
    )
    _dump_csv(artifact_paths["checkpoint_sweep_csv"], checkpoint_rows)
    _dump_csv(artifact_paths["blue_ablation_csv"], blue_rows)

    plots_created: dict[str, bool] = {}
    if not config["skip_plots"] and _MPL_AVAILABLE:
        plots_created["stage1_proxy_trends_png"] = _safe_plot(
            artifact_paths["stage1_proxy_trends_png"],
            lambda path: _plot_stage1_proxy_trends(path, stage1_proxy.get("round_metrics", [])),
        )
        plots_created["stage1_proxy_alignment_png"] = _safe_plot(
            artifact_paths["stage1_proxy_alignment_png"],
            lambda path: _plot_stage1_proxy_alignment(path, stage1_proxy.get("joined_rows", [])),
        )
        plots_created["red_checkpoint_progress_png"] = _safe_plot(
            artifact_paths["red_checkpoint_progress_png"],
            lambda path: _plot_red_checkpoint_progress(path, checkpoint_rows),
        )
        plots_created["ood_generalization_png"] = _safe_plot(
            artifact_paths["ood_generalization_png"],
            lambda path: _plot_ood_generalization(path, ood_summary),
        )
        plots_created["blue_ablation_attack_metrics_png"] = _safe_plot(
            artifact_paths["blue_ablation_attack_metrics_png"],
            lambda path: _plot_blue_ablation_attack_metrics(path, blue_rows),
        )
        plots_created["blue_ablation_runtime_metrics_png"] = _safe_plot(
            artifact_paths["blue_ablation_runtime_metrics_png"],
            lambda path: _plot_blue_ablation_runtime_metrics(path, blue_rows),
        )
        plots_created["blue_detection_vs_fpr_png"] = _safe_plot(
            artifact_paths["blue_detection_vs_fpr_png"],
            lambda path: _plot_blue_detection_vs_fpr(path, blue_rows),
        )
    else:
        logger.info("Skipping plot generation (skip_plots=%s mpl=%s)", config["skip_plots"], _MPL_AVAILABLE)

    for label, path in artifact_paths.items():
        if label.endswith("_png") and not path.exists():
            continue
        summary["artifacts"][label] = str(path.relative_to(output_dir))
    if plots_created:
        summary["artifacts"]["plots_created"] = plots_created

    report = _build_markdown_report(summary, checkpoint_rows, blue_rows)
    report_path = artifact_paths["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    _dump_json(artifact_paths["summary_json"], summary)
    return summary


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    _setup_logging(args.log_level, config["output_dir"])
    summary = run_analysis(config, dry_run=args.dry_run)
    logging.getLogger(__name__).info(
        "Analysis complete. Summary: %s",
        Path(config["output_dir"]) / "summary.json",
    )
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
