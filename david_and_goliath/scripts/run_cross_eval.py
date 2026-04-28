"""scripts/run_cross_eval.py — Cross-evaluation entry point.

Load a trained Red Team checkpoint and evaluate its payloads against a
(possibly different) Blue Team, reporting attack success, reward, detection
rate and — if OOD tasks are provided — in-distribution vs out-of-distribution
transfer metrics.

Usage
-----
  # Basic: evaluate payloads in a checkpoint against the default Blue Team
  python -m david_and_goliath.scripts.run_cross_eval \
      --checkpoint-dir outputs/my_exp/checkpoints/round_020

  # Transfer: swap in a stronger Blue Team model
  python -m david_and_goliath.scripts.run_cross_eval \
      --checkpoint-dir outputs/my_exp/checkpoints/round_020 \
      --target-blue-model gpt-4o

  # OOD generalisation: compare ID vs held-out task set
  python -m david_and_goliath.scripts.run_cross_eval \
      --checkpoint-dir outputs/my_exp/checkpoints/round_020 \
      --ood-tasks-path data/coding_tasks/ood.jsonl

  # Override oracle / eval knobs via YAML
  python -m david_and_goliath.scripts.run_cross_eval \
      --checkpoint-dir outputs/my_exp/checkpoints/round_020 \
      --config configs/cross_eval.yaml

Output
------
  <output_dir>/cross_eval_results.json
  <output_dir>/episodes.jsonl           (if --save-episodes)
  <output_dir>/logs/run_<timestamp>.log
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ---- Import paths for both `python -m ...` and direct script execution ----
_HERE = Path(__file__).resolve()
_PACKAGE_ROOT = _HERE.parent.parent          # david_and_goliath/
_PROJECT_ROOT = _PACKAGE_ROOT.parent         # repo root
for _path in (_PROJECT_ROOT, _PACKAGE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from core.injection_engine import InjectionEngine, load_coding_tasks
from core.strategy_db import MAPElitesDB
from evaluation.cross_evaluator import (
    CrossEvaluator,
    load_payloads_from_checkpoint,
)
from evaluation.metrics import (
    aggregate_metrics,
    compare_metrics,
    print_metrics,
)
from evaluation.ood_evaluator import OODEvaluator


# ===========================================================================
# Default config (mirrors the oracle/blue_team subset of training config)
# ===========================================================================

DEFAULT_CONFIG: dict[str, Any] = {
    # ---- Paths ----
    "coding_tasks_path": "david_and_goliath/data/coding_tasks/tasks.jsonl",
    "ood_tasks_path": None,

    # ---- Eval knobs ----
    "seed": 42,
    "n_tasks_per_payload": 3,
    "top_k_payloads": None,           # None = evaluate all payloads in the checkpoint
    "fpr_samples": 20,                # 0 = skip FPR check
    "save_episodes": True,

    # ---- Hybrid Oracle (must match training weights so rewards are comparable) ----
    "oracle": {
        "judge_model": "gpt-4o-mini",
        "api_key": None,
        "bandit_enabled": True,
        "semgrep_enabled": False,
        "semgrep_rules": "p/security-audit",
        "judge_temperature": 0.1,
        "w_vulnerability": 0.30,
        "w_manipulation": 0.30,
        "w_quality": 0.20,
        "w_diversity": 0.10,
        "w_stealth": 0.10,
        "failure_quality_scale": 0.15,
    },

    # ---- Target Blue Team (can differ from training for transfer-eval) ----
    "blue_team": {
        "model": "gpt-4o-mini",
        "api_key": None,
        "base_url": None,
        "temperature": 0.2,
        "max_turns": 6,
        "max_reflexion": 2,
        "use_tools": True,
        "enable_static_memory_scan": True,
        "enable_defense_memory_retrieval": True,
        "defense_memory_path": None,
        "defense_retrieval_top_k": 3,
    },
}


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="David & Goliath — Cross-Evaluation of Trained Red Team",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- Required / paths ----
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True,
        help="Path to a checkpoint round directory containing strategy_db.json.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional YAML config (merged on top of DEFAULT_CONFIG).",
    )
    parser.add_argument(
        "--tasks-path", type=str, default=None,
        help="Override in-distribution coding tasks JSONL.",
    )
    parser.add_argument(
        "--ood-tasks-path", type=str, default=None,
        help="Path to OOD coding tasks JSONL (enables ID vs OOD comparison).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save results. Defaults to <checkpoint-dir>/cross_eval/.",
    )

    # ---- Target Blue Team overrides ----
    parser.add_argument(
        "--target-blue-model", type=str, default=None,
        help="Override target Blue Team model (e.g. gpt-4o, gpt-4o-mini).",
    )
    parser.add_argument(
        "--target-blue-api-key", type=str, default=None,
        help="Override Blue Team API key (falls back to env var if omitted).",
    )
    parser.add_argument(
        "--target-blue-base-url", type=str, default=None,
        help="Override Blue Team base URL (for local / proxy endpoints).",
    )

    # ---- Eval knobs ----
    parser.add_argument(
        "--n-tasks-per-payload", type=int, default=None,
        help="How many coding tasks each payload is tested against.",
    )
    parser.add_argument(
        "--top-k-payloads", type=int, default=None,
        help="Only evaluate the top-K payloads by training reward (default: all).",
    )
    parser.add_argument(
        "--fpr-samples", type=int, default=None,
        help="Number of clean tasks for Blue Team false-positive check (0 = skip).",
    )
    parser.add_argument(
        "--no-save-episodes", action="store_true",
        help="Do not dump per-episode details to episodes.jsonl.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for task sampling.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ===========================================================================
# Config helpers (shared shape with run_coevolution.py)
# ===========================================================================

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


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)

    yaml_cfg: dict[str, Any] = {}
    if args.config:
        yaml_cfg = _load_yaml(args.config)
        config = _deep_merge(config, yaml_cfg)

    # ---- Paths ----
    config["checkpoint_dir"] = args.checkpoint_dir
    if args.tasks_path:
        config["coding_tasks_path"] = args.tasks_path
    if args.ood_tasks_path:
        config["ood_tasks_path"] = args.ood_tasks_path

    # Output dir defaults to <checkpoint>/cross_eval unless set by YAML/CLI
    if args.output_dir:
        config["output_dir"] = args.output_dir
    elif "output_dir" not in yaml_cfg:
        config["output_dir"] = str(Path(args.checkpoint_dir) / "cross_eval")

    # ---- Target Blue Team overrides ----
    if args.target_blue_model:
        config["blue_team"]["model"] = args.target_blue_model
    if args.target_blue_api_key:
        config["blue_team"]["api_key"] = args.target_blue_api_key
    if args.target_blue_base_url:
        config["blue_team"]["base_url"] = args.target_blue_base_url

    # ---- Eval knobs ----
    if args.n_tasks_per_payload is not None:
        config["n_tasks_per_payload"] = args.n_tasks_per_payload
    if args.top_k_payloads is not None:
        config["top_k_payloads"] = args.top_k_payloads
    if args.fpr_samples is not None:
        config["fpr_samples"] = args.fpr_samples
    if args.no_save_episodes:
        config["save_episodes"] = False
    if args.seed is not None:
        config["seed"] = args.seed

    return config


# ===========================================================================
# Logging
# ===========================================================================

def _setup_logging(level: str, output_dir: str) -> None:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=getattr(logging, level), format=fmt, handlers=handlers)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ===========================================================================
# Component builders (mirror co_evolution_controller._build_oracle / _build_blue_team)
# ===========================================================================

def _build_oracle(config: dict):
    from hybrid_oracle.judge_a import JudgeA
    from hybrid_oracle.judge_b import JudgeB
    from hybrid_oracle.judge_c import JudgeC
    from hybrid_oracle.oracle import HybridOracle

    oc = config["oracle"]
    judge_a = JudgeA(
        bandit_enabled=oc.get("bandit_enabled", True),
        semgrep_enabled=oc.get("semgrep_enabled", False),
        semgrep_rules=oc.get("semgrep_rules", "p/security-audit"),
    )
    judge_b = JudgeB(
        model=oc.get("judge_model", "gpt-4o-mini"),
        api_key=oc.get("api_key"),
        temperature=oc.get("judge_temperature", 0.1),
    )
    judge_c = JudgeC(
        model=oc.get("judge_model", "gpt-4o-mini"),
        api_key=oc.get("api_key"),
        temperature=oc.get("judge_temperature", 0.2),
    )
    return HybridOracle(
        judge_a=judge_a,
        judge_b=judge_b,
        judge_c=judge_c,
        strategy_db=None,              # eval does not update the DB
        w_vulnerability=oc.get("w_vulnerability", 0.30),
        w_manipulation=oc.get("w_manipulation", 0.30),
        w_quality=oc.get("w_quality", 0.20),
        w_diversity=oc.get("w_diversity", 0.10),
        w_stealth=oc.get("w_stealth", 0.10),
        failure_quality_scale=oc.get("failure_quality_scale", 0.15),
    )


def _build_blue_team(config: dict):
    from blue_team.coding_agent import CodingAgent

    bc = config["blue_team"]
    return CodingAgent(
        model=bc.get("model", "gpt-4o-mini"),
        api_key=bc.get("api_key"),
        base_url=bc.get("base_url"),
        temperature=bc.get("temperature", 0.2),
        max_turns=bc.get("max_turns", 6),
        max_reflexion=bc.get("max_reflexion", 2),
        use_tools=bc.get("use_tools", True),
        enable_static_memory_scan=bc.get("enable_static_memory_scan", True),
        enable_defense_memory_retrieval=bc.get(
            "enable_defense_memory_retrieval", True
        ),
        defense_memory_path=bc.get("defense_memory_path"),
        defense_retrieval_top_k=bc.get("defense_retrieval_top_k", 3),
    )


# ===========================================================================
# Persistence
# ===========================================================================

def _episode_to_row(ep) -> dict[str, Any]:
    """Flatten an EpisodeResult into a compact JSON-serialisable dict."""
    row: dict[str, Any] = {
        "payload_id": ep.payload_id,
        "round": ep.round,
        "coding_task_id": ep.coding_task_id,
        "injection_position": ep.injection_position,
        "attack_success": ep.attack_success,
        "total_reward": ep.total_reward,
        "vulnerability_reward": ep.vulnerability_reward,
        "manipulation_reward": ep.manipulation_reward,
        "quality_reward": ep.quality_reward,
        "diversity_bonus": ep.diversity_bonus,
    }
    if ep.judge_a:
        row["judge_a_vulnerability_score"] = ep.judge_a.vulnerability_score
        row["judge_a_has_vulnerability"] = ep.judge_a.has_vulnerability
    if ep.judge_b:
        row["judge_b_manipulation_score"] = ep.judge_b.manipulation_score
        row["judge_b_is_manipulated"] = ep.judge_b.is_manipulated
    if ep.judge_c:
        row["judge_c_quality"] = ep.judge_c.payload_quality_score
        row["judge_c_stealth"] = ep.judge_c.stealth_score
        row["judge_c_creativity"] = ep.judge_c.creativity_score
    if ep.blue_response:
        row["blue_detected_suspicious"] = ep.blue_response.detected_suspicious
        row["blue_num_tool_calls"] = ep.blue_response.num_tool_calls
        row["blue_tools_used"] = ep.blue_response.tools_used
        row["blue_latency_ms"] = ep.blue_response.latency_ms
        row["blue_verification_passed"] = bool(
            (ep.blue_response.verification or {}).get("passed", False)
        )
        row["blue_tests_passed"] = bool(
            (ep.blue_response.verification or {}).get("tests_passed", False)
        )
        row["blue_static_clean"] = bool(
            (ep.blue_response.verification or {}).get("static_clean", False)
        )
        row["blue_memory_scan"] = ep.blue_response.memory_scan
        row["blue_memory_static_match_count"] = int(
            (ep.blue_response.memory_scan or {}).get("static_match_count", 0) or 0
        )
        row["blue_memory_dynamic_match_count"] = int(
            (ep.blue_response.memory_scan or {}).get("dynamic_match_count", 0) or 0
        )
        row["blue_memory_dynamic_high_risk_count"] = int(
            (ep.blue_response.memory_scan or {}).get("dynamic_high_risk_count", 0) or 0
        )
        row["blue_defense_context_applied"] = ep.blue_response.defense_context_applied
        row["blue_retrieved_memory_count"] = ep.blue_response.retrieved_memory_count
    return row


def _dump_episodes(episodes: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(_episode_to_row(ep), ensure_ascii=False) + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = _parse_args()
    config = _build_config(args)

    _setup_logging(args.log_level, config["output_dir"])
    logger = logging.getLogger(__name__)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Config summary ----
    logger.info("=" * 60)
    logger.info("David & Goliath — Cross-Evaluation")
    logger.info("=" * 60)
    logger.info("Checkpoint   : %s", config["checkpoint_dir"])
    logger.info("Output       : %s", config["output_dir"])
    logger.info("Tasks (ID)   : %s", config["coding_tasks_path"])
    logger.info("Tasks (OOD)  : %s", config.get("ood_tasks_path") or "(none)")
    logger.info("Blue target  : %s", config["blue_team"]["model"])
    logger.info("Judge model  : %s", config["oracle"]["judge_model"])
    logger.info("Seed         : %d", config["seed"])
    logger.info(
        "Eval         : %d tasks/payload, top_k=%s, fpr_samples=%d",
        config["n_tasks_per_payload"],
        config["top_k_payloads"],
        config["fpr_samples"],
    )

    # ---- Build components ----
    logger.info("Loading coding tasks...")
    id_tasks = load_coding_tasks(config["coding_tasks_path"])
    if not id_tasks:
        logger.error("No ID coding tasks loaded. Aborting.")
        sys.exit(1)

    ood_tasks = None
    if config.get("ood_tasks_path"):
        ood_tasks = load_coding_tasks(config["ood_tasks_path"])
        if not ood_tasks:
            logger.warning(
                "ood_tasks_path set but no tasks loaded — OOD comparison disabled."
            )
            ood_tasks = None

    injection_engine = InjectionEngine(coding_tasks=id_tasks, seed=config["seed"])
    oracle = _build_oracle(config)
    blue_team = _build_blue_team(config)

    evaluator = CrossEvaluator(
        oracle=oracle,
        blue_team=blue_team,
        injection_engine=injection_engine,
        seed=config["seed"],
    )

    # ---- Load payloads from checkpoint ----
    payloads = load_payloads_from_checkpoint(config["checkpoint_dir"])
    if not payloads:
        logger.error("Checkpoint contains no payloads. Aborting.")
        sys.exit(1)

    top_k = config.get("top_k_payloads")
    if top_k is not None and top_k > 0:
        payloads = payloads[:top_k]
        logger.info("Using top %d payloads by training reward.", len(payloads))

    # ---- Run eval ----
    t0 = time.time()
    results: dict[str, Any] = {
        "checkpoint_dir": config["checkpoint_dir"],
        "target_blue_model": config["blue_team"]["model"],
        "judge_model": config["oracle"]["judge_model"],
        "n_payloads": len(payloads),
        "n_tasks_per_payload": config["n_tasks_per_payload"],
        "seed": config["seed"],
    }

    if ood_tasks is not None:
        logger.info("Running OOD comparison (%d OOD tasks)...", len(ood_tasks))
        ood_eval = OODEvaluator(cross_evaluator=evaluator, ood_tasks=ood_tasks)
        ood_out = ood_eval.compare(
            payloads=payloads,
            n_tasks_per_payload=config["n_tasks_per_payload"],
        )
        # OODEvaluator.compare internally ran BOTH splits — no extra ID pass.
        results.update({
            "id_metrics": ood_out["id_metrics"],
            "ood_metrics": ood_out["ood_metrics"],
            "delta_id_vs_ood": ood_out["delta"],
            "transfer_ratio": ood_out["transfer_ratio"],
            "id_n_episodes": ood_out["id_episodes"],
            "ood_n_episodes": ood_out["ood_episodes"],
        })

        # For episode dump: re-run ID once so we have episodes to save.
        # (OODEvaluator doesn't return raw episodes to avoid duplicate storage.)
        # Cheaper alternative: skip dump when --ood is on.
        if config.get("save_episodes"):
            logger.info("Re-running ID split once for episode dump...")
            episodes = evaluator.run_episodes(
                payloads=payloads,
                tasks=None,
                n_tasks_per_payload=config["n_tasks_per_payload"],
            )
            _dump_episodes(episodes, output_dir / "episodes.jsonl")
    else:
        logger.info("Running in-distribution evaluation...")
        episodes = evaluator.run_episodes(
            payloads=payloads,
            tasks=None,
            n_tasks_per_payload=config["n_tasks_per_payload"],
        )
        results["id_metrics"] = aggregate_metrics(episodes, prefix="id")
        results["id_n_episodes"] = len(episodes)

        if config.get("save_episodes"):
            _dump_episodes(episodes, output_dir / "episodes.jsonl")

    # ---- False-positive check (Blue Team on clean tasks) ----
    fpr_n = int(config.get("fpr_samples") or 0)
    if fpr_n > 0:
        logger.info("Running false-positive check on %d clean tasks...", fpr_n)
        results["fpr_check"] = evaluator.false_positive_check(n_samples=fpr_n)
        fpr_rate = results["fpr_check"].get("false_positive_rate", 0.0)
        if "id_metrics" in results:
            results["id_metrics"]["id/blue_false_positive_rate"] = fpr_rate
        if "ood_metrics" in results:
            results["ood_metrics"]["ood/blue_false_positive_rate"] = fpr_rate
    else:
        logger.info("Skipping FPR check (fpr_samples=0).")

    results["wall_time_s"] = round(time.time() - t0, 1)

    # ---- Strategy DB snapshot (coverage of the loaded checkpoint) ----
    db = MAPElitesDB()
    db.load(Path(config["checkpoint_dir"]) / "strategy_db.json")
    results["checkpoint_strategy_db"] = db.niche_stats()

    # ---- Save + print ----
    results_path = output_dir / "cross_eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    if "id_metrics" in results:
        print_metrics(results["id_metrics"], title="In-Distribution")
    if "ood_metrics" in results:
        print_metrics(results["ood_metrics"], title="Out-of-Distribution")
        logger.info("Transfer ratio (OOD-ASR / ID-ASR): %.2f", results["transfer_ratio"])
        print_metrics(
            compare_metrics(results["id_metrics"], results["ood_metrics"]),
            title="ID → OOD Delta",
        )
    if "fpr_check" in results:
        print_metrics(results["fpr_check"], title="Blue Team False-Positive Check")

    logger.info("Done in %.1fs.", results["wall_time_s"])


if __name__ == "__main__":
    main()
