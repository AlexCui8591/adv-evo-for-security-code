"""Offline Stage-3 judging and memory writeback.

Consumes:
  - Stage-1 `rollouts.jsonl`
  - Stage-2 `blue_responses.jsonl`

Produces:
  - `memory/episodes.jsonl` via JsonlMemoryStore

The script reuses cached Judge C results from Stage-1 through
`HybridOracle.evaluate_episode_with_cached_jc()` so Stage-3 only needs to run
Judge A and Judge B offline.
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

from core.injection_engine import InjectionEngine, load_coding_tasks
from core.types import (
    BlueTeamResponse,
    Carrier,
    InjectionType,
    JudgeCResult,
    Payload,
    StealthLevel,
    ToolCall,
)
from infra.memory_store import JsonlMemoryStore
from scripts.run_offline_blue_team import (
    _derive_episode_key,
    _load_rollouts,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "coding_tasks_path": "david_and_goliath/data/coding_tasks/tasks.jsonl",
    "rollouts_path": None,
    "blue_responses_path": None,
    "output_path": None,
    "limit": None,
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
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="David & Goliath - Offline Stage-3 judging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "--oracle-config",
        dest="config",
        type=str,
        default=None,
        help="Optional YAML config merged on top of DEFAULT_CONFIG.",
    )
    parser.add_argument(
        "--rollouts-path",
        type=str,
        default=None,
        help="Input Stage-1 rollouts.jsonl.",
    )
    parser.add_argument(
        "--blue-responses-path",
        type=str,
        default=None,
        help="Input Stage-2 blue_responses.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output memory episodes JSONL path.",
    )
    parser.add_argument(
        "--tasks-path",
        type=str,
        default=None,
        help="Override coding tasks JSONL used to reconstruct injected tasks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N pending episodes after resume filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config, scan inputs, print summary, and exit.",
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


def _default_episodes_path(blue_responses_path: str) -> str:
    blue_path = Path(blue_responses_path)
    if blue_path.parent.name == "blue_team":
        return str(blue_path.parent.parent / "memory" / "episodes.jsonl")
    return str(blue_path.parent / "memory" / "episodes.jsonl")


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)

    if args.config:
        config = _deep_merge(config, _load_yaml(args.config))

    if args.rollouts_path:
        config["rollouts_path"] = args.rollouts_path
    if args.blue_responses_path:
        config["blue_responses_path"] = args.blue_responses_path
    if args.output_path:
        config["output_path"] = args.output_path
    if args.tasks_path:
        config["coding_tasks_path"] = args.tasks_path
    if args.limit is not None:
        config["limit"] = args.limit

    if not config.get("rollouts_path"):
        raise ValueError("rollouts_path is required. Pass --rollouts-path or set it in YAML.")
    if not config.get("blue_responses_path"):
        raise ValueError(
            "blue_responses_path is required. Pass --blue-responses-path or set it in YAML."
        )
    if not config.get("output_path"):
        config["output_path"] = _default_episodes_path(config["blue_responses_path"])

    return config


def _setup_logging(level: str, output_path: str) -> None:
    output_dir = Path(output_path).parent
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=getattr(logging, level), format=fmt, handlers=handlers)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _load_blue_responses(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    ok_rows: dict[str, dict[str, Any]] = {}
    skipped_rows = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.lstrip("\ufeff").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logging.getLogger(__name__).warning(
                    "Skipping malformed blue-response line %d in %s", line_num, path
                )
                continue

            episode_key = _derive_episode_key(row)
            if not episode_key:
                skipped_rows += 1
                continue

            if row.get("status") == "ok":
                row["episode_key"] = episode_key
                ok_rows[episode_key] = row
            else:
                skipped_rows += 1

    return ok_rows, skipped_rows


def _build_oracle(config: dict[str, Any]):
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
        strategy_db=None,
        w_vulnerability=oc.get("w_vulnerability", 0.30),
        w_manipulation=oc.get("w_manipulation", 0.30),
        w_quality=oc.get("w_quality", 0.20),
        w_diversity=oc.get("w_diversity", 0.10),
        w_stealth=oc.get("w_stealth", 0.10),
        failure_quality_scale=oc.get("failure_quality_scale", 0.15),
    )


def _parse_optional_enum(enum_cls, value: Optional[str]):
    if not value:
        return None
    try:
        return enum_cls[str(value)]
    except KeyError:
        return None


def _reconstruct_payload(rollout_row: dict[str, Any]) -> Payload:
    payload = Payload(
        id=str(rollout_row.get("payload_id", "")),
        round_created=int(rollout_row.get("round") or 0),
        code=str(rollout_row.get("payload_code") or ""),
        prompt_used=str(rollout_row.get("prompt_used") or ""),
    )
    payload.injection_type = _parse_optional_enum(
        InjectionType,
        rollout_row.get("injection_type"),
    )
    payload.stealth_level = _parse_optional_enum(
        StealthLevel,
        rollout_row.get("stealth_level"),
    )
    return payload


def _reconstruct_injected_task(
    rollout_row: dict[str, Any],
    tasks_by_id: dict[str, Any],
    injection_engine: InjectionEngine,
):
    task_id = rollout_row.get("task_id")
    if not task_id:
        raise ValueError("rollout row missing task_id")
    if task_id not in tasks_by_id:
        raise KeyError(f"task_id {task_id!r} not found in coding_tasks_path")

    carrier_name = rollout_row.get("injection_position")
    if not carrier_name:
        raise ValueError("rollout row missing injection_position")

    try:
        carrier = Carrier[str(carrier_name)]
    except KeyError as exc:
        raise ValueError(f"unknown injection_position/carrier: {carrier_name!r}") from exc

    payload = _reconstruct_payload(rollout_row)
    return injection_engine.inject(
        payload=payload,
        task=tasks_by_id[str(task_id)],
        carrier=carrier,
    )


def _reconstruct_blue_response(row: dict[str, Any]) -> BlueTeamResponse:
    blue_row = row.get("blue_response") or {}
    tool_calls = [
        ToolCall(
            tool_name=str(tc.get("tool_name", "")),
            input_args=tc.get("input_args") or {},
            output=str(tc.get("output", "")),
            success=bool(tc.get("success", False)),
            latency_ms=float(tc.get("latency_ms", 0.0)),
        )
        for tc in blue_row.get("tool_calls", [])
    ]
    return BlueTeamResponse(
        generated_code=str(blue_row.get("generated_code", "")),
        tool_calls=tool_calls,
        latency_ms=float(blue_row.get("latency_ms", 0.0)),
        detected_suspicious=bool(blue_row.get("detected_suspicious", False)),
    )


def _reconstruct_judge_c_result(rollout_row: dict[str, Any]) -> JudgeCResult:
    jc_row = rollout_row.get("judge_c")
    if not jc_row:
        raise ValueError("rollout row missing cached judge_c result")

    return JudgeCResult(
        payload_quality_score=float(jc_row.get("payload_quality_score", 0.0)),
        stealth_score=float(jc_row.get("stealth_score", 0.0)),
        creativity_score=float(jc_row.get("creativity_score", 0.0)),
        reasoning=str(jc_row.get("reasoning", "")),
        wall_time_ms=float(jc_row.get("wall_time_ms", 0.0)),
        inferred_injection_type=jc_row.get("inferred_injection_type"),
        inferred_stealth_level=jc_row.get("inferred_stealth_level"),
    )


def _episode_to_record(
    episode_key: str,
    rollout_row: dict[str, Any],
    blue_row: dict[str, Any],
    episode,
) -> dict[str, Any]:
    return {
        "episode_key": episode_key,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "round": episode.round,
        "oracle_mode": rollout_row.get("oracle_mode"),
        "payload_id": episode.payload_id,
        "payload_code": rollout_row.get("payload_code"),
        "prompt_used": rollout_row.get("prompt_used"),
        "coding_task_id": episode.coding_task_id,
        "injection_position": episode.injection_position,
        "attack_success": episode.attack_success,
        "total_reward": episode.total_reward,
        "vulnerability_reward": episode.vulnerability_reward,
        "manipulation_reward": episode.manipulation_reward,
        "quality_reward": episode.quality_reward,
        "diversity_bonus": episode.diversity_bonus,
        "judge_a": episode.judge_a.to_dict() if episode.judge_a else None,
        "judge_b": episode.judge_b.to_dict() if episode.judge_b else None,
        "judge_c": episode.judge_c.to_dict() if episode.judge_c else None,
        "oracle_reward": (
            episode.oracle_reward.to_dict() if episode.oracle_reward else None
        ),
        "stage1_oracle_reward": rollout_row.get("oracle_reward"),
        "blue_team": {
            "model": blue_row.get("blue_model"),
            "base_url": blue_row.get("blue_base_url"),
            "use_tools": blue_row.get("blue_use_tools"),
            "response": blue_row.get("blue_response"),
        },
    }


def _judge_single_episode(
    rollout_row: dict[str, Any],
    blue_row: dict[str, Any],
    tasks_by_id: dict[str, Any],
    injection_engine: InjectionEngine,
    oracle,
) -> dict[str, Any]:
    episode_key = rollout_row["episode_key"]
    injected_task = _reconstruct_injected_task(
        rollout_row=rollout_row,
        tasks_by_id=tasks_by_id,
        injection_engine=injection_engine,
    )
    blue_response = _reconstruct_blue_response(blue_row)
    jc_result = _reconstruct_judge_c_result(rollout_row)
    episode = oracle.evaluate_episode_with_cached_jc(
        payload=injected_task.payload,
        injected_task=injected_task,
        blue_response=blue_response,
        jc_result=jc_result,
        round_num=int(rollout_row.get("round") or 0),
    )
    return _episode_to_record(episode_key, rollout_row, blue_row, episode)


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    _setup_logging(args.log_level, config["output_path"])
    logger = logging.getLogger(__name__)

    rollouts_path = Path(config["rollouts_path"])
    blue_responses_path = Path(config["blue_responses_path"])
    output_path = Path(config["output_path"])

    if not rollouts_path.exists():
        raise FileNotFoundError(f"rollouts_path does not exist: {rollouts_path}")
    if not blue_responses_path.exists():
        raise FileNotFoundError(
            f"blue_responses_path does not exist: {blue_responses_path}"
        )

    coding_tasks = load_coding_tasks(config["coding_tasks_path"])
    if not coding_tasks:
        raise RuntimeError("No coding tasks loaded. Cannot reconstruct injected tasks.")
    tasks_by_id = {task.id: task for task in coding_tasks}
    injection_engine = InjectionEngine(coding_tasks=coding_tasks, seed=config["seed"])
    oracle = _build_oracle(config)

    rollout_rows, duplicate_rollouts = _load_rollouts(rollouts_path)
    blue_rows, skipped_blue_rows = _load_blue_responses(blue_responses_path)
    memory_store = JsonlMemoryStore(output_path, key_field="episode_key")

    joined_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing_blue = 0
    for rollout_row in rollout_rows:
        episode_key = rollout_row["episode_key"]
        blue_row = blue_rows.get(episode_key)
        if blue_row is None:
            missing_blue += 1
            continue
        joined_rows.append((rollout_row, blue_row))

    pending_rows = [
        pair
        for pair in joined_rows
        if not memory_store.contains(pair[0]["episode_key"])
    ]
    if config.get("limit") is not None:
        pending_rows = pending_rows[: int(config["limit"])]

    logger.info("=" * 60)
    logger.info("Offline Stage-3 Judging")
    logger.info("=" * 60)
    logger.info("Rollouts        : %s", rollouts_path)
    logger.info("Blue responses  : %s", blue_responses_path)
    logger.info("Episodes output : %s", output_path)
    logger.info("Tasks           : %s", config["coding_tasks_path"])
    logger.info("Judge model     : %s", config["oracle"]["judge_model"])
    logger.info("Loaded tasks    : %d", len(tasks_by_id))
    logger.info("Rollout rows    : %d", len(rollout_rows))
    logger.info("Rollout dups    : %d", duplicate_rollouts)
    logger.info("Blue ok rows    : %d", len(blue_rows))
    logger.info("Blue skipped    : %d", skipped_blue_rows)
    logger.info("Missing blue    : %d", missing_blue)
    logger.info("Already judged  : %d", len(memory_store))
    logger.info("Pending         : %d", len(pending_rows))

    if args.dry_run:
        logger.info("--dry-run: exiting before offline judging.")
        return

    if not pending_rows:
        logger.info("Nothing to do.")
        return

    ok_count = 0
    error_count = 0
    t0 = time.time()

    for idx, (rollout_row, blue_row) in enumerate(pending_rows, 1):
        episode_key = rollout_row["episode_key"]
        try:
            record = _judge_single_episode(
                rollout_row=rollout_row,
                blue_row=blue_row,
                tasks_by_id=tasks_by_id,
                injection_engine=injection_engine,
                oracle=oracle,
            )
            memory_store.append(record)
            ok_count += 1
        except Exception as exc:
            error_count += 1
            logger.warning("Episode %s failed during Stage-3 judging: %s", episode_key, exc)

        if idx == 1 or idx % 10 == 0 or idx == len(pending_rows):
            logger.info(
                "Processed %d/%d pending episodes (ok=%d, error=%d)",
                idx,
                len(pending_rows),
                ok_count,
                error_count,
            )

    logger.info(
        "Finished Stage-3 in %.1fs (ok=%d, error=%d)",
        time.time() - t0,
        ok_count,
        error_count,
    )


if __name__ == "__main__":
    main()
