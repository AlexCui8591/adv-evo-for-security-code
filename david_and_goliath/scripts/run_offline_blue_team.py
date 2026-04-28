"""Offline Stage-2 Blue Team runner.

Consumes Stage-1 `rollouts.jsonl`, reconstructs each injected task, and
generates Blue Team responses into an append-only `blue_responses.jsonl`.

Key properties:
  - Resume by `episode_key = "{payload_id}:{task_id}"`
  - Skip existing rows with `status == "ok"`
  - Retry failed rows on the next run
  - Concurrent execution via `asyncio.Semaphore`
  - Windows-friendly pure Python orchestration
"""

from __future__ import annotations

import argparse
import asyncio
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

from blue_team.coding_agent import CodingAgent
from core.injection_engine import InjectionEngine, load_coding_tasks
from core.types import BlueTeamResponse, Carrier, Payload


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "coding_tasks_path": "david_and_goliath/data/coding_tasks/tasks.jsonl",
    "rollouts_path": None,
    "blue_responses_path": None,
    "concurrency": 4,
    "limit": None,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="David & Goliath - Offline Blue Team batch runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "--blue-config",
        dest="config",
        type=str,
        default=None,
        help="Optional YAML config merged on top of DEFAULT_CONFIG.",
    )
    parser.add_argument(
        "--rollouts-path",
        type=str,
        default=None,
        help="Input rollouts.jsonl from Stage-1.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output blue_responses.jsonl path.",
    )
    parser.add_argument(
        "--tasks-path",
        type=str,
        default=None,
        help="Override coding tasks JSONL used to reconstruct injected tasks.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Maximum concurrent Blue Team requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N pending rollouts after resume filtering.",
    )
    parser.add_argument(
        "--defense-memory-path",
        type=str,
        default=None,
        help="Optional dynamic Blue defense-memory JSONL used during retrieval.",
    )
    parser.add_argument(
        "--defense-retrieval-top-k",
        type=int,
        default=None,
        help="How many historical defense memories to inject per task.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config, scan input files, print summary, and exit.",
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


def _default_blue_responses_path(rollouts_path: str) -> str:
    rollouts = Path(rollouts_path)
    if rollouts.parent.name == "rollouts":
        return str(rollouts.parent.parent / "blue_team" / "blue_responses.jsonl")
    return str(rollouts.parent / "blue_responses.jsonl")


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)

    yaml_cfg: dict[str, Any] = {}
    if args.config:
        yaml_cfg = _load_yaml(args.config)
        config = _deep_merge(config, yaml_cfg)

    if args.rollouts_path:
        config["rollouts_path"] = args.rollouts_path
    if args.output_path:
        config["blue_responses_path"] = args.output_path
    if args.tasks_path:
        config["coding_tasks_path"] = args.tasks_path
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency
    if args.limit is not None:
        config["limit"] = args.limit
    if args.defense_memory_path is not None:
        config["blue_team"]["defense_memory_path"] = args.defense_memory_path
    if args.defense_retrieval_top_k is not None:
        config["blue_team"]["defense_retrieval_top_k"] = args.defense_retrieval_top_k

    if not config.get("rollouts_path"):
        raise ValueError("rollouts_path is required. Pass --rollouts-path or set it in YAML.")

    if not config.get("blue_responses_path"):
        config["blue_responses_path"] = _default_blue_responses_path(config["rollouts_path"])

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


def _derive_episode_key(row: dict[str, Any]) -> Optional[str]:
    episode_key = row.get("episode_key")
    if episode_key:
        return str(episode_key)

    payload_id = row.get("payload_id")
    task_id = row.get("task_id")
    if payload_id and task_id:
        return f"{payload_id}:{task_id}"
    return None


def _load_rollouts(path: Path) -> tuple[list[dict[str, Any]], int]:
    deduped: dict[str, dict[str, Any]] = {}
    duplicate_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.lstrip("\ufeff").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logging.getLogger(__name__).warning(
                    "Skipping malformed rollout line %d in %s", line_num, path
                )
                continue

            episode_key = _derive_episode_key(row)
            if not episode_key:
                logging.getLogger(__name__).warning(
                    "Skipping rollout line %d with missing episode_key/task_id", line_num
                )
                continue

            if episode_key in deduped:
                duplicate_count += 1
            row["episode_key"] = episode_key
            deduped[episode_key] = row

    return list(deduped.values()), duplicate_count


def _load_completed_ok_keys(path: Path) -> tuple[set[str], int]:
    if not path.exists():
        return set(), 0

    completed: set[str] = set()
    error_rows = 0
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
                continue

            if row.get("status") == "ok":
                completed.add(episode_key)
            elif row.get("status") == "error":
                error_rows += 1

    return completed, error_rows


def _build_blue_team(config: dict[str, Any]) -> CodingAgent:
    blue_cfg = config["blue_team"]
    return CodingAgent(
        model=blue_cfg.get("model", "gpt-4o-mini"),
        api_key=blue_cfg.get("api_key"),
        base_url=blue_cfg.get("base_url"),
        temperature=blue_cfg.get("temperature", 0.2),
        max_turns=blue_cfg.get("max_turns", 6),
        max_reflexion=blue_cfg.get("max_reflexion", 2),
        use_tools=blue_cfg.get("use_tools", True),
        enable_static_memory_scan=blue_cfg.get("enable_static_memory_scan", True),
        enable_defense_memory_retrieval=blue_cfg.get(
            "enable_defense_memory_retrieval", True
        ),
        defense_memory_path=blue_cfg.get("defense_memory_path"),
        defense_retrieval_top_k=blue_cfg.get("defense_retrieval_top_k", 3),
    )


def _compact_blue_response(resp: BlueTeamResponse) -> dict[str, Any]:
    return {
        "generated_code": resp.generated_code,
        "tool_calls": [tc.to_dict() for tc in resp.tool_calls],
        "num_tool_calls": resp.num_tool_calls,
        "tools_used": resp.tools_used,
        "latency_ms": resp.latency_ms,
        "detected_suspicious": resp.detected_suspicious,
        "verification": resp.verification,
        "memory_scan": resp.memory_scan,
        "retrieved_memories": resp.retrieved_memories,
        "defense_context_applied": resp.defense_context_applied,
    }


def _base_output_row(
    rollout_row: dict[str, Any],
    blue_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode_key": rollout_row["episode_key"],
        "round": rollout_row.get("round"),
        "oracle_mode": rollout_row.get("oracle_mode"),
        "payload_id": rollout_row.get("payload_id"),
        "payload_code": rollout_row.get("payload_code"),
        "prompt_used": rollout_row.get("prompt_used"),
        "task_id": rollout_row.get("task_id"),
        "injection_position": rollout_row.get("injection_position"),
        "status": "error",
        "error": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "blue_model": blue_cfg.get("model"),
        "blue_base_url": blue_cfg.get("base_url"),
        "blue_use_tools": blue_cfg.get("use_tools", True),
        "blue_enable_static_memory_scan": blue_cfg.get("enable_static_memory_scan", True),
        "blue_enable_defense_memory_retrieval": blue_cfg.get(
            "enable_defense_memory_retrieval", True
        ),
        "blue_defense_memory_path": blue_cfg.get("defense_memory_path"),
        "blue_defense_retrieval_top_k": blue_cfg.get("defense_retrieval_top_k", 3),
        "blue_response": None,
    }


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

    payload = Payload(
        id=str(rollout_row.get("payload_id", "")),
        round_created=int(rollout_row.get("round") or 0),
        code=str(rollout_row.get("payload_code") or ""),
        prompt_used=str(rollout_row.get("prompt_used") or ""),
    )

    carrier_name = rollout_row.get("injection_position")
    if not carrier_name:
        raise ValueError("rollout row missing injection_position")

    try:
        carrier = Carrier[str(carrier_name)]
    except KeyError as exc:
        raise ValueError(f"unknown injection_position/carrier: {carrier_name!r}") from exc

    return injection_engine.inject(
        payload=payload,
        task=tasks_by_id[str(task_id)],
        carrier=carrier,
    )


def _run_single_rollout(
    rollout_row: dict[str, Any],
    tasks_by_id: dict[str, Any],
    injection_engine: InjectionEngine,
    blue_cfg: dict[str, Any],
) -> dict[str, Any]:
    out = _base_output_row(rollout_row, blue_cfg)
    try:
        injected_task = _reconstruct_injected_task(
            rollout_row=rollout_row,
            tasks_by_id=tasks_by_id,
            injection_engine=injection_engine,
        )
        agent = _build_blue_team({"blue_team": blue_cfg})
        response = agent.process(injected_task)
        out["status"] = "ok"
        out["blue_response"] = _compact_blue_response(response)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _run_pending(
    pending_rows: list[dict[str, Any]],
    output_path: Path,
    tasks_by_id: dict[str, Any],
    injection_engine: InjectionEngine,
    config: dict[str, Any],
) -> tuple[int, int]:
    logger = logging.getLogger(__name__)
    semaphore = asyncio.Semaphore(int(config["concurrency"]))
    blue_cfg = config["blue_team"]

    async def _run_one(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(
                _run_single_rollout,
                row,
                tasks_by_id,
                injection_engine,
                blue_cfg,
            )

    tasks = [asyncio.create_task(_run_one(row)) for row in pending_rows]
    ok_count = 0
    error_count = 0

    for idx, future in enumerate(asyncio.as_completed(tasks), 1):
        row = await future
        _append_jsonl(output_path, row)
        if row["status"] == "ok":
            ok_count += 1
        else:
            error_count += 1

        if idx == 1 or idx % 10 == 0 or idx == len(pending_rows):
            logger.info(
                "Processed %d/%d pending rollouts (ok=%d, error=%d)",
                idx,
                len(pending_rows),
                ok_count,
                error_count,
            )

    return ok_count, error_count


async def _main_async(config: dict[str, Any], dry_run: bool) -> None:
    logger = logging.getLogger(__name__)
    rollouts_path = Path(config["rollouts_path"])
    output_path = Path(config["blue_responses_path"])

    if not rollouts_path.exists():
        raise FileNotFoundError(f"rollouts_path does not exist: {rollouts_path}")

    coding_tasks = load_coding_tasks(config["coding_tasks_path"])
    tasks_by_id = {task.id: task for task in coding_tasks}
    injection_engine = InjectionEngine(coding_tasks=coding_tasks, seed=config["seed"])

    rollout_rows, duplicate_rollouts = _load_rollouts(rollouts_path)
    completed_ok_keys, previous_error_rows = _load_completed_ok_keys(output_path)
    pending_rows = [
        row for row in rollout_rows if row["episode_key"] not in completed_ok_keys
    ]

    if config.get("limit") is not None:
        pending_rows = pending_rows[: int(config["limit"])]

    logger.info("=" * 60)
    logger.info("Offline Blue Team Stage-2")
    logger.info("=" * 60)
    logger.info("Rollouts      : %s", rollouts_path)
    logger.info("Output        : %s", output_path)
    logger.info("Tasks         : %s", config["coding_tasks_path"])
    logger.info("Blue model    : %s", config["blue_team"]["model"])
    logger.info("Use tools     : %s", config["blue_team"].get("use_tools", True))
    logger.info(
        "Static memory : %s",
        config["blue_team"].get("enable_static_memory_scan", True),
    )
    logger.info(
        "Dynamic memory: %s",
        config["blue_team"].get("enable_defense_memory_retrieval", True),
    )
    logger.info(
        "Defense memory: %s",
        config["blue_team"].get("defense_memory_path") or "(none)",
    )
    logger.info(
        "Defense top-k : %s",
        config["blue_team"].get("defense_retrieval_top_k", 3),
    )
    logger.info("Concurrency   : %d", config["concurrency"])
    logger.info("Loaded tasks  : %d", len(tasks_by_id))
    logger.info("Rollout rows  : %d", len(rollout_rows))
    logger.info("Duplicates    : %d", duplicate_rollouts)
    logger.info("Completed ok  : %d", len(completed_ok_keys))
    logger.info("Previous errs : %d", previous_error_rows)
    logger.info("Pending       : %d", len(pending_rows))

    if dry_run:
        logger.info("--dry-run: exiting before Blue Team execution.")
        return

    if not pending_rows:
        logger.info("Nothing to do.")
        return

    t0 = time.time()
    ok_count, error_count = await _run_pending(
        pending_rows=pending_rows,
        output_path=output_path,
        tasks_by_id=tasks_by_id,
        injection_engine=injection_engine,
        config=config,
    )
    logger.info(
        "Finished Stage-2 in %.1fs (ok=%d, error=%d)",
        time.time() - t0,
        ok_count,
        error_count,
    )


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    _setup_logging(args.log_level, config["blue_responses_path"])
    asyncio.run(_main_async(config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
