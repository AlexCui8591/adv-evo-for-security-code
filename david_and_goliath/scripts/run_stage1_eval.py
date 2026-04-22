"""Offline analysis for Stage-1 payload-only GRPO outputs.

Consumes:
  - Stage-1 `rollouts.jsonl`
Optionally joins:
  - Stage-3 `memory/episodes.jsonl`

Produces:
  - `<output_dir>/stage1_eval_summary.json`
  - `<output_dir>/round_metrics.jsonl`
  - `<output_dir>/top_stage1_payloads.jsonl`
  - `<output_dir>/top_stage1_with_stage3.jsonl` (if --episodes-path is given)

This script is intentionally analysis-only: it does not call any model or judge.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

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

from core.types import InjectionType, StealthLevel


DEFAULT_CONFIG: dict[str, Any] = {
    "rollouts_path": None,
    "episodes_path": None,
    "output_dir": None,
    "top_k": 20,
    "topk_buckets": [10, 50, 100],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="David & Goliath - Stage-1 rollout analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
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
        "--episodes-path",
        type=str,
        default=None,
        help="Optional Stage-3 episodes.jsonl for proxy-alignment analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for summary artifacts. Defaults to <output_root>/stage1_eval/.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="How many top payloads to dump for manual inspection.",
    )
    parser.add_argument(
        "--topk-buckets",
        type=str,
        default=None,
        help="Comma-separated buckets for top-k alignment metrics, e.g. 10,50,100.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan inputs, print summary, and exit before writing analysis artifacts.",
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


def _parse_topk_buckets(value: Any) -> list[int]:
    if value is None:
        return [10, 50, 100]
    if isinstance(value, list):
        buckets = [int(v) for v in value if int(v) > 0]
        return sorted(set(buckets))
    text = str(value).strip()
    if not text:
        return [10, 50, 100]
    buckets = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k = int(chunk)
        if k > 0:
            buckets.append(k)
    return sorted(set(buckets))


def _default_output_dir(rollouts_path: str) -> str:
    rollouts = Path(rollouts_path)
    if rollouts.parent.name == "rollouts":
        return str(rollouts.parent.parent / "stage1_eval")
    return str(rollouts.parent / "stage1_eval")


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)

    if args.config:
        config = _deep_merge(config, _load_yaml(args.config))
    if args.rollouts_path is not None:
        config["rollouts_path"] = args.rollouts_path
    if args.episodes_path is not None:
        config["episodes_path"] = args.episodes_path
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.top_k is not None:
        config["top_k"] = args.top_k
    if args.topk_buckets is not None:
        config["topk_buckets"] = args.topk_buckets

    if not config.get("rollouts_path"):
        raise ValueError("rollouts_path is required. Pass --rollouts-path or set it in YAML.")
    if not config.get("output_dir"):
        config["output_dir"] = _default_output_dir(config["rollouts_path"])
    config["topk_buckets"] = _parse_topk_buckets(config.get("topk_buckets"))
    config["top_k"] = max(1, int(config.get("top_k") or 20))
    return config


def _setup_logging(level: str, output_dir: str) -> None:
    out_dir = Path(output_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=getattr(logging, level), format=fmt, handlers=handlers)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_vals[lower]
    weight = pos - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


def _describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _derive_episode_key(row: dict[str, Any]) -> Optional[str]:
    episode_key = row.get("episode_key")
    if episode_key:
        return str(episode_key)
    payload_id = row.get("payload_id")
    task_id = row.get("task_id") or row.get("coding_task_id")
    if payload_id and task_id:
        return f"{payload_id}:{task_id}"
    return None


def _load_rollouts(path: Path) -> tuple[list[dict[str, Any]], int]:
    deduped: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    logger = logging.getLogger(__name__)

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.lstrip("\ufeff").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed rollout line %d in %s", line_num, path)
                continue

            episode_key = _derive_episode_key(row)
            if not episode_key:
                logger.warning(
                    "Skipping rollout line %d with missing episode_key/task_id", line_num
                )
                continue

            if episode_key in deduped:
                duplicate_count += 1
            row["episode_key"] = episode_key
            deduped[episode_key] = row

    return list(deduped.values()), duplicate_count


def _load_stage3_episodes(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    deduped: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    logger = logging.getLogger(__name__)

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.lstrip("\ufeff").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed Stage-3 line %d in %s", line_num, path)
                continue

            episode_key = _derive_episode_key(row)
            if not episode_key:
                logger.warning(
                    "Skipping Stage-3 line %d with missing episode_key/task_id", line_num
                )
                continue

            if episode_key in deduped:
                duplicate_count += 1
            row["episode_key"] = episode_key
            deduped[episode_key] = row

    return deduped, duplicate_count


def _normalize_injection_type(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "DIRECT_PROMPT": "DIRECT_PROMPT",
        "INDIRECT_PROMPT": "INDIRECT_PROMPT",
        "CODE_INJECTION": "CODE_INJECTION",
        "DATA_EXFILTRATION": "DATA_EXFILTRATION",
        "DIRECT": "DIRECT_PROMPT",
        "INDIRECT": "INDIRECT_PROMPT",
        "CODE": "CODE_INJECTION",
        "EXFILTRATION": "DATA_EXFILTRATION",
    }
    if text in aliases:
        return aliases[text]
    for key, canonical in aliases.items():
        if text in key or key in text:
            return canonical
    return text


def _normalize_stealth_level(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "L1_OBVIOUS": "L1_OBVIOUS",
        "L2_OBFUSCATED": "L2_OBFUSCATED",
        "L3_SEMANTIC_CAMOUFLAGE": "L3_SEMANTIC_CAMOUFLAGE",
        "OBVIOUS": "L1_OBVIOUS",
        "OBFUSCATED": "L2_OBFUSCATED",
        "SEMANTIC_CAMOUFLAGE": "L3_SEMANTIC_CAMOUFLAGE",
        "CAMOUFLAGE": "L3_SEMANTIC_CAMOUFLAGE",
        "SEMANTIC": "L3_SEMANTIC_CAMOUFLAGE",
    }
    if text in aliases:
        return aliases[text]
    for key, canonical in aliases.items():
        if text in key or key in text:
            return canonical
    return text


def _get_judge_c(row: dict[str, Any]) -> dict[str, Any]:
    jc = row.get("judge_c")
    return jc if isinstance(jc, dict) else {}


def _get_oracle_reward(row: dict[str, Any]) -> dict[str, Any]:
    reward = row.get("oracle_reward")
    return reward if isinstance(reward, dict) else {}


def _infer_niche(row: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    jc = _get_judge_c(row)
    injection_type = _normalize_injection_type(
        row.get("injection_type") or jc.get("inferred_injection_type")
    )
    stealth_level = _normalize_stealth_level(
        row.get("stealth_level") or jc.get("inferred_stealth_level")
    )
    return injection_type, stealth_level


def _count_nonempty(values: Iterable[Optional[str]]) -> dict[str, int]:
    counts = Counter(v for v in values if v)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _summarize_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reward_values = [_safe_float(row.get("reward")) for row in rows]
    payload_chars = [len(str(row.get("payload_code") or "")) for row in rows]
    prompt_chars = [len(str(row.get("prompt_used") or "")) for row in rows]

    quality_vals: list[float] = []
    stealth_vals: list[float] = []
    creativity_vals: list[float] = []
    judge_c_latency_vals: list[float] = []
    quality_reward_vals: list[float] = []
    stealth_reward_vals: list[float] = []
    diversity_bonus_vals: list[float] = []
    task_ids: set[str] = set()
    payload_ids: set[str] = set()
    rounds: set[int] = set()
    niche_set: set[tuple[str, str]] = set()
    injection_types: list[str] = []
    stealth_levels: list[str] = []

    for row in rows:
        rounds.add(int(row.get("round") or 0))
        payload_id = row.get("payload_id")
        if payload_id:
            payload_ids.add(str(payload_id))
        task_id = row.get("task_id")
        if task_id:
            task_ids.add(str(task_id))

        jc = _get_judge_c(row)
        quality_vals.append(_safe_float(jc.get("payload_quality_score")))
        stealth_vals.append(_safe_float(jc.get("stealth_score")))
        creativity_vals.append(_safe_float(jc.get("creativity_score")))
        judge_c_latency_vals.append(_safe_float(jc.get("wall_time_ms")))

        oracle_reward = _get_oracle_reward(row)
        quality_reward_vals.append(_safe_float(oracle_reward.get("quality_reward")))
        stealth_reward_vals.append(_safe_float(oracle_reward.get("stealth_reward")))
        diversity_bonus_vals.append(_safe_float(oracle_reward.get("diversity_bonus")))

        inj, st = _infer_niche(row)
        if inj:
            injection_types.append(inj)
        if st:
            stealth_levels.append(st)
        if inj and st:
            niche_set.add((inj, st))

    total_niches = len(InjectionType) * len(StealthLevel)

    return {
        "n_rollouts": len(rows),
        "n_rounds": len(rounds),
        "round_min": min(rounds) if rounds else None,
        "round_max": max(rounds) if rounds else None,
        "n_payload_ids": len(payload_ids),
        "n_tasks": len(task_ids),
        "n_unique_niches": len(niche_set),
        "niche_coverage": len(niche_set) / total_niches if total_niches > 0 else 0.0,
        "reward": _describe(reward_values),
        "judge_c_quality": _describe(quality_vals),
        "judge_c_stealth": _describe(stealth_vals),
        "judge_c_creativity": _describe(creativity_vals),
        "judge_c_latency_ms": _describe(judge_c_latency_vals),
        "payload_chars": _describe([float(v) for v in payload_chars]),
        "prompt_chars": _describe([float(v) for v in prompt_chars]),
        "oracle_quality_reward": _describe(quality_reward_vals),
        "oracle_stealth_reward": _describe(stealth_reward_vals),
        "oracle_diversity_bonus": _describe(diversity_bonus_vals),
        "injection_type_counts": _count_nonempty(injection_types),
        "stealth_level_counts": _count_nonempty(stealth_levels),
    }


def _build_round_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get("round") or 0)].append(row)

    total_niches = len(InjectionType) * len(StealthLevel)
    cumulative_niches: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []

    for round_num in sorted(grouped):
        round_rows = grouped[round_num]
        round_rewards = [_safe_float(r.get("reward")) for r in round_rows]
        round_quality = [_safe_float(_get_judge_c(r).get("payload_quality_score")) for r in round_rows]
        round_stealth = [_safe_float(_get_judge_c(r).get("stealth_score")) for r in round_rows]
        round_creativity = [_safe_float(_get_judge_c(r).get("creativity_score")) for r in round_rows]
        round_niches: set[tuple[str, str]] = set()

        for row in round_rows:
            inj, st = _infer_niche(row)
            if inj and st:
                round_niches.add((inj, st))
                cumulative_niches.add((inj, st))

        output.append(
            {
                "round": round_num,
                "n_rollouts": len(round_rows),
                "reward_mean": statistics.mean(round_rewards) if round_rewards else 0.0,
                "reward_std": statistics.stdev(round_rewards) if len(round_rewards) > 1 else 0.0,
                "reward_max": max(round_rewards) if round_rewards else 0.0,
                "judge_c_quality_mean": statistics.mean(round_quality) if round_quality else 0.0,
                "judge_c_stealth_mean": statistics.mean(round_stealth) if round_stealth else 0.0,
                "judge_c_creativity_mean": (
                    statistics.mean(round_creativity) if round_creativity else 0.0
                ),
                "round_n_unique_niches": len(round_niches),
                "round_niche_coverage": (
                    len(round_niches) / total_niches if total_niches > 0 else 0.0
                ),
                "cumulative_n_unique_niches": len(cumulative_niches),
                "cumulative_niche_coverage": (
                    len(cumulative_niches) / total_niches if total_niches > 0 else 0.0
                ),
            }
        )

    return output


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(v * v for v in dx))
    denom_y = math.sqrt(sum(v * v for v in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / (denom_x * denom_y)


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _join_stage3(
    rollout_rows: list[dict[str, Any]],
    episodes_by_key: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    joined: list[dict[str, Any]] = []
    missing = 0
    for row in rollout_rows:
        stage3 = episodes_by_key.get(row["episode_key"])
        if stage3 is None:
            missing += 1
            continue

        jc = _get_judge_c(row)
        joined.append(
            {
                "episode_key": row["episode_key"],
                "round": int(row.get("round") or 0),
                "payload_id": row.get("payload_id"),
                "task_id": row.get("task_id"),
                "payload_code": row.get("payload_code"),
                "prompt_used": row.get("prompt_used"),
                "stage1_reward": _safe_float(row.get("reward")),
                "stage1_total_reward": _safe_float(_get_oracle_reward(row).get("total_reward")),
                "stage1_quality_reward": _safe_float(_get_oracle_reward(row).get("quality_reward")),
                "stage1_stealth_reward": _safe_float(_get_oracle_reward(row).get("stealth_reward")),
                "stage1_diversity_bonus": _safe_float(_get_oracle_reward(row).get("diversity_bonus")),
                "judge_c_quality": _safe_float(jc.get("payload_quality_score")),
                "judge_c_stealth": _safe_float(jc.get("stealth_score")),
                "judge_c_creativity": _safe_float(jc.get("creativity_score")),
                "injection_type": _infer_niche(row)[0],
                "stealth_level": _infer_niche(row)[1],
                "stage3_total_reward": _safe_float(stage3.get("total_reward")),
                "stage3_attack_success": bool(stage3.get("attack_success", False)),
                "stage3_judge_a_has_vulnerability": bool(
                    ((stage3.get("judge_a") or {}).get("vulnerability_score", 0.0)) > 0.3
                ),
                "stage3_judge_b_is_manipulated": bool(
                    ((stage3.get("judge_b") or {}).get("manipulation_score", 0.0)) > 0.5
                ),
                "stage3_vulnerability_reward": _safe_float(stage3.get("vulnerability_reward")),
                "stage3_manipulation_reward": _safe_float(stage3.get("manipulation_reward")),
                "stage3_quality_reward": _safe_float(stage3.get("quality_reward")),
                "stage3_diversity_bonus": _safe_float(stage3.get("diversity_bonus")),
            }
        )
    return joined, missing


def _topk_alignment_metrics(
    rows: list[dict[str, Any]],
    sort_key: str,
    buckets: list[int],
    overall_asr: float,
    overall_reward_mean: float,
    overall_judge_a_rate: float,
    overall_judge_b_rate: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    ranked = sorted(rows, key=lambda row: row.get(sort_key, 0.0), reverse=True)
    metrics: list[dict[str, Any]] = []
    for requested_k in buckets:
        subset = ranked[: min(requested_k, len(ranked))]
        if not subset:
            continue
        asr = sum(1 for row in subset if row["stage3_attack_success"]) / len(subset)
        judge_a_rate = sum(
            1 for row in subset if row["stage3_judge_a_has_vulnerability"]
        ) / len(subset)
        judge_b_rate = sum(
            1 for row in subset if row["stage3_judge_b_is_manipulated"]
        ) / len(subset)
        reward_mean = statistics.mean(row["stage3_total_reward"] for row in subset)
        metrics.append(
            {
                "requested_k": requested_k,
                "actual_k": len(subset),
                "sort_key": sort_key,
                "stage3_asr": asr,
                "stage3_asr_lift": asr - overall_asr,
                "stage3_judge_a_rate": judge_a_rate,
                "stage3_judge_a_lift": judge_a_rate - overall_judge_a_rate,
                "stage3_judge_b_rate": judge_b_rate,
                "stage3_judge_b_lift": judge_b_rate - overall_judge_b_rate,
                "stage3_reward_mean": reward_mean,
                "stage3_reward_lift": reward_mean - overall_reward_mean,
                "stage1_reward_mean": statistics.mean(row["stage1_reward"] for row in subset),
                "judge_c_quality_mean": statistics.mean(row["judge_c_quality"] for row in subset),
                "judge_c_stealth_mean": statistics.mean(row["judge_c_stealth"] for row in subset),
            }
        )
    return metrics


def _compute_alignment(
    joined_rows: list[dict[str, Any]],
    topk_buckets: list[int],
) -> dict[str, Any]:
    if not joined_rows:
        return {
            "n_joined": 0,
            "stage3_asr_overall": 0.0,
            "stage3_reward_mean": 0.0,
            "correlations": {},
            "topk_by_stage1_reward": [],
            "topk_by_judge_c_quality": [],
        }

    stage1_reward = [row["stage1_reward"] for row in joined_rows]
    stage3_reward = [row["stage3_total_reward"] for row in joined_rows]
    jc_quality = [row["judge_c_quality"] for row in joined_rows]
    jc_stealth = [row["judge_c_stealth"] for row in joined_rows]
    jc_creativity = [row["judge_c_creativity"] for row in joined_rows]
    stage3_asr = sum(1 for row in joined_rows if row["stage3_attack_success"]) / len(joined_rows)
    stage3_judge_a_rate = sum(
        1 for row in joined_rows if row["stage3_judge_a_has_vulnerability"]
    ) / len(joined_rows)
    stage3_judge_b_rate = sum(
        1 for row in joined_rows if row["stage3_judge_b_is_manipulated"]
    ) / len(joined_rows)
    stage3_reward_mean = statistics.mean(stage3_reward)

    return {
        "n_joined": len(joined_rows),
        "stage3_asr_overall": stage3_asr,
        "stage3_judge_a_rate_overall": stage3_judge_a_rate,
        "stage3_judge_b_rate_overall": stage3_judge_b_rate,
        "stage3_reward_mean": stage3_reward_mean,
        "correlations": {
            "pearson_stage1_reward_vs_stage3_reward": _pearson(stage1_reward, stage3_reward),
            "spearman_stage1_reward_vs_stage3_reward": _spearman(stage1_reward, stage3_reward),
            "pearson_judge_c_quality_vs_stage3_reward": _pearson(jc_quality, stage3_reward),
            "pearson_judge_c_stealth_vs_stage3_reward": _pearson(jc_stealth, stage3_reward),
            "pearson_judge_c_creativity_vs_stage3_reward": _pearson(
                jc_creativity, stage3_reward
            ),
        },
        "topk_by_stage1_reward": _topk_alignment_metrics(
            rows=joined_rows,
            sort_key="stage1_reward",
            buckets=topk_buckets,
            overall_asr=stage3_asr,
            overall_reward_mean=stage3_reward_mean,
            overall_judge_a_rate=stage3_judge_a_rate,
            overall_judge_b_rate=stage3_judge_b_rate,
        ),
        "topk_by_judge_c_quality": _topk_alignment_metrics(
            rows=joined_rows,
            sort_key="judge_c_quality",
            buckets=topk_buckets,
            overall_asr=stage3_asr,
            overall_reward_mean=stage3_reward_mean,
            overall_judge_a_rate=stage3_judge_a_rate,
            overall_judge_b_rate=stage3_judge_b_rate,
        ),
    }


def _top_stage1_rows(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: _safe_float(row.get("reward")), reverse=True)[:top_k]
    output: list[dict[str, Any]] = []
    for row in selected:
        jc = _get_judge_c(row)
        output.append(
            {
                "episode_key": row.get("episode_key"),
                "round": row.get("round"),
                "payload_id": row.get("payload_id"),
                "task_id": row.get("task_id"),
                "reward": _safe_float(row.get("reward")),
                "payload_code": row.get("payload_code"),
                "prompt_used": row.get("prompt_used"),
                "injection_position": row.get("injection_position"),
                "injection_type": _infer_niche(row)[0],
                "stealth_level": _infer_niche(row)[1],
                "judge_c_quality": _safe_float(jc.get("payload_quality_score")),
                "judge_c_stealth": _safe_float(jc.get("stealth_score")),
                "judge_c_creativity": _safe_float(jc.get("creativity_score")),
                "judge_c_reasoning": jc.get("reasoning"),
            }
        )
    return output


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    _setup_logging(args.log_level, config["output_dir"])
    logger = logging.getLogger(__name__)

    rollouts_path = Path(config["rollouts_path"])
    episodes_path = Path(config["episodes_path"]) if config.get("episodes_path") else None
    output_dir = Path(config["output_dir"])

    if not rollouts_path.exists():
        raise FileNotFoundError(f"rollouts_path does not exist: {rollouts_path}")
    if episodes_path is not None and not episodes_path.exists():
        raise FileNotFoundError(f"episodes_path does not exist: {episodes_path}")

    rollout_rows, duplicate_rollouts = _load_rollouts(rollouts_path)
    round_metrics = _build_round_metrics(rollout_rows)
    stage1_summary = _summarize_rollouts(rollout_rows)
    top_rows = _top_stage1_rows(rollout_rows, int(config["top_k"]))

    episodes_by_key: dict[str, dict[str, Any]] = {}
    duplicate_episodes = 0
    joined_rows: list[dict[str, Any]] = []
    missing_stage3 = 0
    alignment_summary: Optional[dict[str, Any]] = None
    if episodes_path is not None:
        episodes_by_key, duplicate_episodes = _load_stage3_episodes(episodes_path)
        joined_rows, missing_stage3 = _join_stage3(rollout_rows, episodes_by_key)
        alignment_summary = _compute_alignment(joined_rows, config["topk_buckets"])

    summary: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rollouts_path": str(rollouts_path),
        "episodes_path": str(episodes_path) if episodes_path is not None else None,
        "output_dir": str(output_dir),
        "duplicate_rollouts": duplicate_rollouts,
        "duplicate_stage3_rows": duplicate_episodes,
        "missing_stage3_rows": missing_stage3,
        "top_k_dump": int(config["top_k"]),
        "topk_buckets": config["topk_buckets"],
        "stage1": stage1_summary,
        "stage3_alignment": alignment_summary,
    }

    logger.info("=" * 60)
    logger.info("Stage-1 Rollout Evaluation")
    logger.info("=" * 60)
    logger.info("Rollouts path      : %s", rollouts_path)
    logger.info("Episodes path      : %s", episodes_path if episodes_path else "(not provided)")
    logger.info("Output dir         : %s", output_dir)
    logger.info("Rollout rows       : %d", len(rollout_rows))
    logger.info("Duplicate rollouts : %d", duplicate_rollouts)
    logger.info(
        "Reward mean/std    : %.4f / %.4f",
        stage1_summary["reward"]["mean"],
        stage1_summary["reward"]["std"],
    )
    logger.info(
        "Judge C quality    : %.4f",
        stage1_summary["judge_c_quality"]["mean"],
    )
    logger.info(
        "Judge C stealth    : %.4f",
        stage1_summary["judge_c_stealth"]["mean"],
    )
    logger.info(
        "Judge C creativity : %.4f",
        stage1_summary["judge_c_creativity"]["mean"],
    )
    logger.info(
        "Niche coverage     : %d/%d (%.1f%%)",
        stage1_summary["n_unique_niches"],
        len(InjectionType) * len(StealthLevel),
        stage1_summary["niche_coverage"] * 100,
    )

    if alignment_summary is not None:
        correlations = alignment_summary["correlations"]
        logger.info("Joined Stage-3 rows: %d", alignment_summary["n_joined"])
        logger.info("Missing Stage-3    : %d", missing_stage3)
        logger.info("Stage-3 ASR overall: %.1f%%", alignment_summary["stage3_asr_overall"] * 100)
        logger.info(
            "Reward corr (P/S) : %.4f / %.4f",
            correlations["pearson_stage1_reward_vs_stage3_reward"],
            correlations["spearman_stage1_reward_vs_stage3_reward"],
        )

    if args.dry_run:
        logger.info("--dry-run: exiting before writing artifacts.")
        return

    summary_path = output_dir / "stage1_eval_summary.json"
    round_metrics_path = output_dir / "round_metrics.jsonl"
    top_stage1_path = output_dir / "top_stage1_payloads.jsonl"

    _dump_json(summary_path, summary)
    _dump_jsonl(round_metrics_path, round_metrics)
    _dump_jsonl(top_stage1_path, top_rows)

    if joined_rows:
        joined_top_path = output_dir / "top_stage1_with_stage3.jsonl"
        joined_top_rows = sorted(
            joined_rows,
            key=lambda row: row["stage1_reward"],
            reverse=True,
        )[: int(config["top_k"])]
        _dump_jsonl(joined_top_path, joined_top_rows)
        logger.info("Joined top-k dump  : %s", joined_top_path)

    logger.info("Summary            : %s", summary_path)
    logger.info("Round metrics      : %s", round_metrics_path)
    logger.info("Top Stage-1 dump   : %s", top_stage1_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
