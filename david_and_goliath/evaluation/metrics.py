"""evaluation/metrics.py — Aggregate metrics from EpisodeResult lists.

All functions are pure Python with no external dependencies.
They are used by CrossEvaluator, OODEvaluator, and the eval scripts.
"""

from __future__ import annotations

import math
from typing import Any


# ============================================================
# Per-metric helpers
# ============================================================

def compute_asr(episodes: list) -> float:
    """Attack Success Rate: fraction of episodes where attack succeeded."""
    if not episodes:
        return 0.0
    return sum(1 for e in episodes if e.attack_success) / len(episodes)


def compute_judge_trigger_rates(episodes: list) -> dict[str, float]:
    """Fraction of episodes where Judge A (vuln) and Judge B (manip) fired."""
    if not episodes:
        return {"judge_a": 0.0, "judge_b": 0.0}
    n = len(episodes)
    judge_a = sum(
        1 for e in episodes if e.judge_a and e.judge_a.has_vulnerability
    ) / n
    judge_b = sum(
        1 for e in episodes if e.judge_b and e.judge_b.is_manipulated
    ) / n
    return {"judge_a": judge_a, "judge_b": judge_b}


def compute_reward_stats(episodes: list) -> dict[str, float]:
    """Mean / std / min / max of total_reward across episodes."""
    if not episodes:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    rewards = [e.total_reward for e in episodes]
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(rewards),
        "max": max(rewards),
    }


def compute_blue_detection_rate(episodes: list) -> float:
    """Fraction of injected episodes where Blue Team's memory_retrieval flagged suspicious content."""
    if not episodes:
        return 0.0
    n = sum(
        1 for e in episodes
        if e.blue_response and e.blue_response.detected_suspicious
    )
    return n / len(episodes)


def compute_blue_verification_pass_rate(episodes: list) -> float:
    if not episodes:
        return 0.0
    hits = sum(
        1
        for e in episodes
        if e.blue_response and bool((e.blue_response.verification or {}).get("passed", False))
    )
    return hits / len(episodes)


def compute_blue_tests_pass_rate(episodes: list) -> float:
    if not episodes:
        return 0.0
    hits = sum(
        1
        for e in episodes
        if e.blue_response and bool((e.blue_response.verification or {}).get("tests_passed", False))
    )
    return hits / len(episodes)


def compute_blue_static_clean_rate(episodes: list) -> float:
    if not episodes:
        return 0.0
    hits = sum(
        1
        for e in episodes
        if e.blue_response and bool((e.blue_response.verification or {}).get("static_clean", False))
    )
    return hits / len(episodes)


def compute_blue_tool_call_rate(episodes: list) -> float:
    if not episodes:
        return 0.0
    hits = sum(
        1
        for e in episodes
        if e.blue_response and e.blue_response.num_tool_calls > 0
    )
    return hits / len(episodes)


def compute_blue_avg_num_tool_calls(episodes: list) -> float:
    if not episodes:
        return 0.0
    return sum(
        e.blue_response.num_tool_calls if e.blue_response else 0
        for e in episodes
    ) / len(episodes)


def compute_blue_memory_static_hit_rate(episodes: list) -> float:
    if not episodes:
        return 0.0
    hits = sum(
        1
        for e in episodes
        if e.blue_response
        and int((e.blue_response.memory_scan or {}).get("static_match_count", 0) or 0) > 0
    )
    return hits / len(episodes)


def compute_blue_memory_dynamic_hit_rate(episodes: list) -> float:
    if not episodes:
        return 0.0
    hits = sum(
        1
        for e in episodes
        if e.blue_response
        and int((e.blue_response.memory_scan or {}).get("dynamic_match_count", 0) or 0) > 0
    )
    return hits / len(episodes)


def compute_blue_high_risk_memory_hit_rate(episodes: list) -> float:
    if not episodes:
        return 0.0
    hits = sum(
        1
        for e in episodes
        if e.blue_response
        and int((e.blue_response.memory_scan or {}).get("dynamic_high_risk_count", 0) or 0) > 0
    )
    return hits / len(episodes)


def compute_blue_defense_context_applied_rate(episodes: list) -> float:
    if not episodes:
        return 0.0
    hits = sum(
        1
        for e in episodes
        if e.blue_response and e.blue_response.defense_context_applied
    )
    return hits / len(episodes)


def compute_blue_avg_retrieved_memory_count(episodes: list) -> float:
    if not episodes:
        return 0.0
    return sum(
        e.blue_response.retrieved_memory_count if e.blue_response else 0
        for e in episodes
    ) / len(episodes)


def compute_avg_payload_quality(episodes: list) -> float:
    """Mean Judge C payload_quality_score across episodes."""
    qualities = [
        e.judge_c.payload_quality_score
        for e in episodes
        if e.judge_c is not None
    ]
    return sum(qualities) / len(qualities) if qualities else 0.0


def compute_avg_stealth_score(episodes: list) -> float:
    """Mean Judge C stealth_score across episodes."""
    scores = [
        e.judge_c.stealth_score
        for e in episodes
        if e.judge_c is not None
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_avg_latency_ms(episodes: list) -> float:
    """Mean total wall time per episode (Blue + all Judges)."""
    if not episodes:
        return 0.0
    return sum(e.total_wall_time_ms for e in episodes) / len(episodes)


# ============================================================
# Aggregate summary dict
# ============================================================

def aggregate_metrics(episodes: list, prefix: str = "eval") -> dict[str, Any]:
    """Compute all metrics and return a flat dict with *prefix/* keys.

    Suitable for WandB logging, JSON export, or console printing.
    """
    if not episodes:
        return {f"{prefix}/n_episodes": 0}

    asr = compute_asr(episodes)
    judge_rates = compute_judge_trigger_rates(episodes)
    reward = compute_reward_stats(episodes)
    detection_rate = compute_blue_detection_rate(episodes)
    verification_pass_rate = compute_blue_verification_pass_rate(episodes)
    tests_pass_rate = compute_blue_tests_pass_rate(episodes)
    static_clean_rate = compute_blue_static_clean_rate(episodes)
    tool_call_rate = compute_blue_tool_call_rate(episodes)
    avg_num_tool_calls = compute_blue_avg_num_tool_calls(episodes)
    static_hit_rate = compute_blue_memory_static_hit_rate(episodes)
    dynamic_hit_rate = compute_blue_memory_dynamic_hit_rate(episodes)
    high_risk_hit_rate = compute_blue_high_risk_memory_hit_rate(episodes)
    defense_context_rate = compute_blue_defense_context_applied_rate(episodes)
    avg_retrieved_memory_count = compute_blue_avg_retrieved_memory_count(episodes)
    quality = compute_avg_payload_quality(episodes)
    stealth = compute_avg_stealth_score(episodes)
    latency = compute_avg_latency_ms(episodes)

    return {
        f"{prefix}/n_episodes": len(episodes),
        f"{prefix}/asr": asr,
        f"{prefix}/judge_a_trigger_rate": judge_rates["judge_a"],
        f"{prefix}/judge_b_trigger_rate": judge_rates["judge_b"],
        f"{prefix}/avg_reward": reward["mean"],
        f"{prefix}/reward_std": reward["std"],
        f"{prefix}/min_reward": reward["min"],
        f"{prefix}/max_reward": reward["max"],
        f"{prefix}/blue_detection_rate": detection_rate,
        f"{prefix}/blue_verification_pass_rate": verification_pass_rate,
        f"{prefix}/blue_tests_pass_rate": tests_pass_rate,
        f"{prefix}/blue_static_clean_rate": static_clean_rate,
        f"{prefix}/blue_tool_call_rate": tool_call_rate,
        f"{prefix}/blue_avg_num_tool_calls": avg_num_tool_calls,
        f"{prefix}/blue_memory_static_hit_rate": static_hit_rate,
        f"{prefix}/blue_memory_dynamic_hit_rate": dynamic_hit_rate,
        f"{prefix}/blue_high_risk_memory_hit_rate": high_risk_hit_rate,
        f"{prefix}/blue_defense_context_applied_rate": defense_context_rate,
        f"{prefix}/blue_avg_retrieved_memory_count": avg_retrieved_memory_count,
        f"{prefix}/avg_payload_quality": quality,
        f"{prefix}/avg_stealth_score": stealth,
        f"{prefix}/avg_latency_ms": latency,
    }


def compare_metrics(baseline: dict, target: dict) -> dict[str, float]:
    """Compute absolute and percentage delta between two metric dicts.

    Strips any prefix up to the first '/' before comparing keys.
    """
    def _strip_prefix(d: dict) -> dict:
        return {k.split("/", 1)[-1]: v for k, v in d.items()}

    base = _strip_prefix(baseline)
    tgt = _strip_prefix(target)

    delta: dict[str, float] = {}
    for key in sorted(set(list(base.keys()) + list(tgt.keys()))):
        b = base.get(key, 0.0)
        t = tgt.get(key, 0.0)
        if isinstance(b, (int, float)) and isinstance(t, (int, float)):
            delta[f"delta/{key}"] = float(t) - float(b)
            if b != 0:
                delta[f"pct/{key}"] = (float(t) - float(b)) / abs(float(b))
    return delta


# ============================================================
# Pretty printer
# ============================================================

def print_metrics(metrics: dict, title: str = "Metrics") -> None:
    """Print a metric dict to stdout in a readable table."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, bool):
            print(f"  {k:<40}  {v}")
        elif isinstance(v, float):
            if "rate" in k or "/asr" in k or k.endswith("asr"):
                print(f"  {k:<40}  {v:.1%}")
            else:
                print(f"  {k:<40}  {v:.4f}")
        else:
            print(f"  {k:<40}  {v}")
    print(f"{'=' * 50}\n")
