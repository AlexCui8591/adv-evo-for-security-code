"""Result aggregation and summary statistics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _compute_group_stats(
    attacker_id: str,
    scenario: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(items)
    successes = sum(1 for it in items if it["final_label"] == "success")
    refusals = sum(1 for it in items if it["refusal_label"])
    success_token_total = sum(
        it["attacker_prompt_tokens"] for it in items if it["final_label"] == "success"
    )
    total_prompt_tokens = sum(it["attacker_prompt_tokens"] for it in items)
    return {
        "attacker_id": attacker_id,
        "scenario": scenario,
        "cases": total,
        "attack_success_rate": round(successes / total, 4) if total else 0.0,
        "refusal_rate": round(refusals / total, 4) if total else 0.0,
        "avg_success_prompt_tokens": (
            round(success_token_total / successes, 2) if successes else 0.0
        ),
        "success_per_100_tokens": (
            round(100.0 * successes / total_prompt_tokens, 4)
            if total_prompt_tokens
            else 0.0
        ),
    }


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Per-scenario breakdown
    by_scenario: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[(row["attacker_id"], row["scenario"])].append(row)

    summary: list[dict[str, Any]] = []
    for (attacker_id, scenario), items in sorted(by_scenario.items()):
        summary.append(_compute_group_stats(attacker_id, scenario, items))

    # Overall per-attacker
    by_attacker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_attacker[row["attacker_id"]].append(row)

    for attacker_id, items in sorted(by_attacker.items()):
        summary.append(_compute_group_stats(attacker_id, "overall", items))

    return summary
