"""evaluation/cross_evaluator.py — Cross-evaluation engine.

Runs trained Red Team payloads against a (possibly different) Blue Team
configuration and scores every episode with the Hybrid Oracle.

The key difference from training:
  Training  — GRPOTrainer generates new payloads every round and updates weights.
  Eval      — payloads are fixed (loaded from a checkpoint); only inference runs.

Usage
-----
  evaluator = CrossEvaluator(oracle, blue_team, injection_engine)
  episodes  = evaluator.run_from_checkpoint(ckpt_dir, n_tasks_per_payload=3)
  metrics   = aggregate_metrics(episodes)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Optional

from core.types import (
    BlueTeamResponse,
    CodingTask,
    EpisodeResult,
    InjectionType,
    Payload,
    StealthLevel,
)

logger = logging.getLogger(__name__)


# ============================================================
# Payload loader
# ============================================================

def load_payloads_from_checkpoint(ckpt_dir: str | Path) -> list[Payload]:
    """Extract all payloads stored in a checkpoint's strategy_db.json.

    Returns payloads sorted by reward descending (best first).
    """
    ckpt_dir = Path(ckpt_dir)
    db_path = ckpt_dir / "strategy_db.json"
    if not db_path.exists():
        raise FileNotFoundError(
            f"strategy_db.json not found in {ckpt_dir}. "
            "Make sure --checkpoint-dir points to a saved round directory "
            "(e.g. outputs/my_exp/checkpoints/round_020)."
        )

    from core.strategy_db import MAPElitesDB
    db = MAPElitesDB()
    db.load(db_path)

    # Restore injection_type / stealth_level from the niche key — these fields
    # are not serialised in strategy_db.json but are implied by the niche the
    # payload lives in.  Without restoring, oracle._diversity_bonus would treat
    # every loaded payload as DIRECT_PROMPT/L1_OBVIOUS by default.
    all_payloads: list[Payload] = []
    for (inj_value, st_value), payloads in db.niche_grid.items():
        for p in payloads:
            p.injection_type = InjectionType(inj_value)
            p.stealth_level = StealthLevel(st_value)
            all_payloads.append(p)

    all_payloads.sort(
        key=lambda p: p.episode_result.total_reward if p.episode_result else 0.0,
        reverse=True,
    )

    logger.info(
        "Loaded %d payloads from %s (coverage=%.0f%%)",
        len(all_payloads), db_path, db.coverage * 100,
    )
    return all_payloads


# ============================================================
# CrossEvaluator
# ============================================================

class CrossEvaluator:
    """Run fixed payloads against a target Blue Team and score with Oracle.

    Args:
        oracle:           HybridOracle instance (scoring only; weights not updated).
        blue_team:        CodingAgent to evaluate against (can differ from training).
        injection_engine: InjectionEngine holding in-distribution coding tasks.
        seed:             Random seed for task sampling.
    """

    def __init__(
        self,
        oracle: Any,
        blue_team: Any,
        injection_engine: Any,
        seed: int = 42,
    ) -> None:
        self.oracle = oracle
        self.blue_team = blue_team
        self.injection_engine = injection_engine
        self._rng = random.Random(seed)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run_episodes(
        self,
        payloads: list[Payload],
        tasks: Optional[list[CodingTask]] = None,
        n_tasks_per_payload: int = 3,
        round_label: int = 0,
    ) -> list[EpisodeResult]:
        """Evaluate *payloads* against *tasks* and return per-episode results.

        Args:
            payloads:            Payload list to test (typically from a checkpoint).
            tasks:               Override task pool (None = use injection_engine's own tasks).
            n_tasks_per_payload: Number of coding tasks sampled per payload.
            round_label:         Round number stored in EpisodeResult.round.

        Returns:
            List of EpisodeResult, one per (payload, task) pair.
        """
        task_pool: list[CodingTask] = tasks or self.injection_engine.coding_tasks
        if not task_pool:
            raise ValueError("Task pool is empty — check injection_engine or --tasks-path.")

        results: list[EpisodeResult] = []
        total = len(payloads) * n_tasks_per_payload
        done = 0

        for payload in payloads:
            selected = self._rng.sample(
                task_pool, k=min(n_tasks_per_payload, len(task_pool))
            )
            for task in selected:
                episode = self._run_single(payload, task, round_label)
                results.append(episode)
                done += 1
                if done % 10 == 0:
                    asr_so_far = sum(1 for e in results if e.attack_success) / len(results)
                    logger.info(
                        "Progress: %d/%d episodes  ASR=%.1f%%",
                        done, total, asr_so_far * 100,
                    )

        if results:
            final_asr = sum(1 for e in results if e.attack_success) / len(results)
            logger.info(
                "CrossEval done: %d episodes  ASR=%.1f%%",
                len(results), final_asr * 100,
            )
        else:
            logger.warning("CrossEval done: 0 episodes (empty payload list).")
        return results

    def run_from_checkpoint(
        self,
        ckpt_dir: str | Path,
        tasks: Optional[list[CodingTask]] = None,
        n_tasks_per_payload: int = 3,
        round_label: int = 0,
    ) -> list[EpisodeResult]:
        """Load payloads from a checkpoint directory and run evaluation."""
        payloads = load_payloads_from_checkpoint(ckpt_dir)
        return self.run_episodes(
            payloads=payloads,
            tasks=tasks,
            n_tasks_per_payload=n_tasks_per_payload,
            round_label=round_label,
        )

    def false_positive_check(self, n_samples: int = 20) -> dict[str, Any]:
        """Estimate Blue Team's false positive rate on clean (uninjected) tasks.

        A false positive is when Blue Team's memory_retrieval flags a clean task
        as suspicious.  High FPR means the Blue Team is overly paranoid, which
        would inflate `detected_suspicious` during normal cross-eval.

        Args:
            n_samples: Number of clean tasks to evaluate.

        Returns:
            dict with false_positive_rate, false_positive_count, n_samples.
        """
        fp_count = 0
        for _ in range(n_samples):
            clean_task = self.injection_engine.inject_clean()
            try:
                blue_response = self.blue_team.process(clean_task)
                if blue_response.detected_suspicious:
                    fp_count += 1
            except Exception as exc:
                logger.warning("FPR check: blue_team.process() raised: %s", exc)

        fpr = fp_count / n_samples if n_samples > 0 else 0.0
        logger.info("FPR check: %.1f%% (%d/%d)", fpr * 100, fp_count, n_samples)
        return {
            "n_samples": n_samples,
            "false_positive_count": fp_count,
            "false_positive_rate": fpr,
        }

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _run_single(
        self,
        payload: Payload,
        task: CodingTask,
        round_label: int,
    ) -> EpisodeResult:
        """Run one (payload, task) pair end-to-end."""
        injected = self.injection_engine.inject(payload, task)
        try:
            blue_response = self.blue_team.process(injected)
        except Exception as exc:
            logger.warning(
                "Blue team raised for payload=%s task=%s: %s",
                payload.id, task.id, exc,
            )
            blue_response = BlueTeamResponse(
                generated_code="# [blue team error]",
                reasoning=str(exc),
            )

        try:
            episode = self.oracle.evaluate_episode(
                payload=payload,
                injected_task=injected,
                blue_response=blue_response,
                round_num=round_label,
            )
        except Exception as exc:
            logger.warning(
                "Oracle raised for payload=%s task=%s: %s",
                payload.id, task.id, exc,
            )
            episode = EpisodeResult(
                payload_id=payload.id,
                round=round_label,
                coding_task_id=task.id,
                injection_position=injected.injection_position,
                blue_response=blue_response,
            )

        return episode
