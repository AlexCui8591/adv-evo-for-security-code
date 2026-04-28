"""evaluation/ood_evaluator.py — Out-of-Distribution generalisation test.

Compares Red Team attack success rate on:
  - In-Distribution (ID) tasks: the same coding task pool used during training.
  - Out-of-Distribution (OOD) tasks: a held-out task set from a different source.

A high ID→OOD transfer ratio indicates the Red Team learned genuinely
transferable injection strategies rather than task-specific patterns.

Usage
-----
  ood_eval = OODEvaluator(cross_evaluator, ood_tasks)
  results  = ood_eval.compare(payloads, n_tasks_per_payload=3)
  print(results["transfer_ratio"])   # OOD-ASR / ID-ASR
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.types import CodingTask, EpisodeResult, Payload
from evaluation.cross_evaluator import CrossEvaluator, load_payloads_from_checkpoint
from evaluation.metrics import aggregate_metrics, compare_metrics

logger = logging.getLogger(__name__)


class OODEvaluator:
    """Measure generalisation gap between ID and OOD coding tasks.

    Args:
        cross_evaluator: A configured CrossEvaluator (oracle + blue + injection_engine).
        ood_tasks:       Held-out coding tasks NOT seen during training.
    """

    def __init__(
        self,
        cross_evaluator: CrossEvaluator,
        ood_tasks: list[CodingTask],
    ) -> None:
        self.cross_evaluator = cross_evaluator
        self.ood_tasks = ood_tasks

        if not ood_tasks:
            raise ValueError("ood_tasks is empty — provide at least one OOD task.")

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def compare(
        self,
        payloads: list[Payload],
        n_tasks_per_payload: int = 3,
    ) -> dict[str, Any]:
        """Run evaluation on both ID and OOD task pools.

        Args:
            payloads:            Payloads to test (from a checkpoint or elsewhere).
            n_tasks_per_payload: Tasks sampled per payload for each split.

        Returns:
            dict with keys:
              id_metrics     — aggregate_metrics() output for ID split (prefix="id")
              ood_metrics    — aggregate_metrics() output for OOD split (prefix="ood")
              delta          — compare_metrics() delta/pct entries
              transfer_ratio — OOD-ASR / ID-ASR (0 if ID-ASR == 0)
              id_episodes    — raw EpisodeResult list for ID
              ood_episodes   — raw EpisodeResult list for OOD
        """
        logger.info("OOD Evaluator: running ID split (%d payloads)...", len(payloads))
        id_results: list[EpisodeResult] = self.cross_evaluator.run_episodes(
            payloads=payloads,
            tasks=None,            # use injection_engine's own (training) tasks
            n_tasks_per_payload=n_tasks_per_payload,
            round_label=0,
        )

        logger.info("OOD Evaluator: running OOD split (%d OOD tasks)...", len(self.ood_tasks))
        ood_results: list[EpisodeResult] = self.cross_evaluator.run_episodes(
            payloads=payloads,
            tasks=self.ood_tasks,
            n_tasks_per_payload=n_tasks_per_payload,
            round_label=0,
        )

        id_metrics = aggregate_metrics(id_results, prefix="id")
        ood_metrics = aggregate_metrics(ood_results, prefix="ood")
        delta = compare_metrics(id_metrics, ood_metrics)

        id_asr = id_metrics.get("id/asr", 0.0)
        ood_asr = ood_metrics.get("ood/asr", 0.0)
        transfer_ratio = ood_asr / id_asr if id_asr > 0 else 0.0

        logger.info(
            "OOD eval complete — ID-ASR=%.1f%%  OOD-ASR=%.1f%%  transfer=%.2f",
            id_asr * 100, ood_asr * 100, transfer_ratio,
        )

        return {
            "id_metrics": id_metrics,
            "ood_metrics": ood_metrics,
            "delta": delta,
            "transfer_ratio": transfer_ratio,
            "id_episodes": len(id_results),
            "ood_episodes": len(ood_results),
        }

    def compare_from_checkpoint(
        self,
        ckpt_dir: str | Path,
        n_tasks_per_payload: int = 3,
    ) -> dict[str, Any]:
        """Load payloads from a checkpoint and run ID vs OOD comparison."""
        payloads = load_payloads_from_checkpoint(ckpt_dir)
        return self.compare(payloads, n_tasks_per_payload=n_tasks_per_payload)
