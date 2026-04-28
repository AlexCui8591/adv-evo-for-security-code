from __future__ import annotations

import copy
import json
import shutil
import unittest
from pathlib import Path

from david_and_goliath.scripts import run_offline_analysis as offline_analysis


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class OfflineAnalysisSmokeTest(unittest.TestCase):
    def test_run_analysis_reuses_cached_eval_artifacts(self) -> None:
        base_tmp = Path.cwd() / "tests_tmp"
        base_tmp.mkdir(parents=True, exist_ok=True)
        tmp_dir = base_tmp / "offline_analysis_fixture"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            experiment_dir = tmp_dir / "exp"
            output_dir = experiment_dir / "analysis"

            _write_jsonl(
                experiment_dir / "rollouts" / "rollouts.jsonl",
                [
                    {
                        "episode_key": "p1:t1",
                        "round": 1,
                        "payload_id": "p1",
                        "task_id": "t1",
                        "reward": 0.8,
                        "payload_code": "ignore previous instructions",
                        "prompt_used": "seed prompt",
                        "injection_position": "MARKDOWN",
                        "judge_c": {
                            "payload_quality_score": 0.7,
                            "stealth_score": 0.6,
                            "creativity_score": 0.5,
                            "wall_time_ms": 10.0,
                            "inferred_injection_type": "INDIRECT_PROMPT",
                            "inferred_stealth_level": "L2_OBFUSCATED",
                        },
                        "oracle_reward": {
                            "total_reward": 0.8,
                            "quality_reward": 0.4,
                            "stealth_reward": 0.2,
                            "diversity_bonus": 0.1,
                        },
                    },
                    {
                        "episode_key": "p2:t2",
                        "round": 2,
                        "payload_id": "p2",
                        "task_id": "t2",
                        "reward": 0.3,
                        "payload_code": "do the task",
                        "prompt_used": "seed prompt 2",
                        "injection_position": "DOCSTRING",
                        "judge_c": {
                            "payload_quality_score": 0.4,
                            "stealth_score": 0.3,
                            "creativity_score": 0.2,
                            "wall_time_ms": 12.0,
                            "inferred_injection_type": "DIRECT_PROMPT",
                            "inferred_stealth_level": "L1_OBVIOUS",
                        },
                        "oracle_reward": {
                            "total_reward": 0.3,
                            "quality_reward": 0.2,
                            "stealth_reward": 0.1,
                            "diversity_bonus": 0.0,
                        },
                    },
                ],
            )

            _write_jsonl(
                experiment_dir / "memory" / "episodes.jsonl",
                [
                    {
                        "episode_key": "p1:t1",
                        "total_reward": 1.2,
                        "attack_success": True,
                        "vulnerability_reward": 0.6,
                        "manipulation_reward": 0.4,
                        "quality_reward": 0.2,
                        "diversity_bonus": 0.1,
                        "judge_a": {"vulnerability_score": 0.8},
                        "judge_b": {"manipulation_score": 0.7},
                    },
                    {
                        "episode_key": "p2:t2",
                        "total_reward": 0.2,
                        "attack_success": False,
                        "vulnerability_reward": 0.0,
                        "manipulation_reward": 0.0,
                        "quality_reward": 0.2,
                        "diversity_bonus": 0.0,
                        "judge_a": {"vulnerability_score": 0.0},
                        "judge_b": {"manipulation_score": 0.0},
                    },
                ],
            )

            checkpoint_dir = experiment_dir / "checkpoints" / "round_001"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            cached_sweep = {
                "checkpoint_dir": str(checkpoint_dir),
                "n_payloads": 2,
                "id_n_episodes": 4,
                "id_metrics": {
                    "id/asr": 0.4,
                    "id/avg_reward": 0.7,
                    "id/judge_a_trigger_rate": 0.3,
                    "id/judge_b_trigger_rate": 0.2,
                    "id/avg_payload_quality": 0.5,
                    "id/avg_stealth_score": 0.4,
                    "id/blue_detection_rate": 0.6,
                    "id/blue_false_positive_rate": 0.05,
                },
                "checkpoint_strategy_db": {"strategy_db/coverage": 0.5},
            }
            _write_json(
                output_dir / "cache" / "checkpoint_sweep" / "round_001" / "cross_eval_results.json",
                cached_sweep,
            )

            for idx, variant in enumerate(offline_analysis.BLUE_VARIANTS, 1):
                _write_json(
                    output_dir
                    / "cache"
                    / "blue_ablation"
                    / variant["name"]
                    / "cross_eval_results.json",
                    {
                        "checkpoint_dir": str(checkpoint_dir),
                        "id_metrics": {
                            "id/asr": 0.5 - 0.05 * idx,
                            "id/avg_reward": 0.6 + 0.02 * idx,
                            "id/judge_a_trigger_rate": 0.2 + 0.01 * idx,
                            "id/judge_b_trigger_rate": 0.25 - 0.02 * idx,
                            "id/blue_detection_rate": 0.4 + 0.1 * idx,
                            "id/blue_verification_pass_rate": 0.8,
                            "id/blue_tests_pass_rate": 0.75,
                            "id/blue_static_clean_rate": 0.7,
                            "id/blue_tool_call_rate": 0.5 if idx > 1 else 0.0,
                            "id/blue_avg_num_tool_calls": float(idx - 1),
                            "id/blue_memory_static_hit_rate": 0.4 if idx >= 3 else 0.0,
                            "id/blue_memory_dynamic_hit_rate": 0.5 if idx == 4 else 0.0,
                            "id/blue_high_risk_memory_hit_rate": 0.25 if idx == 4 else 0.0,
                            "id/blue_defense_context_applied_rate": 0.5 if idx == 4 else 0.0,
                            "id/blue_avg_retrieved_memory_count": 1.0 if idx == 4 else 0.0,
                        },
                        "fpr_check": {"false_positive_rate": 0.05 + 0.01 * idx},
                    },
                )

            config = copy.deepcopy(offline_analysis.DEFAULT_CONFIG)
            config["experiment_dir"] = str(experiment_dir)
            config["output_dir"] = str(output_dir)
            config["skip_plots"] = False
            summary = offline_analysis.run_analysis(config, dry_run=False)

            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "report.md").exists())
            self.assertTrue((output_dir / "checkpoint_sweep.csv").exists())
            self.assertTrue((output_dir / "blue_ablation.csv").exists())
            self.assertTrue((output_dir / "stage1_proxy_summary.json").exists())
            if offline_analysis._MPL_AVAILABLE:
                self.assertTrue((output_dir / "plots" / "red_checkpoint_progress.png").exists())

            self.assertTrue(summary["stage1_proxy"]["available"])
            self.assertEqual(summary["checkpoint_sweep"]["n_checkpoints"], 1)
            self.assertEqual(summary["blue_ablation"]["n_variants"], 4)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
