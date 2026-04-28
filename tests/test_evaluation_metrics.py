from __future__ import annotations

import unittest

from david_and_goliath.core.types import BlueTeamResponse, EpisodeResult
from david_and_goliath.evaluation.metrics import aggregate_metrics


class EvaluationMetricsTest(unittest.TestCase):
    def test_aggregate_metrics_includes_blue_runtime_fields(self) -> None:
        episodes = [
            EpisodeResult(
                attack_success=True,
                total_reward=1.5,
                blue_response=BlueTeamResponse(
                    detected_suspicious=True,
                    verification={
                        "passed": True,
                        "tests_passed": True,
                        "static_clean": True,
                    },
                    memory_scan={
                        "static_match_count": 2,
                        "dynamic_match_count": 1,
                        "dynamic_high_risk_count": 1,
                    },
                    retrieved_memories=[{"memory_key": "m1"}],
                    defense_context_applied=True,
                    tool_calls=[],
                ),
            ),
            EpisodeResult(
                attack_success=False,
                total_reward=0.5,
                blue_response=BlueTeamResponse(
                    detected_suspicious=False,
                    verification={
                        "passed": False,
                        "tests_passed": False,
                        "static_clean": False,
                    },
                    memory_scan={
                        "static_match_count": 0,
                        "dynamic_match_count": 0,
                        "dynamic_high_risk_count": 0,
                    },
                    retrieved_memories=[],
                    defense_context_applied=False,
                    tool_calls=[],
                ),
            ),
        ]

        metrics = aggregate_metrics(episodes, prefix="eval")

        self.assertAlmostEqual(metrics["eval/asr"], 0.5)
        self.assertAlmostEqual(metrics["eval/blue_detection_rate"], 0.5)
        self.assertAlmostEqual(metrics["eval/blue_verification_pass_rate"], 0.5)
        self.assertAlmostEqual(metrics["eval/blue_tests_pass_rate"], 0.5)
        self.assertAlmostEqual(metrics["eval/blue_static_clean_rate"], 0.5)
        self.assertAlmostEqual(metrics["eval/blue_tool_call_rate"], 0.0)
        self.assertAlmostEqual(metrics["eval/blue_avg_num_tool_calls"], 0.0)
        self.assertAlmostEqual(metrics["eval/blue_memory_static_hit_rate"], 0.5)
        self.assertAlmostEqual(metrics["eval/blue_memory_dynamic_hit_rate"], 0.5)
        self.assertAlmostEqual(metrics["eval/blue_high_risk_memory_hit_rate"], 0.5)
        self.assertAlmostEqual(metrics["eval/blue_defense_context_applied_rate"], 0.5)
        self.assertAlmostEqual(metrics["eval/blue_avg_retrieved_memory_count"], 0.5)


if __name__ == "__main__":
    unittest.main()
