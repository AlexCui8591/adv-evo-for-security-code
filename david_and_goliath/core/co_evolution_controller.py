"""core/co_evolution_controller.py — David & Goliath 主控制器

编排 Red Team GRPO 训练 ↔ Blue Team 响应 ↔ Oracle 评判 的完整 co-evolution 循环。

生命周期:
  setup() → run() [N 轮 _run_round()] → finalize()

当前版本: Blue Team 保持静态 (不进化)，只有 Red Team 通过 GRPO 进化。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from core.types import OracleWeights, RoundRecord
from core.injection_engine import InjectionEngine, load_coding_tasks
from core.strategy_db import MAPElitesDB

logger = logging.getLogger(__name__)


class CoEvolutionController:
    """Co-Evolution 主控制器

    Args:
        config: 实验配置字典，包含所有超参数
    """

    def __init__(self, config: dict):
        self.config = config
        self.oracle_mode: str = config.get("oracle", {}).get("mode", "full")

        # 实验元信息
        self.experiment_id: str = config.get("experiment_id", "coevo_default")
        self.seed: int = config.get("seed", 42)
        self.total_rounds: int = config.get("total_rounds", 20)
        self.checkpoint_every: int = config.get("checkpoint_every", 5)
        self.output_dir = Path(config.get("output_dir", f"outputs/{self.experiment_id}"))

        # 组件引用 (setup() 中初始化)
        self.injection_engine: Optional[InjectionEngine] = None
        self.strategy_db: Optional[MAPElitesDB] = None
        self.oracle = None          # HybridOracle
        self.blue_team = None       # CodingAgent
        self.grpo_trainer = None    # GRPOTrainer

        # 训练记录
        self.round_records: list[RoundRecord] = []
        self.start_round: int = 1   # 支持从 checkpoint 恢复

    # ==============================================================
    # 初始化
    # ==============================================================

    def setup(self):
        """初始化所有组件"""
        logger.info("=" * 60)
        logger.info(f"Setting up experiment: {self.experiment_id}")
        logger.info("=" * 60)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 加载 coding tasks
        tasks_path = self.config.get(
            "coding_tasks_path", "david_and_goliath/data/coding_tasks/tasks.jsonl"
        )
        coding_tasks = load_coding_tasks(tasks_path)
        logger.info(f"Loaded {len(coding_tasks)} coding tasks")

        # 2. InjectionEngine
        self.injection_engine = InjectionEngine(
            coding_tasks=coding_tasks,
            seed=self.seed,
        )

        # 3. MAP-Elites 策略库
        self.strategy_db = MAPElitesDB(
            niche_capacity=self.config.get("niche_capacity", 5),
            seed=self.seed,
        )

        # 4. 三个 Judge → HybridOracle
        self.oracle = self._build_oracle()

        # 5. Blue Team (静态 coding agent)
        self.blue_team = self._build_blue_team()

        # 6. Red Team GRPO Trainer
        self.grpo_trainer = self._build_grpo_trainer()

        # 7. 初始化 GRPO 模型
        logger.info("Initializing Red Team policy model...")
        self.grpo_trainer.initialize()

        # 8. 尝试从 checkpoint 恢复
        self._load_checkpoint()

        logger.info(f"Setup complete. Will run rounds {self.start_round}..{self.total_rounds}")

    def _build_oracle(self):
        """构建 Hybrid Oracle (三 Judge 系统)"""
        from hybrid_oracle.judge_c import JudgeC
        from hybrid_oracle.oracle import HybridOracle, LightRewardOracle

        oracle_cfg = self.config.get("oracle", {})
        oracle_mode = oracle_cfg.get("mode", "full")
        if oracle_mode not in ("full", "payload_only"):
            raise ValueError(
                f"oracle.mode must be 'full' or 'payload_only', got {oracle_mode!r}"
            )

        judge_c = JudgeC(
            model=oracle_cfg.get("judge_model", "gpt-4o-mini"),
            api_key=oracle_cfg.get("api_key"),
            temperature=oracle_cfg.get("judge_temperature", 0.2),
        )

        if oracle_mode == "payload_only":
            oracle = LightRewardOracle(
                judge_c=judge_c,
                strategy_db=self.strategy_db,
                w_quality=oracle_cfg.get("w_quality", 0.20),
                w_stealth=oracle_cfg.get("w_stealth", 0.10),
                w_diversity=oracle_cfg.get("w_diversity", 0.10),
            )
            logger.info("LightRewardOracle initialized (payload_only mode)")
            return oracle

        from hybrid_oracle.judge_a import JudgeA
        from hybrid_oracle.judge_b import JudgeB

        judge_a = JudgeA(
            bandit_enabled=oracle_cfg.get("bandit_enabled", True),
            semgrep_enabled=oracle_cfg.get("semgrep_enabled", True),
            semgrep_rules=oracle_cfg.get("semgrep_rules", "p/security-audit"),
        )

        judge_b = JudgeB(
            model=oracle_cfg.get("judge_model", "gpt-4o-mini"),
            api_key=oracle_cfg.get("api_key"),
            temperature=oracle_cfg.get("judge_temperature", 0.1),
        )

        oracle = HybridOracle(
            judge_a=judge_a,
            judge_b=judge_b,
            judge_c=judge_c,
            strategy_db=self.strategy_db,
            w_vulnerability=oracle_cfg.get("w_vulnerability", 0.30),
            w_manipulation=oracle_cfg.get("w_manipulation", 0.30),
            w_quality=oracle_cfg.get("w_quality", 0.20),
            w_diversity=oracle_cfg.get("w_diversity", 0.10),
            w_stealth=oracle_cfg.get("w_stealth", 0.10),
            failure_quality_scale=oracle_cfg.get("failure_quality_scale", 0.15),
        )

        logger.info("Hybrid Oracle initialized (Judge A + B + C)")
        return oracle

    def _build_blue_team(self):
        """构建 Blue Team coding agent (当前版本: 静态)"""
        from blue_team.coding_agent import CodingAgent

        blue_cfg = self.config.get("blue_team", {})

        agent = CodingAgent(
            model=blue_cfg.get("model", "gpt-4o-mini"),
            api_key=blue_cfg.get("api_key"),
            base_url=blue_cfg.get("base_url"),
            temperature=blue_cfg.get("temperature", 0.2),
            max_turns=blue_cfg.get("max_turns", 6),
            max_reflexion=blue_cfg.get("max_reflexion", 2),
            use_tools=blue_cfg.get("use_tools", True),
        )

        logger.info(f"Blue Team initialized: {blue_cfg.get('model_name', 'gpt-4o-mini')} (static)")
        return agent

    def _build_grpo_trainer(self):
        """构建 Red Team GRPO Trainer"""
        from red_team.grpo_trainer import GRPOTrainer
        from red_team.prompt_builder import RedPromptBuilder

        red_cfg = dict(self.config.get("red_team", {}))
        red_cfg["oracle_mode"] = self.oracle_mode
        red_cfg["output_dir"] = str(self.output_dir)

        prompt_builder = RedPromptBuilder(
            config=red_cfg.get("prompt_builder", {}),
        )

        trainer = GRPOTrainer(
            model_name=red_cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
            lora_path=red_cfg.get("lora_path", ""),
            oracle=self.oracle,
            blue_team=self.blue_team,
            injection_engine=self.injection_engine,
            strategy_db=self.strategy_db,
            prompt_builder=prompt_builder,
            config=red_cfg,
        )

        logger.info(f"Red Team GRPO Trainer initialized: {red_cfg.get('model_name')}")
        return trainer

    # ==============================================================
    # 主循环
    # ==============================================================

    def run(self):
        """主训练循环"""
        logger.info("=" * 60)
        logger.info("Starting co-evolution training")
        logger.info("=" * 60)

        for round_num in range(self.start_round, self.total_rounds + 1):
            record = self._run_round(round_num)
            self.round_records.append(record)

            # 定期 checkpoint
            if round_num % self.checkpoint_every == 0:
                self._save_checkpoint(round_num)

        self.finalize()

    def _run_round(self, round_num: int) -> RoundRecord:
        """执行一轮 co-evolution

        Step 1: Red Team GRPO 训练 (包含完整 episode 循环)
        Step 2: Oracle 权重自适应调整
        Step 3: 日志 & 记录
        """
        round_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_num}/{self.total_rounds}")
        logger.info(f"{'='*60}")

        # ---- Step 1: Red Team GRPO 训练 ----
        logger.info(f"[Step 1] Red Team GRPO training...")

        # 上一轮的 Blue Team 行为摘要 (供 RedPromptBuilder 参考)
        blue_summary = self._get_blue_behavior_summary()

        top_payloads, stats = self.grpo_trainer.train_round(
            round_num=round_num,
            blue_behavior_summary=blue_summary,
        )

        logger.info(
            f"  GRPO done: ASR={stats['attack_success_rate']:.1%}, "
            f"avg_reward={stats['grpo_avg_reward']:.3f}, "
            f"episodes={stats['grpo_num_episodes']}"
        )

        # ---- Step 2: Oracle 权重自适应 ----
        logger.info(f"[Step 2] Oracle curriculum adjustment...")

        asr = stats["attack_success_rate"]
        self.oracle.update_weights_from_feedback(asr, round_num)

        # ---- Step 3: 策略库快照 ----
        self.strategy_db.snapshot(round_num)
        db_stats = self.strategy_db.niche_stats()

        logger.info(
            f"  Strategy DB: coverage={db_stats['strategy_db/coverage']:.0%}, "
            f"total={db_stats['strategy_db/total_payloads']}"
        )

        # ---- Step 4: 构建 RoundRecord ----
        round_time = time.time() - round_start

        record = RoundRecord(
            round=round_num,
            config_id=self.experiment_id,
            seed=self.seed,
            # GRPO 统计
            grpo_num_episodes=stats["grpo_num_episodes"],
            grpo_train_steps=stats["grpo_train_steps"],
            grpo_avg_reward=stats["grpo_avg_reward"],
            grpo_reward_std=stats["grpo_reward_std"],
            grpo_train_time_s=stats["grpo_train_time_s"],
            # 攻击统计
            attack_success_rate=stats["attack_success_rate"],
            judge_a_trigger_rate=stats["judge_a_trigger_rate"],
            judge_b_trigger_rate=stats["judge_b_trigger_rate"],
            avg_payload_quality=stats["avg_payload_quality"],
            # 多样性
            red_diversity_coverage=stats["red_diversity_coverage"],
            # Oracle 权重快照
            oracle_weights=OracleWeights(
                w_vulnerability=self.oracle.w_vulnerability,
                w_manipulation=self.oracle.w_manipulation,
                w_quality=self.oracle.w_quality,
                w_diversity=self.oracle.w_diversity,
                w_stealth=self.oracle.w_stealth,
                failure_quality_scale=self.oracle.failure_quality_scale,
            ),
            # 总体
            round_wall_time_s=round_time,
        )

        # ---- 日志输出 ----
        self._log_round_summary(record)

        return record

    # ==============================================================
    # Checkpoint
    # ==============================================================

    def _save_checkpoint(self, round_num: int):
        """保存检查点"""
        ckpt_dir = self.output_dir / "checkpoints" / f"round_{round_num:03d}"
        logger.info(f"Saving checkpoint to {ckpt_dir}")

        # Red Team model
        self.grpo_trainer.save_checkpoint(ckpt_dir / "red_team")

        # 策略库
        self.strategy_db.save(ckpt_dir / "strategy_db.json")

        # 训练记录
        records_data = [r.to_dict() for r in self.round_records]
        with open(ckpt_dir / "round_records.json", "w", encoding="utf-8") as f:
            json.dump(records_data, f, ensure_ascii=False, indent=2)

        # 元信息 (用于恢复)
        meta = {
            "round": round_num,
            "experiment_id": self.experiment_id,
            "total_rounds": self.total_rounds,
        }
        with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Checkpoint saved: round {round_num}")

    def _load_checkpoint(self):
        """尝试从最新 checkpoint 恢复"""
        ckpt_base = self.output_dir / "checkpoints"
        if not ckpt_base.exists():
            return

        # 找最新的 checkpoint
        ckpt_dirs = sorted(ckpt_base.glob("round_*"))
        if not ckpt_dirs:
            return

        latest = ckpt_dirs[-1]
        meta_path = latest / "meta.json"
        if not meta_path.exists():
            return

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        resumed_round = meta["round"]
        logger.info(f"Found checkpoint at round {resumed_round}, resuming...")

        # 恢复策略库
        db_path = latest / "strategy_db.json"
        if db_path.exists():
            self.strategy_db.load(db_path)

        # 恢复训练记录
        records_path = latest / "round_records.json"
        if records_path.exists():
            with open(records_path, "r", encoding="utf-8") as f:
                records_data = json.load(f)
            # 简单恢复 — 完整反序列化需要 RoundRecord.from_dict()
            logger.info(f"  Restored {len(records_data)} round records")

        # 下一轮从 resumed_round + 1 开始
        self.start_round = resumed_round + 1
        logger.info(f"  Will resume from round {self.start_round}")

    # ==============================================================
    # 收尾
    # ==============================================================

    def finalize(self):
        """训练结束后的收尾工作"""
        logger.info("\n" + "=" * 60)
        logger.info("Co-evolution training complete!")
        logger.info("=" * 60)

        # 保存最终 checkpoint
        self._save_checkpoint(self.total_rounds)

        # 保存完整结果
        results_path = self.output_dir / "results.json"
        results = {
            "experiment_id": self.experiment_id,
            "total_rounds": self.total_rounds,
            "seed": self.seed,
            "rounds": [r.to_dict() for r in self.round_records],
            "final_strategy_db": self.strategy_db.niche_stats(),
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {results_path}")

        # 打印实验摘要
        self._print_experiment_summary()

    # ==============================================================
    # 辅助方法
    # ==============================================================

    def _get_blue_behavior_summary(self) -> str:
        """生成 Blue Team 行为摘要 (供 RedPromptBuilder 参考)

        告诉 Red Team: 上一轮 Blue Team 的检出率、常用工具等，
        引导 Red Team 针对性地进化攻击策略。
        """
        if not self.round_records:
            return "No previous data. Blue Team behavior is unknown."

        last = self.round_records[-1]
        return (
            f"In round {last.round}, Blue Team detected "
            f"{last.blue_detection_rate:.0%} of attacks. "
            f"Attack success rate was {last.attack_success_rate:.0%}. "
            f"Judge A triggered {last.judge_a_trigger_rate:.0%}, "
            f"Judge B triggered {last.judge_b_trigger_rate:.0%}."
        )

    def _log_round_summary(self, record: RoundRecord):
        """打印单轮摘要"""
        logger.info(f"\n--- Round {record.round} Summary ---")
        logger.info(f"  ASR:            {record.attack_success_rate:.1%}")
        logger.info(f"  Avg Reward:     {record.grpo_avg_reward:.3f} ± {record.grpo_reward_std:.3f}")
        logger.info(f"  Judge A Rate:   {record.judge_a_trigger_rate:.1%}")
        logger.info(f"  Judge B Rate:   {record.judge_b_trigger_rate:.1%}")
        logger.info(f"  Payload Quality:{record.avg_payload_quality:.3f}")
        logger.info(f"  DB Coverage:    {record.red_diversity_coverage:.0%}")
        logger.info(f"  Round Time:     {record.round_wall_time_s:.1f}s")
        logger.info(
            f"  Oracle Weights: "
            f"vuln={record.oracle_weights.w_vulnerability:.2f} "
            f"manip={record.oracle_weights.w_manipulation:.2f} "
            f"qual={record.oracle_weights.w_quality:.2f} "
            f"fail_scale={record.oracle_weights.failure_quality_scale:.2f}"
        )

    def _print_experiment_summary(self):
        """打印完整实验摘要"""
        if not self.round_records:
            return

        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)

        # ASR 趋势
        asrs = [r.attack_success_rate for r in self.round_records]
        logger.info(f"  ASR trend:   {asrs[0]:.1%} → {asrs[-1]:.1%}")
        logger.info(f"  ASR max:     {max(asrs):.1%} (round {asrs.index(max(asrs)) + self.start_round})")

        # Reward 趋势
        rewards = [r.grpo_avg_reward for r in self.round_records]
        logger.info(f"  Reward trend:{rewards[0]:.3f} → {rewards[-1]:.3f}")

        # 多样性
        coverages = [r.red_diversity_coverage for r in self.round_records]
        logger.info(f"  Coverage:    {coverages[0]:.0%} → {coverages[-1]:.0%}")

        # 策略库最终状态
        logger.info(f"\n  Final Strategy DB:")
        logger.info(f"\n{self.strategy_db.display_grid()}")

        logger.info("=" * 60)
