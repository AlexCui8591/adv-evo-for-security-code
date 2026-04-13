"""red_team/grpo_trainer.py — 红队 GRPO 训练器

核心模块: 使用 trl.GRPOTrainer 作为底层 RL 训练引擎, 通过自定义 reward_function
回调接入 Oracle pipeline (InjectionEngine → BlueTeam → HybridOracle)。

每轮训练流程 (train_round):
  1. 从 MAPElitesDB 采样 parents/inspirations → RedPromptBuilder 构建 prompt
  2. 红队模型生成一组 payload (trl.GRPOTrainer 内部处理生成 + GRPO 梯度更新)
  3. 自定义 reward_fn 对每个 payload 走完整 episode:
     payload → InjectionEngine → BlueTeam → Oracle → scalar reward
  4. trl.GRPOTrainer 根据 reward 计算组内相对优势, 更新策略
  5. 将高分 payload 存入 MAPElitesDB

依赖:
  - trl >= 0.12.0: GRPOTrainer, GRPOConfig
  - transformers: 模型和 tokenizer
  - peft: LoRA adapter

接口约束 (由 co_evolution_controller.py 定义):
  - __init__(...): 接收 oracle, blue_team, injection_engine, strategy_db, prompt_builder
  - initialize(): 加载模型
  - train_round(round_num, blue_behavior_summary) → (top_payloads, stats)
  - save_checkpoint(path): 保存 LoRA adapter
"""

from __future__ import annotations

import logging
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import torch

from core.types import (
    BlueTeamResponse,
    EpisodeResult,
    Payload,
)
from core.injection_engine import InjectionEngine
from core.strategy_db import MAPElitesDB
from red_team.prompt_builder import RedPromptBuilder
from red_team.models.lora_loader import LoRAModelLoader

logger = logging.getLogger(__name__)

# ---- 可选依赖: wandb / matplotlib ----
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")           # 无头模式, 服务器无 GUI 也能画图
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


class GRPOTrainer:
    """红队 GRPO 训练器

    使用 trl.GRPOTrainer 进行 Group Relative Policy Optimization,
    通过自定义 reward_function 将 Oracle pipeline 接入 RL 训练循环。

    Args:
        model_name: 红队基座模型名 (如 "Qwen/Qwen2.5-7B-Instruct")
        lora_path: 已有 LoRA checkpoint 路径 (空 = 从零开始)
        oracle: HybridOracle 实例 (三 Judge 评分)
        blue_team: CodingAgent 实例 (蓝队)
        injection_engine: InjectionEngine 实例 (载荷注入)
        strategy_db: MAPElitesDB 实例 (策略库)
        prompt_builder: RedPromptBuilder 实例 (prompt 构建)
        config: 训练相关配置字典
    """

    def __init__(
        self,
        model_name: str,
        lora_path: str,
        oracle,                              # hybrid_oracle.oracle.HybridOracle
        blue_team,                           # blue_team.coding_agent.CodingAgent
        injection_engine: InjectionEngine,
        strategy_db: MAPElitesDB,
        prompt_builder: RedPromptBuilder,
        config: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.lora_path = lora_path
        self.oracle = oracle
        self.blue_team = blue_team
        self.injection_engine = injection_engine
        self.strategy_db = strategy_db
        self.prompt_builder = prompt_builder
        self.config = config or {}

        # ---- 从 config 提取超参数 ----

        # GRPO 训练超参
        self.group_size: int = self.config.get("group_size", 8)
        self.num_train_steps: int = self.config.get("num_train_steps", 10)
        self.learning_rate: float = self.config.get("learning_rate", 5e-6)
        self.kl_beta: float = self.config.get("kl_beta", 0.05)
        self.clip_eps: float = self.config.get("clip_eps", 0.2)

        # 生成超参
        self.max_gen_length: int = self.config.get("max_gen_length", 512)
        self.temperature: float = self.config.get("temperature", 0.8)
        self.top_p: float = self.config.get("top_p", 0.95)

        # LoRA 超参
        self.lora_r: int = self.config.get("lora_r", 16)
        self.lora_alpha: int = self.config.get("lora_alpha", 32)
        self.lora_dropout: float = self.config.get("lora_dropout", 0.05)
        self.quantization: Optional[str] = self.config.get("quantization", None)

        # 其他
        self.top_k_save: int = self.config.get("top_k_save", 10)  # 每轮保存 top-K payload

        # ---- WandB 监控配置 ----
        wandb_cfg = self.config.get("wandb", {})
        self.use_wandb: bool = wandb_cfg.get("enabled", True) and _WANDB_AVAILABLE
        self.wandb_project: str = wandb_cfg.get("project", "david-and-goliath")
        self.wandb_run_name: str = wandb_cfg.get("run_name", "")
        self.wandb_tags: list[str] = wandb_cfg.get("tags", ["red-team", "grpo"])

        # ---- 可视化配置 ----
        self.plot_enabled: bool = self.config.get("plot_enabled", True) and _MPL_AVAILABLE
        self.plot_every: int = self.config.get("plot_every", 1)  # 每 N 轮画一次图

        # ---- 运行时状态 (initialize() 中填充) ----
        self._model = None           # LoRA policy model
        self._tokenizer = None       # tokenizer
        self._model_loader = None    # LoRAModelLoader
        self._trl_trainer = None     # trl.GRPOTrainer 实例
        self._current_round = 0      # 当前轮次 (reward_fn 中使用)
        self._wandb_run = None       # wandb.Run 实例

        # ---- 训练历史 (用于可视化) ----
        self._history: list[dict[str, Any]] = []  # 每轮 stats 的累积记录

        # ---- Episode 收集 (reward_fn 回调中填充) ----
        self._episode_buffer: list[EpisodeResult] = []

    # ==============================================================
    # 初始化
    # ==============================================================

    def initialize(self) -> None:
        """加载模型并构建 trl.GRPOTrainer

        分两步:
          1. 通过 LoRAModelLoader 加载 base model + LoRA adapter + tokenizer
          2. 用 trl.GRPOConfig + 自定义 reward_fn 构建 GRPOTrainer
        """
        # ---- Step 1: 加载模型 ----
        self._model_loader = LoRAModelLoader(
            model_name=self.model_name,
            lora_path=self.lora_path,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            quantization=self.quantization,
        )
        self._model, self._tokenizer = self._model_loader.load_model()

        # ---- Step 2: 构建 trl.GRPOTrainer ----
        self._build_trl_trainer()

        # ---- Step 3: 初始化 WandB ----
        self._init_wandb()

        logger.info(
            f"GRPOTrainer initialized: model={self.model_name}, "
            f"group_size={self.group_size}, steps={self.num_train_steps}, "
            f"lr={self.learning_rate}, kl_beta={self.kl_beta}"
        )

    def _build_trl_trainer(self) -> None:
        """构建 trl.GRPOTrainer 实例

        关键配置:
          - reward_funcs: 自定义 reward 函数, 接入 Oracle pipeline
          - num_generations: group_size (每个 prompt 生成的 completion 数)
          - max_completion_length: payload 最大 token 长度
          - beta: KL 正则化系数
        """
        from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig

        # ---- GRPOConfig: 训练超参 ----
        grpo_config = GRPOConfig(
            # 输出目录 (trl 要求)
            output_dir=self.config.get("output_dir", "outputs/grpo_temp"),

            # GRPO 核心参数
            num_generations=self.group_size,              # 每个 prompt 生成 G 个 completion
            max_completion_length=self.max_gen_length,    # payload 最大 token 长度
            beta=self.kl_beta,                            # KL 散度正则化系数

            # 优化器参数
            learning_rate=self.learning_rate,
            per_device_train_batch_size=1,                # 每 step 处理 1 个 prompt (生成 G 个)
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),

            # 生成参数
            temperature=self.temperature,

            # 训练控制
            num_train_epochs=1,                           # 每次 train_round 只跑 1 epoch
            logging_steps=1,
            save_strategy="no",                           # 我们自己管理 checkpoint
            report_to="none",                             # 日志由我们的 logger 处理

            # 显存优化
            bf16=True,
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
        )

        # ---- 构建训练数据集 (每轮动态生成, 这里先用占位) ----
        # trl.GRPOTrainer 需要一个 Dataset, 我们在 train_round() 中动态构建
        # 这里先用空 dataset 初始化, 后续通过 trainer.train() 传入

        # ---- 自定义 reward 函数 ----
        # trl.GRPOTrainer 的 reward_funcs 签名:
        #   reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]
        # 其中 completions 是模型生成的 payload 文本列表

        self._trl_trainer = TRLGRPOTrainer(
            model=self._model,
            reward_funcs=self._reward_function,   # 自定义 reward 回调
            args=grpo_config,
            processing_class=self._tokenizer,
        )

        logger.info("trl.GRPOTrainer built successfully")

    # ==============================================================
    # 每轮训练
    # ==============================================================

    def train_round(
        self,
        round_num: int,
        blue_behavior_summary: str = "",
    ) -> tuple[list[Payload], dict[str, Any]]:
        """执行一轮 GRPO 训练

        完整流程:
          1. 从策略库采样 → 构建 prompt dataset
          2. trl.GRPOTrainer.train() 内部循环:
             - 模型生成 group_size 个 payload
             - 调用 reward_fn → 走完整 episode pipeline → 返回 reward
             - 计算组内相对优势 → 梯度更新
          3. 收集所有 episode 结果 → 存入策略库 → 汇总统计

        Args:
            round_num: 当前轮次
            blue_behavior_summary: 蓝队上一轮行为摘要

        Returns:
            (top_payloads, stats): 本轮最优载荷列表 + 训练统计字典
        """
        train_start = time.time()
        self._current_round = round_num
        self._episode_buffer.clear()

        logger.info(f"[Round {round_num}] Starting GRPO training...")

        # ---- Step 1: 构建 prompt dataset ----
        # 为每个 train step 生成一个不同的 prompt (采样不同的 parents/inspirations)
        prompts = self._build_prompt_dataset(
            num_prompts=self.num_train_steps,
            blue_summary=blue_behavior_summary,
            round_num=round_num,
        )

        logger.info(f"  Built {len(prompts)} prompts for this round")

        # ---- Step 2: 将 prompts 包装为 HuggingFace Dataset ----
        from datasets import Dataset
        train_dataset = Dataset.from_dict({"prompt": prompts})

        # ---- Step 3: 更新 trainer 的 dataset 并训练 ----
        self._trl_trainer.train_dataset = train_dataset
        self._trl_trainer.train()

        # ---- Step 4: 将 episode 中的 payload 存入策略库 ----
        payloads_added = 0
        all_payloads: list[Payload] = []
        for episode in self._episode_buffer:
            # 从 episode 重建 Payload 对象
            payload = Payload(
                id=episode.payload_id,
                round_created=round_num,
                code=self._get_payload_code(episode),
            )
            payload.episode_result = episode
            all_payloads.append(payload)

            if self.strategy_db.add_payload(payload):
                payloads_added += 1

        logger.info(
            f"  {payloads_added}/{len(all_payloads)} payloads added to strategy DB"
        )

        # ---- Step 5: 汇总统计 ----
        train_time = time.time() - train_start
        stats = self._compute_round_stats(self._episode_buffer, train_time)

        # ---- Step 6: 取 top-K payload ----
        all_payloads.sort(
            key=lambda p: p.episode_result.total_reward if p.episode_result else 0,
            reverse=True,
        )
        top_payloads = all_payloads[:self.top_k_save]

        # ---- Step 6: WandB 日志 ----
        self._log_to_wandb(round_num, stats, top_payloads)

        # ---- Step 7: 累积历史 + 可视化 ----
        self._history.append({"round": round_num, **stats})
        if self.plot_enabled and round_num % self.plot_every == 0:
            self._plot_training_curves()

        logger.info(
            f"  Round {round_num} complete: "
            f"ASR={stats['attack_success_rate']:.1%}, "
            f"avg_reward={stats['grpo_avg_reward']:.3f}, "
            f"time={train_time:.1f}s"
        )

        return top_payloads, stats

    # ==============================================================
    # 自定义 Reward 函数 (接入 Oracle Pipeline)
    # ==============================================================

    def _reward_function(
        self,
        completions: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[float]:
        """trl.GRPOTrainer 的自定义 reward 回调

        trl v0.12+ 的 reward_funcs 签名:
          completions: list[list[dict]], 每个元素是一个 completion 的 messages
                       例如 [[{"role": "assistant", "content": "payload text"}]]
          **kwargs: 额外的上下文信息

        对每个生成的 payload 执行完整 episode:
          payload → InjectionEngine 注入 → BlueTeam 处理 → Oracle 评分

        Returns:
            rewards: 与 completions 等长的 float 列表
        """
        rewards: list[float] = []

        for completion in completions:
            # 提取 payload 文本
            # trl 格式: completion = [{"role": "assistant", "content": "..."}]
            payload_text = self._extract_payload_text(completion)

            # 运行完整 episode
            episode = self._run_single_episode(
                payload_text=payload_text,
                round_num=self._current_round,
            )

            # 收集 episode 到 buffer (用于后续统计和存入策略库)
            self._episode_buffer.append(episode)
            rewards.append(episode.total_reward)

        logger.debug(
            f"  Reward batch: n={len(rewards)}, "
            f"mean={sum(rewards)/len(rewards):.3f}, "
            f"max={max(rewards):.3f}, min={min(rewards):.3f}"
        )

        return rewards

    # ==============================================================
    # Prompt Dataset 构建
    # ==============================================================

    def _build_prompt_dataset(
        self,
        num_prompts: int,
        blue_summary: str,
        round_num: int,
    ) -> list[str]:
        """为本轮训练构建 prompt 列表

        每个 prompt 采样不同的 parents/inspirations, 增加生成多样性。
        使用 tokenizer.apply_chat_template 将 messages 转为模型可接受的格式。

        Args:
            num_prompts: 需要生成的 prompt 数量 (= num_train_steps)
            blue_summary: 蓝队行为摘要
            round_num: 当前轮次

        Returns:
            prompts: 文本 prompt 列表 (已经过 chat_template 格式化)
        """
        prompts: list[str] = []

        for _ in range(num_prompts):
            # 从策略库采样 (每次采样不同, 增加多样性)
            parents, inspirations = self.strategy_db.sample(
                n_parents=self.prompt_builder.n_parents,
                n_inspirations=self.prompt_builder.n_inspirations,
            )

            # 构建 messages 格式的 prompt
            messages = self.prompt_builder.build(
                parents=parents,
                inspirations=inspirations,
                blue_summary=blue_summary,
                round_num=round_num,
            )

            # 使用 tokenizer 的 chat template 格式化
            # add_generation_prompt=True: 在末尾添加 assistant 的起始标记, 引导模型续写
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt_text)

        return prompts

    # ==============================================================
    # Checkpoint
    # ==============================================================

    def save_checkpoint(self, path: str | Path) -> None:
        """保存 LoRA adapter checkpoint

        Args:
            path: 保存目录路径
        """
        if self._model_loader is None:
            logger.warning("Model not initialized, skipping checkpoint save")
            return

        self._model_loader.save_adapter(path)
        logger.info(f"Red team checkpoint saved to {path}")

    # ==============================================================
    # 统计计算
    # ==============================================================

    def _compute_round_stats(
        self,
        episodes: list[EpisodeResult],
        train_time: float,
    ) -> dict[str, Any]:
        """从本轮所有 episode 汇总训练统计

        返回的 stats dict 必须包含 co_evolution_controller.py 期望的所有字段:
          - attack_success_rate, grpo_avg_reward, grpo_reward_std,
          - grpo_num_episodes, grpo_train_steps, grpo_train_time_s,
          - judge_a_trigger_rate, judge_b_trigger_rate, avg_payload_quality,
          - red_diversity_coverage

        Args:
            episodes: 本轮所有 episode 结果
            train_time: 本轮总训练时间 (秒)

        Returns:
            stats: 统计字典
        """
        n = len(episodes) if episodes else 1  # 防除零

        # 各项聚合
        rewards = [ep.total_reward for ep in episodes]
        attack_successes = [ep.attack_success for ep in episodes]
        judge_a_triggers = [
            ep.judge_a.has_vulnerability if ep.judge_a else False
            for ep in episodes
        ]
        judge_b_triggers = [
            ep.judge_b.is_manipulated if ep.judge_b else False
            for ep in episodes
        ]
        payload_qualities = [
            ep.judge_c.payload_quality_score if ep.judge_c else 0.0
            for ep in episodes
        ]

        avg_reward = statistics.mean(rewards) if rewards else 0.0
        std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0

        stats = {
            # GRPO 训练统计
            "grpo_num_episodes": n,
            "grpo_train_steps": self.num_train_steps,
            "grpo_avg_reward": avg_reward,
            "grpo_reward_std": std_reward,
            "grpo_train_time_s": train_time,

            # 攻击成功统计
            "attack_success_rate": sum(attack_successes) / n,
            "judge_a_trigger_rate": sum(judge_a_triggers) / n,
            "judge_b_trigger_rate": sum(judge_b_triggers) / n,
            "avg_payload_quality": (
                statistics.mean(payload_qualities)
                if payload_qualities else 0.0
            ),

            # 多样性 (从策略库获取)
            "red_diversity_coverage": self.strategy_db.coverage,
        }

        return stats

    # ==============================================================
    # 辅助方法
    # ==============================================================

    @staticmethod
    def _extract_payload_text(completion: list[dict[str, str]]) -> str:
        """从 trl 的 completion 格式中提取纯文本

        trl 返回的 completion 格式:
          [{"role": "assistant", "content": "generated payload text"}]

        Args:
            completion: 单个 completion 的 messages 列表

        Returns:
            payload_text: 提取的纯文本
        """
        if isinstance(completion, list):
            # messages 格式: 取最后一个 assistant message
            for msg in reversed(completion):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
            # fallback: 取最后一条消息的 content
            if completion and isinstance(completion[-1], dict):
                return completion[-1].get("content", "")
        if isinstance(completion, str):
            return completion
        return str(completion)

    def _get_payload_code(self, episode: EpisodeResult) -> str:
        """从 EpisodeResult 中恢复 payload 代码

        episode 中没有直接存储 payload.code (因为 Payload 对象在 reward_fn 中是临时的),
        但我们可以从 blue_response.reasoning 或 injected_content 中间接获取。

        如果无法恢复, 返回一个标记字符串。
        """
        # 从 episode_buffer 中对应的 payload 获取
        # 注意: 这个方法在 train_round 中被调用, 此时 episode 是从 _episode_buffer 取的
        # 我们需要一个更好的方式来追踪 payload text...
        # 暂时在 _run_single_episode 中将 payload.code 存入 episode 的一个临时字段
        if hasattr(episode, '_payload_code'):
            return episode._payload_code
        return f"[payload_{episode.payload_id}]"

    def _run_single_episode(
        self,
        payload_text: str,
        round_num: int,
    ) -> EpisodeResult:
        """运行单个 episode: payload → 注入 → 蓝队 → 评判

        这是 reward 计算的核心路径:
          1. 将 payload 文本包装为 Payload 对象
          2. InjectionEngine 将 payload 注入随机编程任务
          3. BlueTeam (coding agent) 处理注入后的任务, 生成代码
          4. HybridOracle 三 Judge 评分 → 得到 OracleReward

        Args:
            payload_text: 红队模型生成的 payload 原始文本
            round_num: 当前训练轮次

        Returns:
            episode: 完整的 EpisodeResult (包含所有 Judge 评分)
        """
        # ---- 1. 构建 Payload ----
        payload = Payload(
            round_created=round_num,
            code=payload_text.strip(),
        )

        # ---- 2. 注入 ----
        injected_task = self.injection_engine.inject(payload)

        # ---- 3. 蓝队处理 ----
        try:
            blue_response: BlueTeamResponse = self.blue_team.process(injected_task)
        except Exception as e:
            logger.warning(f"Blue team error for payload {payload.id}: {e}")
            blue_response = BlueTeamResponse(
                generated_code="",
                reasoning=f"Error: {e}",
            )

        # ---- 4. Oracle 评分 ----
        try:
            oracle_reward = self.oracle.evaluate(
                payload=payload,
                injected_task=injected_task,
                blue_response=blue_response,
                round_num=round_num,
            )
        except Exception as e:
            logger.warning(f"Oracle error for payload {payload.id}: {e}")
            from core.types import OracleReward
            oracle_reward = OracleReward(total_reward=0.0)

        # ---- 5. 组装 EpisodeResult ----
        episode = EpisodeResult(
            payload_id=payload.id,
            round=round_num,
            coding_task_id=injected_task.original_task.id,
            injection_position=injected_task.injection_position,
            blue_response=blue_response,
        )
        episode.sync_from_oracle(oracle_reward)

        # 保存 Judge 详细结果 (如果 oracle 返回了)
        if hasattr(oracle_reward, '_judge_a_result'):
            episode.judge_a = oracle_reward._judge_a_result
        if hasattr(oracle_reward, '_judge_b_result'):
            episode.judge_b = oracle_reward._judge_b_result
        if hasattr(oracle_reward, '_judge_c_result'):
            episode.judge_c = oracle_reward._judge_c_result

        # 暂存 payload 原文, 供后续存入策略库时使用
        episode._payload_code = payload_text.strip()

        return episode

    # ==============================================================
    # WandB 监控
    # ==============================================================

    def _init_wandb(self) -> None:
        """初始化 WandB run

        记录超参数配置, 方便在 WandB Dashboard 上筛选和对比实验。
        如果 wandb 未安装或用户未启用, 则跳过。
        """
        if not self.use_wandb:
            logger.info("WandB disabled or not installed, skipping")
            return

        # 构造 run name: 默认 = 模型短名 + 时间戳
        if not self.wandb_run_name:
            model_short = self.model_name.split("/")[-1]  # "Qwen2.5-7B-Instruct"
            self.wandb_run_name = f"red-grpo-{model_short}"

        # 记录到 wandb 的超参数
        wandb_config = {
            "model_name": self.model_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "quantization": self.quantization,
            "group_size": self.group_size,
            "num_train_steps": self.num_train_steps,
            "learning_rate": self.learning_rate,
            "kl_beta": self.kl_beta,
            "clip_eps": self.clip_eps,
            "max_gen_length": self.max_gen_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k_save": self.top_k_save,
        }

        self._wandb_run = wandb.init(
            project=self.wandb_project,
            name=self.wandb_run_name,
            tags=self.wandb_tags,
            config=wandb_config,
            reinit=True,              # 允许同一进程多次 init
        )

        logger.info(
            f"WandB initialized: project={self.wandb_project}, "
            f"run={self.wandb_run_name}"
        )

    def _log_to_wandb(
        self,
        round_num: int,
        stats: dict[str, Any],
        top_payloads: list[Payload],
    ) -> None:
        """将一轮训练的统计数据记录到 WandB

        记录内容:
          - 标量指标: reward, ASR, Judge 触发率, 多样性覆盖率等
          - Reward 分布直方图: 观察 reward 是否有模式崩塌
          - Top payload 文本表格: 便于人工检查生成质量

        Args:
            round_num: 当前轮次
            stats: _compute_round_stats() 返回的统计字典
            top_payloads: 本轮 top-K payload 列表
        """
        if not self.use_wandb or self._wandb_run is None:
            return

        # ---- 1. 标量指标 (按分组整理, WandB 自动分 panel) ----
        log_dict = {
            # 训练核心指标
            "train/avg_reward": stats["grpo_avg_reward"],
            "train/reward_std": stats["grpo_reward_std"],
            "train/num_episodes": stats["grpo_num_episodes"],
            "train/train_time_s": stats["grpo_train_time_s"],

            # 攻击效果指标
            "attack/success_rate": stats["attack_success_rate"],
            "attack/judge_a_trigger_rate": stats["judge_a_trigger_rate"],
            "attack/judge_b_trigger_rate": stats["judge_b_trigger_rate"],
            "attack/avg_payload_quality": stats["avg_payload_quality"],

            # 多样性指标
            "diversity/coverage": stats["red_diversity_coverage"],

            # 轮次 (用作 x 轴)
            "round": round_num,
        }

        # ---- 2. Reward 分布直方图 ----
        # 帮助观察: reward 是否集中在某个值 (模式崩塌)
        episode_rewards = [ep.total_reward for ep in self._episode_buffer]
        if episode_rewards:
            log_dict["train/reward_histogram"] = wandb.Histogram(episode_rewards)

        # ---- 3. Top payload 文本表格 ----
        # 每轮记录 top payloads 的文本, 人工可在 WandB 上浏览
        if top_payloads:
            table = wandb.Table(columns=["rank", "payload_text", "reward", "type", "stealth"])
            for rank, p in enumerate(top_payloads[:5], 1):  # 最多记 5 条
                reward = p.episode_result.total_reward if p.episode_result else 0.0
                inj_type = p.injection_type.name if p.injection_type else "N/A"
                stealth = p.stealth_level.name if p.stealth_level else "N/A"
                table.add_data(rank, p.code[:300], reward, inj_type, stealth)
            log_dict["payloads/top_samples"] = table

        wandb.log(log_dict, step=round_num)

    def finish_wandb(self) -> None:
        """结束 WandB run (训练完成时调用)

        由 co_evolution_controller.finalize() 在训练结束后调用,
        确保所有日志都被上传。
        """
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
            logger.info("WandB run finished")

    # ==============================================================
    # 训练可视化
    # ==============================================================

    def _plot_training_curves(self) -> None:
        """绘制训练曲线并保存为 PNG

        生成 2×2 子图:
          - 左上: Avg Reward ± std (训练效果)
          - 右上: Attack Success Rate (攻击成功率趋势)
          - 左下: Judge A/B 触发率 (分 Judge 追踪)
          - 右下: Diversity Coverage (策略库多样性)

        文件保存到 output_dir/plots/training_curves.png,
        同时上传到 WandB (如果启用)。
        """
        if not _MPL_AVAILABLE:
            logger.debug("matplotlib not available, skipping plot")
            return

        if len(self._history) < 2:
            return  # 至少 2 个点才画图

        # ---- 提取数据序列 ----
        rounds = [h["round"] for h in self._history]
        avg_rewards = [h["grpo_avg_reward"] for h in self._history]
        reward_stds = [h["grpo_reward_std"] for h in self._history]
        asrs = [h["attack_success_rate"] for h in self._history]
        ja_rates = [h["judge_a_trigger_rate"] for h in self._history]
        jb_rates = [h["judge_b_trigger_rate"] for h in self._history]
        coverages = [h["red_diversity_coverage"] for h in self._history]

        # ---- 绘制 2×2 子图 ----
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Red Team GRPO Training Progress", fontsize=14, fontweight="bold")

        # 左上: Avg Reward ± std
        ax = axes[0, 0]
        ax.plot(rounds, avg_rewards, "b-o", markersize=4, label="avg reward")
        # 绘制 ±1 std 阴影区间
        reward_upper = [r + s for r, s in zip(avg_rewards, reward_stds)]
        reward_lower = [r - s for r, s in zip(avg_rewards, reward_stds)]
        ax.fill_between(rounds, reward_lower, reward_upper, alpha=0.2, color="blue")
        ax.set_xlabel("Round")
        ax.set_ylabel("Reward")
        ax.set_title("Average Reward (±1 std)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # 右上: Attack Success Rate
        ax = axes[0, 1]
        ax.plot(rounds, asrs, "r-o", markersize=4, label="ASR")
        ax.set_xlabel("Round")
        ax.set_ylabel("Rate")
        ax.set_title("Attack Success Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # 左下: Judge A/B 触发率
        ax = axes[1, 0]
        ax.plot(rounds, ja_rates, "g-s", markersize=4, label="Judge A (vuln)")
        ax.plot(rounds, jb_rates, "m-^", markersize=4, label="Judge B (manip)")
        ax.set_xlabel("Round")
        ax.set_ylabel("Trigger Rate")
        ax.set_title("Judge Trigger Rates")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # 右下: Diversity Coverage
        ax = axes[1, 1]
        ax.plot(rounds, coverages, "k-D", markersize=4, label="coverage")
        ax.set_xlabel("Round")
        ax.set_ylabel("Coverage")
        ax.set_title("Strategy DB Diversity Coverage")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # ---- 保存到文件 ----
        output_dir = Path(self.config.get("output_dir", "outputs/grpo_temp"))
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "training_curves.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Training curves saved to {plot_path}")

        # ---- 同步到 WandB ----
        if self.use_wandb and self._wandb_run is not None:
            wandb.log({"plots/training_curves": wandb.Image(str(plot_path))})

    def plot_reward_distribution(self, round_num: int) -> None:
        """绘制单轮的 reward 分布直方图 (按需调用)

        展示本轮所有 episode 的 reward 分布, 帮助诊断:
          - 双峰分布: 说明 payload 质量差异大 (好现象, 有探索)
          - 集中在 0: 攻击效果差, 需要调参
          - 集中在高分: 可能过拟合某种攻击模式

        Args:
            round_num: 当前轮次 (用于文件名和标题)
        """
        if not _MPL_AVAILABLE or not self._episode_buffer:
            return

        rewards = [ep.total_reward for ep in self._episode_buffer]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(rewards, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(statistics.mean(rewards), color="red", linestyle="--",
                   label=f"mean={statistics.mean(rewards):.3f}")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        ax.set_title(f"Reward Distribution — Round {round_num}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_dir = Path(self.config.get("output_dir", "outputs/grpo_temp"))
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"reward_dist_round_{round_num:03d}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Reward distribution plot saved to {plot_path}")

        if self.use_wandb and self._wandb_run is not None:
            wandb.log({
                "plots/reward_distribution": wandb.Image(str(plot_path)),
            }, step=round_num)
