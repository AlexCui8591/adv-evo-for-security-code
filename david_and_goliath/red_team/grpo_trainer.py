"""red_team/grpo_trainer.py — 红队 GRPO 训练器 (OpenRLHF + Ray + vLLM)

训练集群架构:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Ray Cluster                                                     │
  │                                                                  │
  │  Training GPUs (2–4 × A100-80G)  ← DeepSpeed ZeRO-3            │
  │  ┌───────────────────────────────────────────────────────────┐   │
  │  │  Actor  (LoRA policy, ZeRO-3 参数分片)                    │   │
  │  │  Reference (冻结副本, 用于 KL 惩罚)                       │   │
  │  └───────────────────────────────────────────────────────────┘   │
  │                                                                  │
  │  Inference GPUs (6–8 × A100-80G)  ← vLLM Engine Pool           │
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
  │  │ vLLM Eng. 0 │  │ vLLM Eng. 1 │  │ vLLM Eng. N │  tp=2       │
  │  │  (2 GPUs)   │  │  (2 GPUs)   │  │  (2 GPUs)   │             │
  │  └─────────────┘  └─────────────┘  └─────────────┘             │
  │                                                                  │
  │  CPU Workers                                                     │
  │  ┌──────────────────────────────────────────────────────────┐    │
  │  │  OracleRewardWorker × N                                  │    │
  │  │  (InjectionEngine → BlueTeam → HybridOracle → reward)   │    │
  │  └──────────────────────────────────────────────────────────┘    │
  └──────────────────────────────────────────────────────────────────┘

设计决策:
  - 推理 / 训练 GPU 分离: vLLM 生成时, 训练卡做 optimizer 更新, 互不阻塞
  - OracleRewardWorker Pool: BlueTeam API 调用并发化, 消除 reward 计算瓶颈
  - DeepSpeed ZeRO-3: 14B 模型在 2–4 × A100 上可全量训练, 无需量化
  - GRPO group_size=8: 每 prompt 生成 8 个 completion, 组内相对 advantage

每轮 train_round 流程:
  1. RedPromptBuilder 从 MAPElitesDB 采样, 构建 prompt batch
  2. vLLM Engine Pool 并行生成 (group_size × num_prompts 个 completion)
  3. OracleRewardWorker Pool 并行跑完整 episode pipeline → scalar reward
  4. GRPO: advantage = (r_i - mean_group) / (std_group + eps)
  5. Actor (DeepSpeed ZeRO-3) 执行 policy gradient 更新 + KL 惩罚
  6. 高分 payload → MAPElitesDB

依赖:
  - openrlhf >= 0.3.0: Actor, PPORayActorGroup, LLMRayActor, DeepspeedStrategy
  - ray >= 2.9.0: 分布式 actor 调度
  - vllm >= 0.4.0: 高吞吐推理
  - deepspeed >= 0.14.0: ZeRO-3 分布式优化

接口约束 (由 co_evolution_controller.py 定义):
  - initialize(): 初始化 Ray 集群 + 模型
  - train_round(round_num, blue_behavior_summary) → (top_payloads, stats)
  - save_checkpoint(path): 保存 LoRA adapter
"""

from __future__ import annotations

import logging
import statistics
import time
from pathlib import Path
from typing import Any, Optional

# OpenRLHF 核心组件
from openrlhf.models import Actor
from openrlhf.trainer.ray import (
    ActorModelRayActor,
    PPORayActorGroup,
)
from openrlhf.trainer.ray.vllm_engine import LLMRayActor
from openrlhf.utils import DeepspeedStrategy

# Ray 分布式调度
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from core.types import (
    BlueTeamResponse,
    EpisodeResult,
    OracleReward,
    Payload,
)
from core.injection_engine import InjectionEngine
from core.strategy_db import MAPElitesDB
from red_team.prompt_builder import RedPromptBuilder

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


# ============================================================
#  Ray Actor: Oracle Reward Pipeline Worker
# ============================================================

@ray.remote(num_cpus=2)
class OracleRewardWorker:
    """Ray 远程 Actor — 运行完整的 reward pipeline

    设计目的:
      BlueTeam 的 LLM 推理是 reward 计算的主要延迟来源。
      把它包成 Ray Actor Pool, 多 worker 并发处理 payload batch,
      让 vLLM 生成和 reward 计算之间不存在串行等待。

    每个 worker 持有独立的 oracle/blue_team/injection_engine 实例。
    """

    def __init__(
        self,
        oracle,
        blue_team,
        injection_engine: InjectionEngine,
    ):
        self.oracle = oracle
        self.blue_team = blue_team
        self.injection_engine = injection_engine

    def score_batch(
        self,
        payload_texts: list[str],
        round_num: int,
    ) -> list[dict[str, Any]]:
        """对一批 payload 文本跑完整 episode pipeline。

        Args:
            payload_texts: vLLM 生成的 payload 文本列表
            round_num: 当前训练轮次 (写入 Payload.round_created)

        Returns:
            每条 payload 对应的 result dict, 包含 reward 和各分项分数
        """
        return [self._run_episode(text, round_num) for text in payload_texts]

    def _run_episode(self, payload_text: str, round_num: int) -> dict[str, Any]:
        """payload → 注入 → 蓝队 → Oracle → result dict"""
        payload = Payload(
            round_created=round_num,
            code=payload_text.strip(),
        )

        # 1. 注入编程任务
        injected_task = self.injection_engine.inject(payload)

        # 2. 蓝队处理
        try:
            blue_response: BlueTeamResponse = self.blue_team.process(injected_task)
        except Exception as exc:
            logger.warning("BlueTeam error [%s]: %s", payload.id, exc)
            blue_response = BlueTeamResponse(generated_code="", reasoning=f"Error: {exc}")

        # 3. Oracle 评分
        try:
            oracle_reward: OracleReward = self.oracle.evaluate(
                payload=payload,
                injected_task=injected_task,
                blue_response=blue_response,
                round_num=round_num,
            )
        except Exception as exc:
            logger.warning("Oracle error [%s]: %s", payload.id, exc)
            oracle_reward = OracleReward(total_reward=0.0)

        return {
            "reward": oracle_reward.total_reward,
            "payload_id": payload.id,
            "payload_code": payload.code,
            "attack_success": oracle_reward.attack_success,
            "judge_a_score": oracle_reward.vulnerability_reward,
            "judge_b_score": oracle_reward.manipulation_reward,
            "judge_c_score": oracle_reward.quality_reward,
            "stealth_score": oracle_reward.stealth_reward,
            "oracle_reward": oracle_reward,
            "payload": payload,
            "injected_task": injected_task,
            "blue_response": blue_response,
        }


# ============================================================
#  主训练器
# ============================================================

class GRPOTrainer:
    """红队 GRPO 训练器 — OpenRLHF + Ray + vLLM 版

    GPU 资源分配 (config["cluster"] 控制):
      Training GPUs (2–4 × A100): Actor + Reference, DeepSpeed ZeRO-3
      Inference GPUs (6–8 × A100): vLLM Engine Pool, tensor parallel
      CPU workers: OracleRewardWorker Pool (oracle pipeline 并发)

    Args:
        model_name: 红队基座模型 (如 "Qwen/Qwen2.5-14B-Instruct")
        lora_path:  已有 LoRA checkpoint 路径 (空 = 从零 fine-tune)
        oracle:     HybridOracle 实例
        blue_team:  CodingAgent 实例
        injection_engine: InjectionEngine 实例
        strategy_db: MAPElitesDB 实例
        prompt_builder: RedPromptBuilder 实例
        config:     dict, 包含 cluster/deepspeed/vllm/grpo/wandb 子配置
    """

    def __init__(
        self,
        model_name: str,
        lora_path: str,
        oracle,
        blue_team,
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

        # ---- 集群配置 ----
        cluster_cfg = self.config.get("cluster", {})
        self.num_training_gpus: int = cluster_cfg.get("num_training_gpus", 4)
        self.num_inference_gpus: int = cluster_cfg.get("num_inference_gpus", 8)
        self.tensor_parallel_size: int = cluster_cfg.get("tensor_parallel_size", 2)
        self.num_reward_workers: int = cluster_cfg.get("num_reward_workers", 4)
        # num_vllm_engines = inference GPUs ÷ tensor_parallel_size
        self.num_vllm_engines: int = (
            self.num_inference_gpus // self.tensor_parallel_size
        )

        # ---- DeepSpeed 配置 ----
        ds_cfg = self.config.get("deepspeed", {})
        self.zero_stage: int = ds_cfg.get("zero_stage", 3)
        self.offload_optimizer: bool = ds_cfg.get("offload_optimizer", False)
        self.offload_param: bool = ds_cfg.get("offload_param", False)

        # ---- vLLM 配置 ----
        vllm_cfg = self.config.get("vllm", {})
        self.vllm_gpu_mem_util: float = vllm_cfg.get("gpu_memory_utilization", 0.90)
        self.vllm_max_model_len: int = vllm_cfg.get("max_model_len", 4096)
        self.vllm_enforce_eager: bool = vllm_cfg.get("enforce_eager", False)

        # ---- GRPO 超参 ----
        grpo_cfg = self.config.get("grpo", {})
        self.group_size: int = grpo_cfg.get("group_size", 8)
        self.grpo_eps: float = grpo_cfg.get("eps", 1e-8)
        self.kl_coeff: float = grpo_cfg.get("kl_coeff", 0.01)
        self.clip_eps: float = grpo_cfg.get("clip_eps", 0.2)
        self.max_gen_length: int = grpo_cfg.get("max_gen_length", 512)
        self.temperature: float = grpo_cfg.get("temperature", 0.9)
        self.top_p: float = grpo_cfg.get("top_p", 0.95)
        self.learning_rate: float = grpo_cfg.get("learning_rate", 5e-6)
        self.prompts_per_round: int = grpo_cfg.get("prompts_per_round", 64)
        self.top_k_save: int = grpo_cfg.get("top_k_save", 10)

        # ---- LoRA 配置 (Actor 初始化时使用) ----
        lora_cfg = self.config.get("lora", {})
        self.lora_r: int = lora_cfg.get("r", 16)
        self.lora_alpha: int = lora_cfg.get("alpha", 32)
        self.lora_dropout: float = lora_cfg.get("dropout", 0.05)

        # ---- WandB 监控配置 ----
        wandb_cfg = self.config.get("wandb", {})
        self.use_wandb: bool = wandb_cfg.get("enabled", True) and _WANDB_AVAILABLE
        self.wandb_project: str = wandb_cfg.get("project", "david-and-goliath")
        self.wandb_run_name: str = wandb_cfg.get("run_name", "")
        self.wandb_tags: list[str] = wandb_cfg.get("tags", ["red-team", "grpo", "openrlhf"])

        # ---- 可视化配置 ----
        self.plot_enabled: bool = self.config.get("plot_enabled", True) and _MPL_AVAILABLE
        self.plot_every: int = self.config.get("plot_every", 5)
        self._plot_dir = Path(self.config.get("output_dir", "outputs/grpo")) / "plots"

        # ---- 运行时状态 (initialize() 后有效) ----
        self._initialized = False
        self._current_round = 0
        self._wandb_run = None
        self._history: list[dict[str, Any]] = []

        # Ray actors
        self._actor_group: Optional[PPORayActorGroup] = None
        self._vllm_engines: list = []
        self._reward_workers: list = []
        self._training_pg = None
        self._inference_pg = None

    # ==============================================================
    #  Public Interface
    # ==============================================================

    def initialize(self) -> None:
        """初始化 Ray 集群、DeepSpeed Actor、vLLM Engine Pool、Reward Workers。

        调用前确保 Ray 集群已启动:
          ray start --head --num-gpus=<total_gpus>
        或在代码里调用 ray.init(address="auto")。
        """
        logger.info("Initializing GRPOTrainer (OpenRLHF + Ray + vLLM)...")

        # 1. Ray 初始化
        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)
        logger.info("Ray cluster resources: %s", ray.cluster_resources())

        # 2. GPU Placement Groups: 训练卡 / 推理卡分开
        self._build_placement_groups()

        # 3. vLLM 推理引擎 (在 inference PG 上)
        self._setup_vllm_engines()

        # 4. DeepSpeed Actor (在 training PG 上)
        self._setup_actor_model()

        # 5. Oracle Reward Worker Pool (CPU)
        self._setup_reward_workers()

        # 6. WandB
        if self.use_wandb:
            self._init_wandb()

        self._initialized = True
        logger.info(
            "GRPOTrainer ready: %d training GPUs (ZeRO-%d), "
            "%d vLLM engines (tp=%d, %d inference GPUs), "
            "%d reward workers",
            self.num_training_gpus,
            self.zero_stage,
            self.num_vllm_engines,
            self.tensor_parallel_size,
            self.num_inference_gpus,
            self.num_reward_workers,
        )

    def train_round(
        self,
        round_num: int,
        blue_behavior_summary: str = "",
    ) -> tuple[list[Payload], dict[str, Any]]:
        """执行一轮 GRPO 训练。

        完整流程:
          1. RedPromptBuilder 构建 prompt batch (采样策略库 inspirations)
          2. vLLM Engine Pool 并行生成 group_size × num_prompts 个 completion
          3. OracleRewardWorker Pool 并行跑完整 episode pipeline → scalar reward
          4. GRPO advantage = (r_i - mean_group) / (std_group + eps)
          5. Actor (DeepSpeed ZeRO-3) policy gradient update + KL 惩罚
          6. 高分 payload → MAPElitesDB

        Args:
            round_num: 当前训练轮次
            blue_behavior_summary: 蓝队上一轮行为摘要 (用于 prompt 构建)

        Returns:
            top_payloads: 本轮最高分 payload 列表
            stats: 本轮训练统计指标字典
        """
        assert self._initialized, "Call initialize() first"
        self._current_round = round_num
        t_start = time.time()

        # ---- 1. 构建 prompts ----
        prompts = self._build_prompts(round_num, blue_behavior_summary)

        # ---- 2. vLLM 并行生成 ----
        # completions_grouped: List[List[str]], shape [num_prompts, group_size]
        logger.info(
            "[Round %d] vLLM generating %d × %d completions...",
            round_num, len(prompts), self.group_size,
        )
        completions_grouped = self._run_generation(prompts)

        # Flatten → 一维列表, 同时记录每条属于哪个 prompt
        flat_completions: list[str] = []
        flat_prompt_indices: list[int] = []
        for prompt_idx, group in enumerate(completions_grouped):
            for completion in group:
                flat_completions.append(completion)
                flat_prompt_indices.append(prompt_idx)

        # ---- 3. Oracle reward pipeline (并行) ----
        logger.info(
            "[Round %d] Scoring %d completions via Oracle workers...",
            round_num, len(flat_completions),
        )
        flat_results = self._run_reward_pipeline(flat_completions, round_num)

        # ---- 4. GRPO advantage 计算 ----
        rewards_by_group = self._group_rewards(
            flat_results, flat_prompt_indices, len(prompts)
        )
        advantages_by_group = self._compute_grpo_advantages(rewards_by_group)

        # ---- 5. DeepSpeed policy gradient update ----
        logger.info(
            "[Round %d] GRPO update (DeepSpeed ZeRO-%d)...",
            round_num, self.zero_stage,
        )
        self._grpo_update(prompts, completions_grouped, advantages_by_group)

        # ---- 6. 存高分 payload 到策略库 ----
        top_payloads = self._store_top_payloads(flat_results)

        # ---- 7. 统计 & 监控 ----
        elapsed = time.time() - t_start
        stats = self._compute_round_stats(flat_results, elapsed)

        self._log_to_wandb(round_num, stats, top_payloads)
        self._history.append({"round": round_num, **stats})

        if self.plot_enabled and round_num % self.plot_every == 0:
            self._plot_training_curves()

        logger.info(
            "[Round %d] reward_mean=%.3f ASR=%.2f%% elapsed=%.1fs",
            round_num,
            stats["grpo_avg_reward"],
            stats["attack_success_rate"] * 100,
            elapsed,
        )
        return top_payloads, stats

    def save_checkpoint(self, path: str) -> None:
        """保存 Actor LoRA adapter 权重。"""
        assert self._initialized, "Call initialize() first"
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        # rank-0 worker 执行 save
        ray.get(self._actor_group.actors[0].save_model.remote(str(save_path)))
        logger.info("Checkpoint saved to %s", save_path)

    def finish_wandb(self) -> None:
        """结束 WandB run (训练完成时调用)。"""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
            logger.info("WandB run finished")

    def finish(self) -> None:
        """清理 Ray actors 和 WandB run。"""
        self.finish_wandb()
        if self._training_pg is not None:
            remove_placement_group(self._training_pg)
        if self._inference_pg is not None:
            remove_placement_group(self._inference_pg)

    # ==============================================================
    #  Ray 集群初始化
    # ==============================================================

    def _build_placement_groups(self) -> None:
        """为 training 和 inference 分别创建 GPU placement groups。

        Training PG  : num_training_gpus 个 GPU → Actor + Reference (ZeRO-3)
        Inference PG : num_inference_gpus 个 GPU → vLLM engines (每组 tp 个 GPU)

        PACK 策略: 尽量把 bundle 调度到同一节点, 保证 NVLink/NVSwitch 带宽。
        """
        # 训练 PG: 每个 GPU 独立一个 bundle, ZeRO-3 worker 1-GPU-per-rank
        self._training_pg = placement_group(
            bundles=[{"GPU": 1, "CPU": 4}] * self.num_training_gpus,
            strategy="PACK",
            name="training_pg",
        )
        ray.get(self._training_pg.ready())
        logger.info("Training PG ready: %d GPUs (ZeRO-%d)", self.num_training_gpus, self.zero_stage)

        # 推理 PG: 每个 vLLM engine 占 tensor_parallel_size 个 GPU
        self._inference_pg = placement_group(
            bundles=[{"GPU": self.tensor_parallel_size, "CPU": 2}] * self.num_vllm_engines,
            strategy="PACK",
            name="inference_pg",
        )
        ray.get(self._inference_pg.ready())
        logger.info(
            "Inference PG ready: %d engines × tp=%d GPUs",
            self.num_vllm_engines, self.tensor_parallel_size,
        )

    def _setup_vllm_engines(self) -> None:
        """在 inference PG 上启动 vLLM Ray Actors。

        多引擎并行生成: 吞吐 ≈ num_vllm_engines × single_engine_qps
        每 engine 的 LoRA 权重从 actor_group 同步 (需要在 train_round 开始时 broadcast)。
        """
        self._vllm_engines = []
        for i in range(self.num_vllm_engines):
            engine = LLMRayActor.options(
                num_gpus=self.tensor_parallel_size,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._inference_pg,
                    placement_group_bundle_index=i,
                ),
                name=f"vllm_engine_{i}",
            ).remote(
                model=self.model_name,
                trust_remote_code=True,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_mem_util,
                max_model_len=self.vllm_max_model_len,
                enforce_eager=self.vllm_enforce_eager,
                dtype="bfloat16",
                enable_lora=True,         # 支持 LoRA 权重热更新
                max_lora_rank=self.lora_r,
            )
            self._vllm_engines.append(engine)
        logger.info("Launched %d vLLM engines", len(self._vllm_engines))

    def _setup_actor_model(self) -> None:
        """在 training PG 上用 OpenRLHF PPORayActorGroup 初始化 DeepSpeed Actor。

        ZeRO-3 把模型参数分片到所有训练 GPU:
          - 14B 模型 × bf16 ≈ 28GB; ZeRO-3 分片到 4 卡 → 每卡 ~7GB 参数
          - optimizer states + gradients 也均匀分片
          - offload_optimizer=True 可进一步省显存 (牺牲一些速度)
        """
        ds_strategy = DeepspeedStrategy(
            zero_stage=self.zero_stage,
            offload_optimizer=self.offload_optimizer,
            offload_param=self.offload_param,
            bf16=True,
            learning_rate=self.learning_rate,
        )

        self._actor_group = PPORayActorGroup(
            num_nodes=1,
            num_gpus_per_node=self.num_training_gpus,
            ray_actor_type=ActorModelRayActor,
            pg=self._training_pg,
            num_gpus_per_actor=1,   # ZeRO-3: 每 rank 1 GPU
        )

        init_refs = self._actor_group.async_init_model_from_pretrained(
            strategy=ds_strategy,
            pretrain=self.model_name,
            lora_path=self.lora_path if self.lora_path else None,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            use_flash_attention_2=True,
        )
        ray.get(init_refs)
        logger.info(
            "Actor model initialized: %s on %d GPUs (ZeRO-%d)",
            self.model_name, self.num_training_gpus, self.zero_stage,
        )

    def _setup_reward_workers(self) -> None:
        """启动 Oracle Reward Worker Pool (CPU workers)。

        Worker 数量 = num_reward_workers; 一般设为 BlueTeam 并发 API 调用的上限。
        每个 worker 持有独立的 oracle/blue_team/injection_engine 实例。
        """
        self._reward_workers = [
            OracleRewardWorker.remote(
                oracle=self.oracle,
                blue_team=self.blue_team,
                injection_engine=self.injection_engine,
            )
            for _ in range(self.num_reward_workers)
        ]
        logger.info("Launched %d Oracle reward workers", self.num_reward_workers)

    # ==============================================================
    #  每轮训练步骤
    # ==============================================================

    def _build_prompts(
        self,
        round_num: int,
        blue_summary: str,
    ) -> list[str]:
        """从策略库采样 parents + inspirations, 构建 prompt batch。

        策略库为空时 (Round 1 冷启动) sample() 返回 ([], []),
        prompt_builder.build() 会生成无示例的基础 prompt。
        """
        prompts = []
        for _ in range(self.prompts_per_round):
            # sample() 返回 (parents, inspirations) 两个列表
            parents, inspirations = self.strategy_db.sample(
                n_parents=self.prompt_builder.n_parents,
                n_inspirations=self.prompt_builder.n_inspirations,
            )
            prompt = self.prompt_builder.build_text(
                parents=parents,
                inspirations=inspirations,
                round_num=round_num,
                blue_summary=blue_summary,
            )
            prompts.append(prompt)
        return prompts

    def _run_generation(self, prompts: list[str]) -> list[list[str]]:
        """vLLM Engine Pool 并行生成 completions。

        每个 prompt 生成 group_size 个 completion (GRPO 组内 reward 对比)。
        Prompts 均匀分发到各 engine, 异步 gather。

        Returns:
            completions_grouped: shape [num_prompts, group_size]
        """
        # 在 每轮 generation 前, 把最新 LoRA 权重同步到 vLLM engines
        self._sync_lora_to_vllm()

        # 按 engine 数量拆分 prompts
        n = len(self._vllm_engines)
        chunk_size = max(1, (len(prompts) + n - 1) // n)
        chunks = [prompts[i: i + chunk_size] for i in range(0, len(prompts), chunk_size)]

        sampling_params = {
            "n": self.group_size,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_gen_length,
        }

        # 异步发出所有 generate 请求
        futures = [
            engine.generate.remote(chunk, sampling_params)
            for engine, chunk in zip(self._vllm_engines, chunks)
        ]

        # 收集 + 拼回完整 list
        completions_grouped: list[list[str]] = []
        for chunk_result in ray.get(futures):
            completions_grouped.extend(chunk_result)

        return completions_grouped[: len(prompts)]

    def _sync_lora_to_vllm(self) -> None:
        """将训练卡上最新的 LoRA 权重广播到所有 vLLM engines。

        OpenRLHF 的 ActorModelRayActor 提供 get_model_state_dict.remote()
        接口, 用于导出 LoRA delta weights; vLLM engine 通过
        update_lora_weights.remote() 热更新, 无需重启推理进程。
        """
        # 从 rank-0 actor 导出 LoRA adapter 权重
        lora_state_dict = ray.get(
            self._actor_group.actors[0].get_lora_state_dict.remote()
        )
        # 异步广播到全部 vLLM engines
        sync_refs = [
            engine.update_lora_weights.remote(lora_state_dict)
            for engine in self._vllm_engines
        ]
        ray.get(sync_refs)

    def _run_reward_pipeline(
        self,
        flat_completions: list[str],
        round_num: int,
    ) -> list[dict[str, Any]]:
        """将 flat completions 均匀分发给 reward workers, 并行跑 Oracle pipeline。

        Returns:
            flat_results: 与 flat_completions 一一对应的 result dict 列表
        """
        n = self.num_reward_workers
        chunk_size = max(1, (len(flat_completions) + n - 1) // n)
        chunks = [flat_completions[i: i + chunk_size] for i in range(0, len(flat_completions), chunk_size)]

        futures = [
            worker.score_batch.remote(chunk, round_num)
            for worker, chunk in zip(self._reward_workers, chunks)
        ]

        flat_results: list[dict] = []
        for chunk_result in ray.get(futures):
            flat_results.extend(chunk_result)

        return flat_results[: len(flat_completions)]

    @staticmethod
    def _group_rewards(
        flat_results: list[dict],
        flat_prompt_indices: list[int],
        num_prompts: int,
    ) -> list[list[float]]:
        """将 flat rewards 按 prompt group 重新分组。

        Returns:
            shape [num_prompts, group_size]
        """
        groups: list[list[float]] = [[] for _ in range(num_prompts)]
        for result, pidx in zip(flat_results, flat_prompt_indices):
            groups[pidx].append(result["reward"])
        return groups

    def _compute_grpo_advantages(
        self,
        rewards_by_group: list[list[float]],
    ) -> list[list[float]]:
        """GRPO 组内相对优势:

            A_i = (r_i - mean(group)) / (std(group) + eps)

        每个 prompt 的 group 内独立归一化, 消除 prompt 难度差异对梯度的影响。
        group_size=1 时直接返回 [0.0] (无法比较)。
        """
        advantages_by_group = []
        for group in rewards_by_group:
            if len(group) < 2:
                advantages_by_group.append([0.0] * len(group))
                continue
            mean_r = statistics.mean(group)
            std_r = statistics.stdev(group)
            adv = [(r - mean_r) / (std_r + self.grpo_eps) for r in group]
            advantages_by_group.append(adv)
        return advantages_by_group

    def _grpo_update(
        self,
        prompts: list[str],
        completions_grouped: list[list[str]],
        advantages_by_group: list[list[float]],
    ) -> None:
        """通过 OpenRLHF Actor Group 执行 GRPO policy gradient 更新。

        训练步骤:
          1. 把 prompts/completions/advantages flatten 成 experience batch
          2. Actor group (DeepSpeed ZeRO-3) 异步执行 training_step
          3. 内部: log_prob 计算 → GRPO loss → KL penalty → backward → optimizer step

        OpenRLHF ActorModelRayActor.training_step 接口:
          experience = {
              "prompts":     List[str],
              "completions": List[str],
              "advantages":  List[float],
          }
        """
        experience = self._build_experience_batch(
            prompts, completions_grouped, advantages_by_group
        )
        train_refs = self._actor_group.async_run_method(
            method_name="training_step",
            experience=experience,
            kl_coeff=self.kl_coeff,
            clip_eps=self.clip_eps,
        )
        ray.get(train_refs)

    @staticmethod
    def _build_experience_batch(
        prompts: list[str],
        completions_grouped: list[list[str]],
        advantages_by_group: list[list[float]],
    ) -> dict[str, Any]:
        """Flatten prompts/completions/advantages 为 experience batch dict。"""
        flat_prompts: list[str] = []
        flat_completions: list[str] = []
        flat_advantages: list[float] = []

        for prompt, comp_group, adv_group in zip(
            prompts, completions_grouped, advantages_by_group
        ):
            for completion, advantage in zip(comp_group, adv_group):
                flat_prompts.append(prompt)
                flat_completions.append(completion)
                flat_advantages.append(advantage)

        return {
            "prompts": flat_prompts,
            "completions": flat_completions,
            "advantages": flat_advantages,
        }

    def _store_top_payloads(self, flat_results: list[dict]) -> list[Payload]:
        """将本轮 top-K payload 存入 MAPElitesDB。

        strategy_db.add_payload() 通过 payload.episode_result.total_reward
        读取 reward，因此在入库前先把 oracle_reward 回填到 payload.episode_result。
        """
        from core.types import EpisodeResult

        sorted_results = sorted(flat_results, key=lambda r: r["reward"], reverse=True)
        top_payloads: list[Payload] = []

        for result in sorted_results[: self.top_k_save]:
            payload: Optional[Payload] = result.get("payload")
            oracle_reward: Optional[OracleReward] = result.get("oracle_reward")
            if payload is None or oracle_reward is None:
                continue

            # 回填 episode_result，strategy_db._get_reward() 从这里读 reward
            if payload.episode_result is None:
                payload.episode_result = EpisodeResult(
                    payload_id=payload.id,
                    total_reward=oracle_reward.total_reward,
                    attack_success=oracle_reward.attack_success,
                )
            else:
                payload.episode_result.total_reward = oracle_reward.total_reward
                payload.episode_result.attack_success = oracle_reward.attack_success

            self.strategy_db.add_payload(payload)
            top_payloads.append(payload)

        return top_payloads

    def _compute_round_stats(
        self,
        flat_results: list[dict],
        elapsed: float,
    ) -> dict[str, Any]:
        """汇总本轮统计指标, 用于 WandB 和可视化。

        字段名与原 trl 版保持一致, 避免 co_evolution_controller 需要改动。
        """
        rewards = [r["reward"] for r in flat_results]
        successes = [r["attack_success"] for r in flat_results]
        ja = [r["judge_a_score"] for r in flat_results]
        jb = [r["judge_b_score"] for r in flat_results]
        jc = [r["judge_c_score"] for r in flat_results]

        asr = sum(successes) / len(successes) if successes else 0.0

        return {
            # 与 trl 版 key 保持一致
            "grpo_avg_reward": statistics.mean(rewards) if rewards else 0.0,
            "grpo_reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "grpo_num_episodes": len(flat_results),
            "grpo_train_steps": max(1, len(flat_results) // max(1, self.group_size)),
            "grpo_train_time_s": elapsed,
            "attack_success_rate": asr,
            "judge_a_trigger_rate": sum(1 for s in ja if s > 0.3) / len(ja) if ja else 0.0,
            "judge_b_trigger_rate": sum(1 for s in jb if s > 0.5) / len(jb) if jb else 0.0,
            "avg_payload_quality": statistics.mean(jc) if jc else 0.0,
            # 策略库多样性覆盖率
            "red_diversity_coverage": self.strategy_db.coverage(),
        }

    # ==============================================================
    #  WandB 监控
    # ==============================================================

    def _init_wandb(self) -> None:
        """初始化 WandB run, 记录集群 + 超参配置。"""
        if not self.use_wandb:
            logger.info("WandB disabled or not installed, skipping")
            return

        if not self.wandb_run_name:
            model_short = self.model_name.split("/")[-1]
            self.wandb_run_name = f"red-grpo-{model_short}"

        self._wandb_run = wandb.init(
            project=self.wandb_project,
            name=self.wandb_run_name,
            tags=self.wandb_tags,
            config={
                "model_name": self.model_name,
                "backend": "openrlhf+ray+vllm",
                "num_training_gpus": self.num_training_gpus,
                "num_inference_gpus": self.num_inference_gpus,
                "tensor_parallel_size": self.tensor_parallel_size,
                "zero_stage": self.zero_stage,
                "group_size": self.group_size,
                "kl_coeff": self.kl_coeff,
                "clip_eps": self.clip_eps,
                "lora_r": self.lora_r,
                "learning_rate": self.learning_rate,
                "max_gen_length": self.max_gen_length,
                "temperature": self.temperature,
            },
            reinit=True,
        )
        logger.info(
            "WandB initialized: project=%s, run=%s",
            self.wandb_project, self.wandb_run_name,
        )

    def _log_to_wandb(
        self,
        round_num: int,
        stats: dict[str, Any],
        top_payloads: list[Payload],
    ) -> None:
        """将一轮训练的统计数据记录到 WandB。

        记录内容:
          - 标量指标: reward, ASR, Judge 触发率, 多样性覆盖率
          - Top payload 文本表格: 人工抽查生成质量
        """
        if not self.use_wandb or self._wandb_run is None:
            return

        log_dict = {
            "train/avg_reward": stats["grpo_avg_reward"],
            "train/reward_std": stats["grpo_reward_std"],
            "train/num_episodes": stats["grpo_num_episodes"],
            "train/train_time_s": stats["grpo_train_time_s"],
            "attack/success_rate": stats["attack_success_rate"],
            "attack/judge_a_trigger_rate": stats["judge_a_trigger_rate"],
            "attack/judge_b_trigger_rate": stats["judge_b_trigger_rate"],
            "attack/avg_payload_quality": stats["avg_payload_quality"],
            "diversity/coverage": stats["red_diversity_coverage"],
            "round": round_num,
        }

        # Top payload 文本表格
        if top_payloads:
            table = wandb.Table(columns=["rank", "payload_text", "reward", "type", "stealth"])
            for rank, p in enumerate(top_payloads[:5], 1):
                reward = p.episode_result.total_reward if p.episode_result else 0.0
                inj_type = p.injection_type.name if p.injection_type else "N/A"
                stealth = p.stealth_level.name if p.stealth_level else "N/A"
                table.add_data(rank, p.code[:300], reward, inj_type, stealth)
            log_dict["payloads/top_samples"] = table

        wandb.log(log_dict, step=round_num)

    # ==============================================================
    #  训练可视化
    # ==============================================================

    def _plot_training_curves(self) -> None:
        """绘制 2×2 训练曲线并保存为 PNG。

        子图:
          左上: Avg Reward ± std
          右上: Attack Success Rate
          左下: Judge A / B 触发率
          右下: Strategy DB Diversity Coverage
        """
        if not _MPL_AVAILABLE or len(self._history) < 2:
            return

        rounds = [h["round"] for h in self._history]
        avg_rewards = [h["grpo_avg_reward"] for h in self._history]
        reward_stds = [h["grpo_reward_std"] for h in self._history]
        asrs = [h["attack_success_rate"] for h in self._history]
        ja_rates = [h["judge_a_trigger_rate"] for h in self._history]
        jb_rates = [h["judge_b_trigger_rate"] for h in self._history]
        coverages = [h["red_diversity_coverage"] for h in self._history]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Red Team GRPO Training Progress (OpenRLHF)", fontsize=14, fontweight="bold")

        # 左上: Avg Reward ± std
        ax = axes[0, 0]
        ax.plot(rounds, avg_rewards, "b-o", markersize=4, label="avg reward")
        upper = [r + s for r, s in zip(avg_rewards, reward_stds)]
        lower = [r - s for r, s in zip(avg_rewards, reward_stds)]
        ax.fill_between(rounds, lower, upper, alpha=0.2, color="blue")
        ax.set_xlabel("Round")
        ax.set_ylabel("Reward")
        ax.set_title("Average Reward (±1 std)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # 右上: ASR
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

        self._plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = self._plot_dir / "training_curves.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training curves saved to %s", plot_path)

        if self.use_wandb and self._wandb_run is not None:
            wandb.log({"plots/training_curves": wandb.Image(str(plot_path))})

    def plot_reward_distribution(self, round_num: int) -> None:
        """绘制单轮 reward 分布直方图 (按需调用, 不在主循环里)。"""
        if not _MPL_AVAILABLE or not self._history:
            return

        round_history = [h for h in self._history if h["round"] == round_num]
        if not round_history:
            return

        avg = round_history[-1]["grpo_avg_reward"]
        std = round_history[-1]["grpo_reward_std"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axvline(avg, color="red", linestyle="--", label=f"mean={avg:.3f}")
        ax.axvspan(avg - std, avg + std, alpha=0.15, color="red", label=f"±std={std:.3f}")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Density")
        ax.set_title(f"Reward Distribution — Round {round_num}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self._plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = self._plot_dir / f"reward_dist_round_{round_num:03d}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Reward distribution plot saved to %s", plot_path)

        if self.use_wandb and self._wandb_run is not None:
            wandb.log(
                {"plots/reward_distribution": wandb.Image(str(plot_path))},
                step=round_num,
            )
