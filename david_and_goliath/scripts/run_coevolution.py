"""scripts/run_coevolution.py — David & Goliath 主训练入口

用法:
  # From the repository root:
  python david_and_goliath/scripts/run_coevolution.py

  # 指定配置文件
  python david_and_goliath/scripts/run_coevolution.py --config david_and_goliath/configs/experiment/coevo_8b.yaml

  # 覆盖部分参数
  python david_and_goliath/scripts/run_coevolution.py \
      --config david_and_goliath/configs/experiment/coevo_8b.yaml \
      --experiment-id my_exp_v2 \
      --total-rounds 30 \
      --resume

流程:
  1. 解析 CLI 参数 → 合并到 config dict
  2. 初始化日志 / WandB
  3. CoEvolutionController(config).setup()   — 加载数据、构建所有组件、初始化 GRPO
  4. CoEvolutionController.run()             — N 轮 co-evolution 主循环
  5. 捕获 KeyboardInterrupt / SIGTERM → 优雅退出（保存 checkpoint）

Config 层次结构 (YAML):
  experiment_id, seed, total_rounds, checkpoint_every, output_dir
  coding_tasks_path, niche_capacity
  oracle:      {judge_model, api_key, bandit_enabled, semgrep_enabled, w_*}
  blue_team:   {model, api_key, base_url, temperature, max_turns, max_reflexion, use_tools}
  red_team:
    model_name, lora_path
    cluster:   {num_training_gpus, num_inference_gpus, tensor_parallel_size, num_reward_workers}
    deepspeed: {zero_stage, offload_optimizer, offload_param}
    vllm:      {gpu_memory_utilization, max_model_len, enforce_eager}
    grpo:      {group_size, kl_coeff, clip_eps, learning_rate, prompts_per_round, ...}
    lora:      {r, alpha, dropout}
    wandb:     {enabled, project, run_name, tags}
"""

from __future__ import annotations

import argparse
import copy
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

# ---- Import paths for both `python -m ...` and direct script execution ----
_HERE = Path(__file__).resolve()
_PACKAGE_ROOT = _HERE.parent.parent          # david_and_goliath/
_PROJECT_ROOT = _PACKAGE_ROOT.parent         # repo root
for _path in (_PROJECT_ROOT, _PACKAGE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# ---- 可选依赖: YAML ----
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from core.co_evolution_controller import CoEvolutionController

# ===========================================================================
# 默认配置（无 config 文件时使用）
# ===========================================================================

DEFAULT_CONFIG: dict[str, Any] = {
    # ---- 实验元信息 ----
    "experiment_id": "coevo_default",
    "seed": 42,
    "total_rounds": 20,
    "checkpoint_every": 5,
    "output_dir": "outputs/coevo_default",
    "coding_tasks_path": "david_and_goliath/data/coding_tasks/tasks.jsonl",
    "niche_capacity": 5,

    # ---- Hybrid Oracle ----
    "oracle": {
        "mode": "full",
        "judge_model": "gpt-4o-mini",
        "api_key": None,                  # 从 OPENAI_API_KEY 环境变量读取
        "base_url": None,                 # 可指向 OpenAI-compatible judge endpoint
        "bandit_enabled": True,
        "semgrep_enabled": False,         # 默认关，semgrep 需要额外安装
        "semgrep_rules": "p/security-audit",
        "judge_temperature": 0.1,
        "w_vulnerability": 0.30,
        "w_manipulation": 0.30,
        "w_quality": 0.20,
        "w_diversity": 0.10,
        "w_stealth": 0.10,
        "failure_quality_scale": 0.15,
    },

    # ---- Blue Team ----
    "blue_team": {
        "model": "gpt-4o-mini",
        "api_key": None,
        "base_url": None,
        "temperature": 0.2,
        "max_turns": 6,
        "max_reflexion": 2,
        "use_tools": True,
    },

    # ---- Red Team ----
    "red_team": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "lora_path": "",                  # 空 = 从零初始化 LoRA

        "prompt_builder": {
            "n_parents": 2,
            "n_inspirations": 3,
            "max_payload_chars": 500,
        },

        # Ray 集群 GPU 分配
        "cluster": {
            "num_training_gpus": 2,
            "num_inference_gpus": 4,
            "tensor_parallel_size": 2,
            "num_reward_workers": 4,
        },

        # DeepSpeed ZeRO
        "deepspeed": {
            "zero_stage": 3,
            "offload_optimizer": False,
            "offload_param": False,
        },

        # vLLM 推理
        "vllm": {
            "gpu_memory_utilization": 0.90,
            "max_model_len": 2048,
            "enforce_eager": False,
        },

        # GRPO 超参
        "grpo": {
            "group_size": 8,
            "eps": 1e-8,
            "kl_coeff": 0.01,
            "clip_eps": 0.2,
            "max_gen_length": 512,
            "temperature": 0.9,
            "top_p": 0.95,
            "learning_rate": 5e-6,
            "prompts_per_round": 32,
            "top_k_save": 10,
        },

        # LoRA adapter
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
        },

        # WandB
        "wandb": {
            "enabled": True,
            "project": "david-and-goliath",
            "run_name": "",               # 空 = WandB 自动生成
            "tags": ["red-team", "grpo", "openrlhf"],
        },
    },
}


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="David & Goliath — Red Team Co-Evolution Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. Merged on top of DEFAULT_CONFIG.",
    )
    parser.add_argument(
        "--experiment-id", type=str, default=None,
        help="Override experiment_id (also used as output_dir suffix).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output_dir.",
    )
    parser.add_argument(
        "--total-rounds", type=int, default=None,
        help="Override total_rounds.",
    )
    parser.add_argument(
        "--tasks-path", type=str, default=None,
        help="Override coding_tasks_path.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--oracle-mode", type=str, default=None,
        choices=["full", "payload_only"],
        help="Override oracle.mode for online training.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved config and exit before touching any component.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ===========================================================================
# Config 加载 & 合并
# ===========================================================================

def _load_yaml(path: str) -> dict[str, Any]:
    if not _YAML_AVAILABLE:
        raise RuntimeError(
            "PyYAML is required to load a config file. "
            "Install with: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    # Deep copy so CLI/YAML overrides never mutate the module-level DEFAULT_CONFIG.
    config = copy.deepcopy(DEFAULT_CONFIG)

    # 1. Merge YAML file on top of defaults (load once, reuse).
    yaml_cfg: dict[str, Any] = {}
    if args.config:
        yaml_cfg = _load_yaml(args.config)
        config = _deep_merge(config, yaml_cfg)

    # 2. CLI overrides (highest priority)
    if args.experiment_id:
        config["experiment_id"] = args.experiment_id
        # Keep output_dir in sync unless explicitly set in YAML
        if args.output_dir is None and "output_dir" not in yaml_cfg:
            config["output_dir"] = f"outputs/{args.experiment_id}"

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    if args.total_rounds is not None:
        config["total_rounds"] = args.total_rounds

    if args.tasks_path is not None:
        config["coding_tasks_path"] = args.tasks_path

    if args.seed is not None:
        config["seed"] = args.seed

    if args.oracle_mode is not None:
        config["oracle"]["mode"] = args.oracle_mode

    return config


# ===========================================================================
# 日志
# ===========================================================================

def _setup_logging(level: str, output_dir: str) -> None:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=getattr(logging, level), format=fmt, handlers=handlers)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ray").setLevel(logging.WARNING)


# ===========================================================================
# 优雅退出
# ===========================================================================

class _GracefulExit:
    """SIGTERM / SIGINT → KeyboardInterrupt so the outer try/except in main()
    can run the emergency checkpoint path.

    First signal: raise KeyboardInterrupt (caught → checkpoint → exit).
    Second signal: restore the default handler so a subsequent signal force-kills.
    """

    def __init__(self) -> None:
        self._triggered = False
        # SIGTERM is not deliverable on Windows but signal.signal accepts it.
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handler)
            except (ValueError, OSError):
                # Some runtimes (e.g. non-main thread) reject signal installation.
                pass

    def _handler(self, signum, frame) -> None:  # noqa: ANN001
        logger = logging.getLogger(__name__)
        if self._triggered:
            logger.warning("Second signal %d — restoring default handler.", signum)
            signal.signal(signum, signal.SIG_DFL)
            return
        self._triggered = True
        logger.warning(
            "Signal %d received — stopping gracefully "
            "(press Ctrl-C again to force quit).",
            signum,
        )
        raise KeyboardInterrupt

    @property
    def triggered(self) -> bool:
        return self._triggered


# ===========================================================================
# 主函数
# ===========================================================================

def main() -> None:
    args = _parse_args()
    config = _build_config(args)

    _setup_logging(args.log_level, config["output_dir"])
    logger = logging.getLogger(__name__)

    # ---- 打印配置摘要 ----
    logger.info("=" * 60)
    logger.info("David & Goliath — Co-Evolution Training")
    logger.info("=" * 60)
    logger.info("Experiment : %s", config["experiment_id"])
    logger.info("Output dir : %s", config["output_dir"])
    logger.info("Rounds     : %d", config["total_rounds"])
    logger.info("Seed       : %d", config["seed"])
    logger.info("Oracle     : %s", config["oracle"]["mode"])
    logger.info("Red model  : %s", config["red_team"]["model_name"])
    logger.info("Blue model : %s", config["blue_team"]["model"])
    logger.info(
        "Cluster    : %d train GPU / %d inference GPU",
        config["red_team"]["cluster"]["num_training_gpus"],
        config["red_team"]["cluster"]["num_inference_gpus"],
    )
    if args.dry_run:
        logger.info("--dry-run: exiting before training.")
        return

    # ---- 初始化控制器 ----
    controller = CoEvolutionController(config)

    # Install signal handlers so Ctrl-C / SIGTERM trigger KeyboardInterrupt,
    # which the outer except block catches and converts to a clean checkpoint + exit.
    _GracefulExit()

    try:
        logger.info("Setting up components...")
        controller.setup()
        logger.info("Setup complete. Starting training.")

        controller.run()

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — saving checkpoint and exiting.")
        _emergency_checkpoint(controller, logger)

    except Exception as exc:
        logger.exception("Unhandled exception: %s", exc)
        _emergency_checkpoint(controller, logger)
        raise

    finally:
        _shutdown_ray(logger)

    logger.info("Training finished. Results: %s/results.json", config["output_dir"])


def _emergency_checkpoint(controller: CoEvolutionController, logger: logging.Logger) -> None:
    """尽力保存当前进度，失败也不抛异常。"""
    try:
        if controller.grpo_trainer is not None and controller.round_records:
            last_round = controller.round_records[-1].round
            ckpt_dir = controller.output_dir / "checkpoints" / f"emergency_round_{last_round:03d}"
            logger.info("Emergency checkpoint → %s", ckpt_dir)
            controller.grpo_trainer.save_checkpoint(ckpt_dir / "red_team")
            controller.strategy_db.save(ckpt_dir / "strategy_db.json")
    except Exception as exc:
        logger.warning("Emergency checkpoint failed: %s", exc)


def _shutdown_ray(logger: logging.Logger) -> None:
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray cluster shut down.")
    except Exception as exc:
        logger.warning("Ray shutdown error: %s", exc)

if __name__ == "__main__":
    main()
