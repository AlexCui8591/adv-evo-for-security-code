"""Stage-1 offline GRPO runner backed by veRL."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_PACKAGE_ROOT = _HERE.parent.parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent
for _path in (_PROJECT_ROOT, _PACKAGE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from scripts.prepare_verl_stage1_data import DEFAULT_CONFIG, _deep_merge, prepare_datasets


def _load_yaml(path: str) -> dict[str, Any]:
    if not _YAML_AVAILABLE:
        raise RuntimeError("PyYAML is required to load --config. Install: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="David & Goliath - veRL Stage-1 GRPO runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tasks-path", type=str, default=None)
    parser.add_argument("--total-rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rollouts-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--reward-mode", type=str, choices=["rule", "judge_c", "hybrid"], default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--val-num-prompts", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--rollout-n", type=int, default=None)
    parser.add_argument("--n-gpus-per-node", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verl", action="store_true", help="Prepare data only; do not launch veRL.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def _setup_logging(level: str, output_dir: str) -> None:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stage1_verl_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.config:
        config = _deep_merge(config, _load_yaml(args.config))
        config["config"] = args.config
    if args.experiment_id:
        config["experiment_id"] = args.experiment_id
    if args.output_dir:
        config["output_dir"] = args.output_dir
    elif not config.get("output_dir") and config.get("experiment_id"):
        config["output_dir"] = str(Path("outputs") / str(config["experiment_id"]))
    if args.tasks_path:
        config["coding_tasks_path"] = args.tasks_path
    if args.total_rounds is not None:
        config["total_rounds"] = args.total_rounds
    if args.seed is not None:
        config["seed"] = args.seed
    if args.rollouts_path:
        config["rollouts_path"] = args.rollouts_path
    else:
        config["rollouts_path"] = str(Path(config["output_dir"]) / "rollouts" / "rollouts.jsonl")

    stage1_cfg = config.setdefault("stage1", {})
    verl_cfg = config.setdefault("verl", {})
    data_cfg = verl_cfg.setdefault("data", {})
    rollout_cfg = verl_cfg.setdefault("rollout", {})
    trainer_cfg = verl_cfg.setdefault("trainer", {})
    if args.model_path:
        verl_cfg["model_path"] = args.model_path
    if args.reward_mode:
        stage1_cfg["reward_mode"] = args.reward_mode
    if args.num_prompts is not None:
        stage1_cfg["num_prompts"] = args.num_prompts
    if args.val_num_prompts is not None:
        stage1_cfg["val_num_prompts"] = args.val_num_prompts
    if args.train_batch_size is not None:
        data_cfg["train_batch_size"] = args.train_batch_size
    if args.rollout_n is not None:
        rollout_cfg["n"] = args.rollout_n
    if args.n_gpus_per_node is not None:
        trainer_cfg["n_gpus_per_node"] = args.n_gpus_per_node
    return config


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, list):
        return value[0] if value else default
    return value


def _convert_verl_rollouts(raw_dir: Path, output_path: Path) -> int:
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"veRL rollout dump not found in {raw_dir}. "
            "Ensure trainer.rollout_data_dir is enabled and at least one train step ran."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[str] = set()
    with open(output_path, "w", encoding="utf-8") as out:
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    payload_code = str(_scalar(raw.get("payload_code"), raw.get("output") or "")).strip()
                    if not payload_code:
                        continue
                    payload_id = str(_scalar(raw.get("payload_id"), "")) or f"payload_{count:08d}"
                    task_id = str(_scalar(raw.get("task_id"), ""))
                    injection_position = str(_scalar(raw.get("injection_position"), "NATURAL_LANGUAGE"))
                    episode_key = f"{payload_id}:{task_id}"
                    if episode_key in seen:
                        continue
                    seen.add(episode_key)

                    score = float(_scalar(raw.get("score"), 0.0) or 0.0)
                    quality = float(_scalar(raw.get("payload_quality_score"), 0.0) or 0.0)
                    stealth = float(_scalar(raw.get("stealth_score"), 0.0) or 0.0)
                    creativity = float(_scalar(raw.get("creativity_score"), 0.0) or 0.0)
                    row = {
                        "round": int(_scalar(raw.get("round"), raw.get("step") or 0) or 0),
                        "oracle_mode": "payload_only",
                        "payload_id": payload_id,
                        "payload_code": payload_code,
                        "prompt_used": str(_scalar(raw.get("input"), "")),
                        "task_id": task_id,
                        "episode_key": episode_key,
                        "injection_position": injection_position,
                        "injection_type": str(_scalar(raw.get("inferred_injection_type"), "")),
                        "stealth_level": str(_scalar(raw.get("inferred_stealth_level"), "")),
                        "judge_c": {
                            "payload_quality_score": quality,
                            "stealth_score": stealth,
                            "creativity_score": creativity,
                            "reasoning": str(_scalar(raw.get("judge_c_reasoning"), "")),
                            "wall_time_ms": float(_scalar(raw.get("judge_c_wall_time_ms"), 0.0) or 0.0),
                            "inferred_injection_type": str(_scalar(raw.get("inferred_injection_type"), "")),
                            "inferred_stealth_level": str(_scalar(raw.get("inferred_stealth_level"), "")),
                        },
                        "oracle_reward": {
                            "attack_success": False,
                            "total_reward": score,
                            "vulnerability_reward": 0.0,
                            "manipulation_reward": 0.0,
                            "quality_reward": quality,
                            "stealth_reward": stealth,
                            "diversity_bonus": creativity,
                        },
                        "reward": score,
                        "attack_success": False,
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
    return count


def _override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        rendered = "True" if value else "False"
    elif isinstance(value, list):
        rendered = "[" + ",".join(str(v) for v in value) + "]"
    elif value is None:
        rendered = "null"
    else:
        rendered = str(value)
    return f"{key}={rendered}"


def _build_verl_cmd(config: dict[str, Any], train_path: Path, val_path: Path, rollout_data_dir: Path) -> list[str]:
    verl_cfg = config.get("verl", {})
    data_cfg = verl_cfg.get("data", {})
    model_cfg = verl_cfg.get("model", {})
    actor_cfg = verl_cfg.get("actor", {})
    rollout_cfg = verl_cfg.get("rollout", {})
    ref_cfg = verl_cfg.get("ref", {})
    trainer_cfg = verl_cfg.get("trainer", {})

    reward_path = Path(verl_cfg.get("reward_path") or (_PACKAGE_ROOT / "red_team" / "verl_reward.py")).resolve()
    checkpoint_dir = Path(trainer_cfg.get("default_local_dir") or (Path(config["output_dir"]) / "verl_checkpoints")).resolve()

    overrides = [
        _override("algorithm.adv_estimator", "grpo"),
        _override("algorithm.use_kl_in_reward", False),
        _override("data.train_files", str(train_path.resolve())),
        _override("data.val_files", str(val_path.resolve())),
        _override("data.prompt_key", "prompt"),
        _override("data.max_prompt_length", data_cfg.get("max_prompt_length", 2048)),
        _override("data.max_response_length", data_cfg.get("max_response_length", 512)),
        _override("data.train_batch_size", data_cfg.get("train_batch_size", 64)),
        _override("data.shuffle", data_cfg.get("shuffle", True)),
        _override("data.seed", config.get("seed", 42)),
        _override("data.filter_overlong_prompts", data_cfg.get("filter_overlong_prompts", True)),
        _override("data.truncation", data_cfg.get("truncation", "left")),
        _override("data.trust_remote_code", data_cfg.get("trust_remote_code", True)),
        _override("actor_rollout_ref.model.path", verl_cfg.get("model_path", "Qwen/Qwen3-8B")),
        _override("actor_rollout_ref.model.trust_remote_code", verl_cfg.get("trust_remote_code", True)),
        _override("actor_rollout_ref.model.use_remove_padding", model_cfg.get("use_remove_padding", True)),
        _override("actor_rollout_ref.model.enable_gradient_checkpointing", actor_cfg.get("gradient_checkpointing", True)),
        _override("actor_rollout_ref.actor.optim.lr", actor_cfg.get("lr", "5e-6")),
        _override("actor_rollout_ref.actor.ppo_mini_batch_size", actor_cfg.get("ppo_mini_batch_size", 32)),
        _override("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", actor_cfg.get("micro_batch_size_per_gpu", 1)),
        _override("actor_rollout_ref.actor.use_kl_loss", actor_cfg.get("use_kl_loss", True)),
        _override("actor_rollout_ref.actor.kl_loss_coef", actor_cfg.get("kl_loss_coef", 0.001)),
        _override("actor_rollout_ref.actor.kl_loss_type", actor_cfg.get("kl_loss_type", "low_var_kl")),
        _override("actor_rollout_ref.actor.clip_ratio", actor_cfg.get("clip_ratio", 0.2)),
        _override("actor_rollout_ref.actor.ppo_epochs", actor_cfg.get("ppo_epochs", 1)),
        _override("actor_rollout_ref.actor.fsdp_config.param_offload", actor_cfg.get("param_offload", False)),
        _override("actor_rollout_ref.actor.fsdp_config.optimizer_offload", actor_cfg.get("optimizer_offload", False)),
        _override("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", ref_cfg.get("log_prob_micro_batch_size_per_gpu", 1)),
        _override("actor_rollout_ref.ref.fsdp_config.param_offload", ref_cfg.get("param_offload", True)),
        _override("actor_rollout_ref.rollout.name", rollout_cfg.get("name", "vllm")),
        _override("actor_rollout_ref.rollout.n", rollout_cfg.get("n", 8)),
        _override("actor_rollout_ref.rollout.temperature", rollout_cfg.get("temperature", 1.0)),
        _override("actor_rollout_ref.rollout.top_p", rollout_cfg.get("top_p", 0.95)),
        _override("actor_rollout_ref.rollout.tensor_model_parallel_size", rollout_cfg.get("tensor_model_parallel_size", 1)),
        _override("actor_rollout_ref.rollout.gpu_memory_utilization", rollout_cfg.get("gpu_memory_utilization", 0.55)),
        _override("actor_rollout_ref.rollout.max_num_batched_tokens", rollout_cfg.get("max_num_batched_tokens", 8192)),
        _override("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", rollout_cfg.get("log_prob_micro_batch_size_per_gpu", 1)),
        _override("reward_model.enable", False),
        _override("reward_model.reward_manager", verl_cfg.get("reward_manager", "naive")),
        _override("custom_reward_function.path", str(reward_path)),
        _override("custom_reward_function.name", "compute_score"),
        _override("trainer.project_name", trainer_cfg.get("project_name", "david_and_goliath")),
        _override("trainer.experiment_name", config.get("experiment_id", "verl_grpo_8b")),
        _override("trainer.logger", trainer_cfg.get("logger", ["console"])),
        _override("trainer.nnodes", trainer_cfg.get("nnodes", 1)),
        _override("trainer.n_gpus_per_node", trainer_cfg.get("n_gpus_per_node", 4)),
        _override("trainer.total_epochs", trainer_cfg.get("total_epochs", 1)),
        _override("trainer.save_freq", trainer_cfg.get("save_freq", 10)),
        _override("trainer.test_freq", trainer_cfg.get("test_freq", -1)),
        _override("trainer.val_before_train", trainer_cfg.get("val_before_train", False)),
        _override("trainer.default_local_dir", str(checkpoint_dir)),
        _override("++trainer.rollout_data_dir", str(rollout_data_dir.resolve())),
    ]
    overrides.extend(str(item) for item in verl_cfg.get("extra_overrides", []))

    return [sys.executable, "-m", verl_cfg.get("command_module", "verl.trainer.main_ppo"), *overrides]


def _build_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(_PROJECT_ROOT), str(_PACKAGE_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    oracle_cfg = config.get("oracle", {})
    if oracle_cfg.get("judge_model"):
        env["DG_JUDGE_MODEL"] = str(oracle_cfg["judge_model"])
    if oracle_cfg.get("api_key"):
        env["DG_JUDGE_API_KEY"] = str(oracle_cfg["api_key"])
    env.setdefault("DG_JUDGE_TEMPERATURE", str(oracle_cfg.get("judge_temperature", 0.1)))
    env.setdefault("DG_W_QUALITY", str(oracle_cfg.get("w_quality", 0.50)))
    env.setdefault("DG_W_STEALTH", str(oracle_cfg.get("w_stealth", 0.30)))
    env.setdefault("DG_W_CREATIVITY", str(oracle_cfg.get("w_creativity", oracle_cfg.get("w_diversity", 0.20))))

    stage1_cfg = config.get("stage1", {})
    env.setdefault("DG_REWARD_MODE", str(stage1_cfg.get("reward_mode", "rule")))
    env.setdefault("DG_REWARD_FAIL_FAST", str(stage1_cfg.get("reward_fail_fast", False)))

    runtime_defaults = {
        "HYDRA_FULL_ERROR": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_USE_V1": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    }
    runtime_defaults.update({str(k): str(v) for k, v in config.get("runtime", {}).items()})
    for key, value in runtime_defaults.items():
        env.setdefault(key, value)
    return env


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    _setup_logging(args.log_level, config["output_dir"])
    logger = logging.getLogger(__name__)

    output_dir = Path(config["output_dir"])
    data_dir = output_dir / "verl_data"
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    raw_rollout_dir = output_dir / "rollouts" / "verl_raw"
    rollouts_path = Path(config["rollouts_path"])

    cmd = _build_verl_cmd(config, train_path, val_path, raw_rollout_dir)
    logger.info("=" * 60)
    logger.info("Stage-1 veRL GRPO")
    logger.info("=" * 60)
    logger.info("Config       : %s", config.get("config") or "(defaults)")
    logger.info("Output dir   : %s", output_dir)
    logger.info("Train data   : %s", train_path)
    logger.info("Val data     : %s", val_path)
    logger.info("Raw rollouts : %s", raw_rollout_dir)
    logger.info("Rollouts     : %s", rollouts_path)
    logger.info("veRL command : %s", subprocess.list2cmdline(cmd))

    if args.dry_run:
        logger.info("--dry-run: exiting before data preparation and veRL launch.")
        return

    train_n, val_n = prepare_datasets(config, train_path, val_path)
    logger.info("Prepared veRL datasets: train=%d val=%d", train_n, val_n)

    if not args.skip_verl:
        raw_rollout_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True, env=_build_env(config))
    else:
        logger.info("--skip-verl: not launching veRL or converting rollouts.")
        return

    written = _convert_verl_rollouts(raw_rollout_dir, rollouts_path)
    logger.info("Converted %d veRL rollouts into %s", written, rollouts_path)


if __name__ == "__main__":
    main()
