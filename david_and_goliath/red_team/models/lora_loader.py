"""red_team/models/lora_loader.py — LoRA 模型加载工具

职责:
  1. 加载 base model (Qwen2.5-7B/14B-Instruct 等) + tokenizer
  2. 配置并注入 LoRA adapter (通过 peft)
  3. 提供 checkpoint 保存/恢复接口

依赖:
  - transformers: 基座模型和 tokenizer
  - peft: LoRA adapter 配置与注入
  - torch: 模型权重操作

使用:
  loader = LoRAModelLoader(model_name="Qwen/Qwen2.5-7B-Instruct", ...)
  model, tokenizer = loader.load_model()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class LoRAModelLoader:
    """LoRA 模型加载器

    封装 base model + LoRA adapter 的加载、初始化、保存流程。
    支持从零初始化 LoRA 或从已有 checkpoint 恢复。

    Args:
        model_name: HuggingFace 模型名 (如 "Qwen/Qwen2.5-7B-Instruct")
        lora_path: 已有 LoRA adapter 路径 (空字符串 = 从零初始化)
        lora_r: LoRA 秩 (rank), 控制可训练参数量
        lora_alpha: LoRA 缩放因子, 通常设为 2*r
        lora_dropout: LoRA dropout 比例
        lora_target_modules: 注入 LoRA 的目标模块名 (None = 使用默认值)
        quantization: 量化方式 ("4bit" / "8bit" / None)
        torch_dtype: 模型精度 ("float16" / "bfloat16" / "float32")
        device_map: 设备分配策略 ("auto" = 自动分配到可用 GPU)
        trust_remote_code: 是否信任远程代码 (Qwen 模型需要)
    """

    # Qwen 系列模型的默认 LoRA 目标模块
    DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_path: str = "",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list[str]] = None,
        quantization: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.lora_path = lora_path
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or self.DEFAULT_TARGET_MODULES
        self.quantization = quantization
        self.torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

        # 加载后的引用
        self._model = None
        self._tokenizer = None

    # ===================== 公共 API =====================

    def load_model(self) -> tuple[PeftModel, AutoTokenizer]:
        """加载 base model + LoRA adapter + tokenizer

        流程:
          1. 加载 tokenizer (处理 padding 等特殊设置)
          2. 加载 base model (可选量化)
          3. 注入 LoRA adapter (从零初始化 或 从 checkpoint 恢复)

        Returns:
            (model, tokenizer) 元组
        """
        # ---- Tokenizer ----
        logger.info(f"Loading tokenizer: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            padding_side="left",  # 生成任务用左 padding
        )
        # 确保有 pad_token (部分模型没有)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # ---- Base Model ----
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
        }

        # 可选量化配置
        if self.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization (QLoRA)")
        elif self.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("Using 8-bit quantization")

        logger.info(f"Loading base model: {self.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # ---- LoRA Adapter ----
        if self.lora_path and Path(self.lora_path).exists():
            # 从已有 checkpoint 恢复 LoRA adapter
            logger.info(f"Loading LoRA adapter from checkpoint: {self.lora_path}")
            model = PeftModel.from_pretrained(
                base_model,
                self.lora_path,
                is_trainable=True,  # 需要继续训练
            )
        else:
            # 从零初始化 LoRA adapter
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias="none",
            )
            logger.info(
                f"Initializing new LoRA adapter: r={self.lora_r}, "
                f"alpha={self.lora_alpha}, targets={self.lora_target_modules}"
            )
            model = get_peft_model(base_model, lora_config)

        # 打印可训练参数统计
        trainable, total = self._count_parameters(model)
        logger.info(
            f"Model loaded: {total / 1e6:.1f}M total params, "
            f"{trainable / 1e6:.1f}M trainable ({trainable / total:.2%})"
        )

        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def save_adapter(self, save_path: str | Path) -> None:
        """保存 LoRA adapter 权重 (不保存 base model)

        只保存 LoRA 增量权重, 体积极小 (通常 < 100MB)。
        恢复时用 load_model(lora_path=save_path) 即可。

        Args:
            save_path: 保存目录路径
        """
        if self._model is None:
            raise RuntimeError("Model not loaded yet. Call load_model() first.")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)

        logger.info(f"LoRA adapter saved to {save_path}")

    # ===================== 内部方法 =====================

    @staticmethod
    def _count_parameters(model) -> tuple[int, int]:
        """统计可训练参数和总参数"""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total
