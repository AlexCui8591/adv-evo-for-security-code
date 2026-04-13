"""core/strategy_db.py — MAP-Elites 策略数据库

质量-多样性 (Quality-Diversity) 存储结构，防止 Red Team GRPO 训练中的模式坍塌。

网格结构:
  行 = InjectionType  (4 种: DIRECT_PROMPT, INDIRECT_PROMPT, CODE_INJECTION, DATA_EXFILTRATION)
  列 = StealthLevel   (3 级: L1_OBVIOUS, L2_OBFUSCATED, L3_SEMANTIC_CAMOUFLAGE)
  → 4 × 3 = 12 个 niche，每个 niche 存 top-K payload (按 reward 排序)

使用者:
  - GRPOTrainer:    训练后 add_payload() 存入 top payload
  - RedPromptBuilder: sample() 跨 niche 采样 parents/inspirations 构建 few-shot prompt
  - HybridOracle:   查询 niche 填充状态计算 diversity bonus
  - Controller:     snapshot() 保存 / coverage 跟踪多样性覆盖率
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from core.types import (
    InjectionType,
    NicheKey,
    Payload,
    StealthLevel,
)

logger = logging.getLogger(__name__)


class MAPElitesDB:
    """MAP-Elites 策略数据库

    Args:
        niche_capacity: 每个 niche 保留的 top-K payload 数量
        seed: 随机种子
    """

    # 网格维度
    INJECTION_TYPES = list(InjectionType)   # 4 种
    STEALTH_LEVELS = list(StealthLevel)     # 3 级
    NUM_NICHES = len(INJECTION_TYPES) * len(STEALTH_LEVELS)  # 12

    def __init__(
        self,
        niche_capacity: int = 5,
        seed: Optional[int] = None,
    ):
        self.niche_capacity = niche_capacity
        self.rng = random.Random(seed)

        # 核心数据结构: NicheKey → 按 reward 降序排列的 payload 列表
        self.niche_grid: dict[NicheKey, list[Payload]] = {}
        for inj in self.INJECTION_TYPES:
            for st in self.STEALTH_LEVELS:
                self.niche_grid[(inj.value, st.value)] = []

        # 历史快照 (用于实验分析)
        self._snapshots: list[dict] = []

    # ----------------------------------------------------------
    # 写入
    # ----------------------------------------------------------

    def add_payload(self, payload: Payload) -> bool:
        """将 payload 放入对应 niche (根据 Judge C 推断的分类)

        如果 niche 未满 → 直接加入。
        如果 niche 已满 → 与最低分 payload 竞争，优胜者留下。

        Returns:
            True 如果 payload 被成功加入 (新 niche 或击败了已有的)
        """
        key = self._classify(payload)
        niche = self.niche_grid[key]
        reward = self._get_reward(payload)

        if len(niche) < self.niche_capacity:
            # niche 未满，直接加入
            niche.append(payload)
            niche.sort(key=self._get_reward, reverse=True)
            logger.debug(
                f"Added payload {payload.id} to niche {key} "
                f"({len(niche)}/{self.niche_capacity})"
            )
            return True

        # niche 已满 → 和最低分竞争
        worst_reward = self._get_reward(niche[-1])
        if reward > worst_reward:
            evicted = niche.pop()
            niche.append(payload)
            niche.sort(key=self._get_reward, reverse=True)
            logger.debug(
                f"Payload {payload.id} (reward={reward:.3f}) "
                f"evicted {evicted.id} (reward={worst_reward:.3f}) "
                f"in niche {key}"
            )
            return True

        return False

    # ----------------------------------------------------------
    # 采样 (RedPromptBuilder 用)
    # ----------------------------------------------------------

    def sample(
        self,
        n_parents: int = 2,
        n_inspirations: int = 3,
    ) -> tuple[list[Payload], list[Payload]]:
        """从策略库采样 parents 和 inspirations

        采样策略: 偏向欠探索的 niche (少 payload 的 niche 采样概率更高)

        Args:
            n_parents: 从高分 payload 中采样的「父代」数量 (用于 prompt 中的优秀示例)
            n_inspirations: 从多样 niche 中采样的「灵感」数量 (探索新方向)

        Returns:
            (parents, inspirations) — 两组 payload
        """
        non_empty = [
            (key, payloads)
            for key, payloads in self.niche_grid.items()
            if payloads
        ]

        if not non_empty:
            return [], []

        # --- Parents: 全局 top-K ---
        all_payloads = []
        for _, payloads in non_empty:
            all_payloads.extend(payloads)
        all_payloads.sort(key=self._get_reward, reverse=True)
        parents = all_payloads[:min(n_parents, len(all_payloads))]

        # --- Inspirations: 偏向欠探索 niche 采样 ---
        # 权重 = 1 / (niche_size + 1)，空 niche 不参与 (没有 payload 可采)
        weights = [1.0 / (len(payloads) + 1) for _, payloads in non_empty]
        inspirations: list[Payload] = []

        for _ in range(min(n_inspirations, len(non_empty))):
            chosen_key, chosen_payloads = self.rng.choices(
                non_empty, weights=weights, k=1
            )[0]
            inspirations.append(self.rng.choice(chosen_payloads))

        return parents, inspirations

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    @property
    def coverage(self) -> float:
        """niche 覆盖率: 有至少 1 个 payload 的 niche 数 / 总 niche 数"""
        filled = sum(1 for v in self.niche_grid.values() if v)
        return filled / self.NUM_NICHES

    @property
    def total_payloads(self) -> int:
        """策略库中的总 payload 数"""
        return sum(len(v) for v in self.niche_grid.values())

    def niche_size(self, key: NicheKey) -> int:
        """查询某个 niche 的当前大小"""
        return len(self.niche_grid.get(key, []))

    def best_per_niche(self) -> dict[NicheKey, Optional[Payload]]:
        """每个 niche 的最佳 payload"""
        result = {}
        for key, payloads in self.niche_grid.items():
            result[key] = payloads[0] if payloads else None
        return result

    def niche_stats(self) -> dict[str, any]:
        """策略库统计信息 (用于日志/WandB)"""
        sizes = [len(v) for v in self.niche_grid.values()]
        rewards = [
            self._get_reward(p)
            for v in self.niche_grid.values()
            for p in v
        ]

        return {
            "strategy_db/coverage": self.coverage,
            "strategy_db/total_payloads": self.total_payloads,
            "strategy_db/filled_niches": sum(1 for s in sizes if s > 0),
            "strategy_db/full_niches": sum(
                1 for s in sizes if s >= self.niche_capacity
            ),
            "strategy_db/avg_niche_size": (
                sum(sizes) / len(sizes) if sizes else 0
            ),
            "strategy_db/avg_reward": (
                sum(rewards) / len(rewards) if rewards else 0
            ),
            "strategy_db/max_reward": max(rewards) if rewards else 0,
        }

    # ----------------------------------------------------------
    # 快照 & 恢复
    # ----------------------------------------------------------

    def snapshot(self, round_num: int) -> dict:
        """保存当前网格状态的快照

        Returns:
            快照 dict (同时保存到内部历史)
        """
        snap = {
            "round": round_num,
            "coverage": self.coverage,
            "total_payloads": self.total_payloads,
            "niches": {},
        }

        for key, payloads in self.niche_grid.items():
            niche_name = self._key_to_name(key)
            snap["niches"][niche_name] = {
                "count": len(payloads),
                "best_reward": (
                    self._get_reward(payloads[0]) if payloads else 0
                ),
                "payload_ids": [p.id for p in payloads],
            }

        self._snapshots.append(snap)
        logger.info(
            f"Strategy DB snapshot round={round_num}: "
            f"coverage={self.coverage:.1%}, "
            f"total={self.total_payloads}"
        )
        return snap

    def save(self, path: str | Path) -> None:
        """将策略库完整序列化到 JSON 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "niche_capacity": self.niche_capacity,
            "niches": {},
        }
        for key, payloads in self.niche_grid.items():
            niche_name = self._key_to_name(key)
            data["niches"][niche_name] = [
                {
                    "id": p.id,
                    "code": p.code,
                    "round_created": p.round_created,
                    "reward": self._get_reward(p),
                }
                for p in payloads
            ]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Strategy DB saved to {path}")

    def load(self, path: str | Path) -> None:
        """从 JSON 文件恢复策略库"""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Snapshot file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.niche_capacity = data.get("niche_capacity", self.niche_capacity)

        for niche_name, payload_dicts in data.get("niches", {}).items():
            key = self._name_to_key(niche_name)
            if key is None:
                continue
            self.niche_grid[key] = []
            for pd in payload_dicts:
                p = Payload(
                    id=pd["id"],
                    code=pd["code"],
                    round_created=pd.get("round_created", 0),
                )
                # 将 reward 存入 episode_result 的 total_reward
                from core.types import EpisodeResult
                p.episode_result = EpisodeResult(
                    total_reward=pd.get("reward", 0.0)
                )
                self.niche_grid[key].append(p)

        logger.info(
            f"Strategy DB loaded from {path}: "
            f"coverage={self.coverage:.1%}, total={self.total_payloads}"
        )

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _classify(self, payload: Payload) -> NicheKey:
        """根据 payload 上的分类信息确定 niche 坐标

        分类来源 (优先级):
        1. payload 自身的 injection_type / stealth_level (由 Judge C 标注)
        2. episode_result 中 Judge C 推断的类型
        3. 默认值 (DIRECT_PROMPT, L1_OBVIOUS)
        """
        # 尝试从 payload 字段获取
        inj_type = payload.injection_type
        stealth = payload.stealth_level

        # 尝试从 Judge C 推断结果获取
        if payload.episode_result and payload.episode_result.judge_c:
            jc = payload.episode_result.judge_c
            if inj_type is None and jc.inferred_injection_type:
                try:
                    inj_type = InjectionType[
                        jc.inferred_injection_type.upper()
                    ]
                except KeyError:
                    pass
            if stealth is None and jc.inferred_stealth_level:
                try:
                    stealth = StealthLevel[
                        jc.inferred_stealth_level.upper()
                    ]
                except KeyError:
                    pass
            # 也可以从 stealth_score 推断 stealth level
            if stealth is None:
                stealth = StealthLevel(
                    min(int(jc.stealth_score * 3), 2)
                )

        # 默认值
        if inj_type is None:
            inj_type = InjectionType.DIRECT_PROMPT
        if stealth is None:
            stealth = StealthLevel.L1_OBVIOUS

        # 回写到 payload (方便后续使用)
        payload.injection_type = inj_type
        payload.stealth_level = stealth

        return (inj_type.value, stealth.value)

    @staticmethod
    def _get_reward(payload: Payload) -> float:
        """获取 payload 的 reward 值"""
        if payload.episode_result is not None:
            return payload.episode_result.total_reward
        return 0.0

    @staticmethod
    def _key_to_name(key: NicheKey) -> str:
        """NicheKey → 可读字符串 (用于 JSON 序列化)"""
        inj = InjectionType(key[0]).name
        st = StealthLevel(key[1]).name
        return f"{inj}__{st}"

    @staticmethod
    def _name_to_key(name: str) -> Optional[NicheKey]:
        """可读字符串 → NicheKey (用于反序列化)"""
        parts = name.split("__")
        if len(parts) != 2:
            return None
        try:
            inj = InjectionType[parts[0]].value
            st = StealthLevel[parts[1]].value
            return (inj, st)
        except KeyError:
            return None

    # ----------------------------------------------------------
    # 调试 / 可视化
    # ----------------------------------------------------------

    def display_grid(self) -> str:
        """生成策略库的文本网格表示 (用于日志打印)"""
        header = "            " + "  ".join(
            f"{st.name:>22}" for st in self.STEALTH_LEVELS
        )
        lines = [header]

        for inj in self.INJECTION_TYPES:
            row_cells = []
            for st in self.STEALTH_LEVELS:
                key = (inj.value, st.value)
                payloads = self.niche_grid[key]
                n = len(payloads)
                best = self._get_reward(payloads[0]) if payloads else 0
                cell = f"{n}/{self.niche_capacity} (best={best:.2f})"
                row_cells.append(f"{cell:>22}")
            lines.append(f"{inj.name:<12}" + "  ".join(row_cells))

        return "\n".join(lines)
