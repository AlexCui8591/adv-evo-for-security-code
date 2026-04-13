"""scripts/prepare_data.py — 下载并格式化 HumanEval + MBPP 数据

自动完成:
  1. 从 HuggingFace 下载 openai_humaneval 和 google-research-datasets/mbpp
  2. 转化为 CodingTask 格式 → data/coding_tasks/tasks.jsonl
  3. 提取 reference solution → data/benign/humaneval_mbpp.jsonl
  4. 从 normalized_pool.jsonl 过滤 is_code_relevant=True → data/red_seed_payloads.jsonl

用法:
  python -m david_and_goliath.scripts.prepare_data
  # 或者
  python david_and_goliath/scripts/prepare_data.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # david_and_goliath/
DATA_DIR = PROJECT_ROOT / "data"
CODING_TASKS_FILE = DATA_DIR / "coding_tasks" / "tasks.jsonl"
BENIGN_FILE = DATA_DIR / "benign" / "humaneval_mbpp.jsonl"
RED_SEED_FILE = DATA_DIR / "red_seed_payloads.jsonl"

# 外部数据路径 (normalized_pool)
OUTER_PROJECT = PROJECT_ROOT.parent                             # project/
NORMALIZED_POOL = OUTER_PROJECT / "data_collection" / "normalized_data" / "normalized_pool.jsonl"


# ============================================================
# Step 1: 下载并处理 HumanEval
# ============================================================

def process_humaneval() -> list[dict]:
    """下载 openai_humaneval, 转化为 CodingTask 格式"""
    from datasets import load_dataset

    logger.info("Downloading openai_humaneval from HuggingFace...")
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    logger.info(f"  HumanEval loaded: {len(ds)} problems")

    tasks = []
    for item in ds:
        task_id = item["task_id"]            # e.g. "HumanEval/0"
        prompt = item["prompt"]              # 函数签名 + docstring
        solution = item["canonical_solution"]  # 参考解法 (函数体)
        test_code = item["test"]             # 测试函数
        entry_point = item["entry_point"]    # 函数名

        # 构造完整的任务描述
        description = (
            f"Complete the following Python function.\n\n"
            f"```python\n{prompt}```\n\n"
            f"The function should be named `{entry_point}`."
        )

        # 完整的参考解法 = prompt + canonical_solution
        full_solution = prompt + solution

        # 从 test 代码中提取 assert 语句作为 test_cases
        test_cases = _extract_asserts(test_code)

        tasks.append({
            "id": task_id.replace("/", "_").lower(),  # humaneval_0
            "source": "humaneval",
            "description": description,
            "reference_solution": full_solution,
            "test_cases": test_cases,
            "test_code": test_code,
            "entry_point": entry_point,
            "difficulty": _estimate_difficulty_humaneval(prompt, solution),
            "tags": ["humaneval", "function_completion"],
        })

    logger.info(f"  HumanEval processed: {len(tasks)} tasks")
    return tasks


def _extract_asserts(test_code: str) -> list[str]:
    """从测试代码中提取 assert 语句"""
    asserts = []
    for line in test_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("assert "):
            asserts.append(stripped)
    # 如果没有独立 assert，就保留整个 test_code 作为一个 test case
    if not asserts and test_code.strip():
        asserts = [test_code.strip()]
    return asserts


def _estimate_difficulty_humaneval(prompt: str, solution: str) -> str:
    """基于代码长度粗估难度"""
    sol_lines = len(solution.strip().split("\n"))
    if sol_lines <= 3:
        return "easy"
    elif sol_lines <= 10:
        return "medium"
    else:
        return "hard"


# ============================================================
# Step 2: 下载并处理 MBPP
# ============================================================

def process_mbpp() -> list[dict]:
    """下载 google-research-datasets/mbpp, 转化为 CodingTask 格式"""
    from datasets import load_dataset

    logger.info("Downloading google-research-datasets/mbpp from HuggingFace...")
    # MBPP 有 sanitized 版本 (更干净), 默认全量加载
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test",
                          trust_remote_code=True)
        subset_name = "sanitized"
    except Exception:
        ds = load_dataset("google-research-datasets/mbpp", "full", split="test",
                          trust_remote_code=True)
        subset_name = "full"

    logger.info(f"  MBPP ({subset_name}) loaded: {len(ds)} problems")

    tasks = []
    for item in ds:
        task_id = item.get("task_id", item.get("qid", 0))
        text = item.get("text", item.get("prompt", ""))       # 任务描述
        code = item.get("code", item.get("solution", ""))     # 参考解法
        test_list = item.get("test_list", [])                 # assert 列表

        if not text:
            continue

        description = (
            f"Write a Python function to solve the following task:\n\n"
            f"{text}"
        )

        tasks.append({
            "id": f"mbpp_{task_id}",
            "source": "mbpp",
            "description": description,
            "reference_solution": code,
            "test_cases": test_list,
            "test_code": "\n".join(test_list),
            "entry_point": _guess_entry_point(code),
            "difficulty": _estimate_difficulty_mbpp(code),
            "tags": ["mbpp", "function_writing"],
        })

    logger.info(f"  MBPP processed: {len(tasks)} tasks")
    return tasks


def _guess_entry_point(code: str) -> str:
    """从代码中猜测主函数名"""
    import re
    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else "solution"


def _estimate_difficulty_mbpp(code: str) -> str:
    """基于代码长度粗估难度"""
    lines = len(code.strip().split("\n"))
    if lines <= 5:
        return "easy"
    elif lines <= 15:
        return "medium"
    else:
        return "hard"


# ============================================================
# Step 3: 过滤 Red Team 种子 payload
# ============================================================

def filter_red_seed_payloads() -> list[dict]:
    """从 normalized_pool.jsonl 过滤出 is_code_relevant=True 的记录"""
    if not NORMALIZED_POOL.exists():
        logger.warning(f"normalized_pool.jsonl not found at {NORMALIZED_POOL}, skipping red seed extraction")
        return []

    logger.info(f"Reading {NORMALIZED_POOL}...")
    total = 0
    code_relevant = []

    with open(NORMALIZED_POOL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            d = json.loads(line)
            if d.get("is_code_relevant", False):
                code_relevant.append(d)

    logger.info(f"  Total: {total}, code_relevant: {len(code_relevant)}")
    return code_relevant


# ============================================================
# Step 4: 保存
# ============================================================

def save_jsonl(data: list[dict], path: Path, description: str):
    """保存为 JSONL 格式"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"  ✅ Saved {len(data)} {description} → {path}")


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("David & Goliath — Data Preparation Pipeline")
    logger.info("=" * 60)

    # --- Coding Tasks (for Blue Team + InjectionEngine) ---
    logger.info("\n[1/3] Preparing coding tasks (HumanEval + MBPP)...")
    humaneval_tasks = process_humaneval()
    mbpp_tasks = process_mbpp()
    all_tasks = humaneval_tasks + mbpp_tasks

    save_jsonl(all_tasks, CODING_TASKS_FILE, "coding tasks")

    # --- Benign Code (for FPR evaluation) ---
    logger.info("\n[2/3] Preparing benign code (reference solutions)...")
    benign_records = []
    for task in all_tasks:
        if task["reference_solution"].strip():
            benign_records.append({
                "id": task["id"],
                "source": task["source"],
                "code": task["reference_solution"],
                "test_cases": task["test_cases"],
                "entry_point": task.get("entry_point", ""),
                "is_benign": True,
            })

    save_jsonl(benign_records, BENIGN_FILE, "benign code samples")

    # --- Red Team Seed Payloads ---
    logger.info("\n[3/3] Filtering red team seed payloads...")
    red_seeds = filter_red_seed_payloads()
    if red_seeds:
        save_jsonl(red_seeds, RED_SEED_FILE, "red team seed payloads")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Coding tasks:     {len(all_tasks):>6}  ({CODING_TASKS_FILE})")
    logger.info(f"    - HumanEval:    {len(humaneval_tasks):>6}")
    logger.info(f"    - MBPP:         {len(mbpp_tasks):>6}")
    logger.info(f"  Benign code:      {len(benign_records):>6}  ({BENIGN_FILE})")
    logger.info(f"  Red seed payload: {len(red_seeds):>6}  ({RED_SEED_FILE})")

    # Difficulty distribution
    from collections import Counter
    diff_dist = Counter(t["difficulty"] for t in all_tasks)
    logger.info(f"\n  Difficulty distribution:")
    for d in ["easy", "medium", "hard"]:
        logger.info(f"    {d}: {diff_dist.get(d, 0)}")

    logger.info("\n✅ Data preparation complete! Ready for co-evolution.")


if __name__ == "__main__":
    main()
