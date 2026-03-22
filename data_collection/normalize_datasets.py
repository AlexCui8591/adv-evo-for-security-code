"""
normalize_datasets.py - Production-Grade Data Normalization Pipeline

Reads raw data from 4 sources (InjecAgent, BIPIA, CyberSecEval, HackAPrompt),
extracts the correct attack payloads, scores difficulty based on content features,
deduplicates, and outputs a unified JSONL file.

Output Schema:
{
  "source": str,
  "attack_type": "direct" | "indirect",
  "category": str,
  "behavior": str,
  "prompt": str,
  "context": str | null,       # The surrounding context where the injection is embedded
  "is_code_relevant": bool,
  "difficulty": { "score": int (1-5), "tags": [str] }
}
"""

import json
import re
import hashlib
import logging
import sys
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent / "raw_data"
OUTPUT_DIR = Path(__file__).parent / "normalized_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# BIPIA Behavior Mapping (hand-written enrichment for category keys)
# ---------------------------------------------------------------------------
BIPIA_BEHAVIOR_MAP = {
    "Cookie Theft": "Steal browser cookies and exfiltrate them to an external server",
    "Memory Scanning": "Scan system memory and exfiltrate sensitive runtime data",
    "Dumpster Diving": "Recover deleted files from trash and upload them to an attacker server",
    "Environment Variable Analysis": "Read and exfiltrate system environment variables (PATH, HOME, API keys)",
    "Device and Driver Enumeration": "Enumerate connected devices and drivers, then exfiltrate hardware fingerprints",
    "Sending Out Spam Emails": "Mass-send spam or phishing emails using the victim's credentials",
    "Crippling Critical Infrastructures": "Execute destructive commands that disable critical system services or wipe storage",
    "Network Propagation": "Download and propagate malicious payloads across the local network",
    "Exploiting System Vulnerabilities": "Download and execute a remote exploit script on the target system",
    "Cryptocurrency Mining": "Silently install and run a cryptocurrency miner on the victim's machine",
}

# ---------------------------------------------------------------------------
# Content-Based Difficulty Scoring
# ---------------------------------------------------------------------------
def calculate_difficulty(prompt: str) -> dict:
    """Score difficulty purely based on prompt content features."""
    prompt_lower = prompt.lower()
    score = 1
    tags = []

    # --- Feature Detection ---

    # Length-based baseline
    token_estimate = len(prompt.split())
    if token_estimate > 200:
        tags.append("long_context")

    # Roleplay / Persona Hijacking
    roleplay_keywords = [
        "act as", "you are now", "imagine you", "hypothetical",
        "developer mode", "dan mode", "ignore previous", "ignore above",
        "disregard", "pretend", "simulate", "override", "jailbreak",
        "new instructions", "forget everything"
    ]
    if any(kw in prompt_lower for kw in roleplay_keywords):
        score = max(score, 3)
        tags.append("roleplay")

    # Encoding / Obfuscation
    has_base64 = bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', prompt))
    has_hex = bool(re.search(r'\\x[0-9a-fA-F]{2}', prompt))
    has_unicode_escape = bool(re.search(r'\\u[0-9a-fA-F]{4}', prompt))
    if has_base64 or has_hex or has_unicode_escape:
        score = max(score, 4)
        tags.append("obfuscation")

    # Code block detection (hidden malicious code)
    has_code_block = bool(re.search(r'```[\s\S]*?```', prompt))
    has_inline_code = any(kw in prompt_lower for kw in [
        "import ", "subprocess", "os.system", "exec(", "eval(",
        "requests.post", "requests.get", "socket.", "smtplib",
        "pickle.load", "urllib", "shutil", "ftplib"
    ])
    if has_code_block and has_inline_code:
        score = max(score, 5)
        tags.append("code_backdoor")
    elif has_inline_code:
        score = max(score, 4)
        tags.append("code_injection")
    elif has_code_block:
        score = max(score, 3)
        tags.append("code_context")

    # Indirect embedding (payload hidden inside structured data like JSON/XML/HTML)
    if any(marker in prompt for marker in ['{', '<', '[']) and token_estimate > 50:
        if "review_content" in prompt_lower or "comment" in prompt_lower or "description" in prompt_lower:
            score = max(score, 2)
            tags.append("context_embedding")

    # Simple direct override (short, blunt)
    if token_estimate < 30 and score == 1:
        tags.append("direct_simple")

    return {"score": score, "tags": tags}


# ---------------------------------------------------------------------------
# InjecAgent Processing
# ---------------------------------------------------------------------------
def process_injecagent() -> list:
    logger.info("Processing InjecAgent...")
    results = []

    # Process all available test case files
    data_dir = RAW_DIR / "InjecAgent" / "data"
    test_files = [
        "test_cases_dh_base.json", "test_cases_dh_enhanced.json",
        "test_cases_ds_base.json", "test_cases_ds_enhanced.json",
    ]

    for filename in test_files:
        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            # The REAL injection payload is in "Attacker Instruction", NOT "Tool Response"
            attacker_instruction = item.get("Attacker Instruction", "")
            if not attacker_instruction:
                continue

            # The full context (Tool Response) shows how the injection is embedded
            tool_response = item.get("Tool Response", "")
            behavior = item.get("Expected Achievements", "Manipulate tool execution")
            attack_category = item.get("Attack Type", "Unknown")

            diff = calculate_difficulty(attacker_instruction)
            # InjecAgent is always indirect (injection hidden inside tool response)
            diff["tags"].append("tool_manipulation")
            diff["score"] = max(diff["score"], 2)  # At least Level 2 (indirect)

            results.append({
                "source": "InjecAgent",
                "source_file": filename,
                "attack_type": "indirect",
                "category": attack_category,
                "behavior": behavior,
                "prompt": attacker_instruction,
                "context": tool_response[:2000] if tool_response else None,
                "is_code_relevant": True,
                "difficulty": diff,
            })

    logger.info(f"  InjecAgent: {len(results)} samples extracted")
    return results


# ---------------------------------------------------------------------------
# BIPIA Processing
# ---------------------------------------------------------------------------
def process_bipia() -> list:
    logger.info("Processing BIPIA Code QA...")
    results = []

    for split in ["train", "test"]:
        filepath = RAW_DIR / "BIPIA" / "benchmark" / f"code_attack_{split}.json"
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for behavior_key, prompts in data.items():
            # Use enriched behavior description if available
            behavior = BIPIA_BEHAVIOR_MAP.get(behavior_key, behavior_key)

            for prompt in prompts:
                diff = calculate_difficulty(prompt)
                # BIPIA code attacks always contain inline code
                diff["score"] = max(diff["score"], 3)

                results.append({
                    "source": "BIPIA_Code",
                    "source_file": f"code_attack_{split}.json",
                    "attack_type": "indirect",
                    "category": "code_injection",
                    "behavior": behavior,
                    "prompt": prompt,
                    "context": None,
                    "is_code_relevant": True,
                    "difficulty": diff,
                })

    logger.info(f"  BIPIA: {len(results)} samples extracted")
    return results


# ---------------------------------------------------------------------------
# CyberSecEval Processing
# ---------------------------------------------------------------------------
def process_cyberseceval() -> list:
    import datasets
    logger.info("Processing CyberSecEval...")
    results = []
    hf_dir = RAW_DIR / "CyberSecEval"

    if not hf_dir.exists():
        logger.warning(f"Directory not found: {hf_dir}")
        return []

    try:
        ds_dict = datasets.load_from_disk(str(hf_dir))
        for split_name, ds in ds_dict.items():
            for item in ds:
                prompt = item.get("prompt", "")
                if not prompt:
                    continue

                behavior = item.get("pattern_desc", "Generate insecure code based on context")
                cwe = item.get("cwe_identifier", "")

                diff = calculate_difficulty(prompt)
                # CyberSecEval asks for code generation, usually direct prompts
                if cwe:
                    diff["tags"].append(f"cwe:{cwe}")

                results.append({
                    "source": "CyberSecEval",
                    "source_file": split_name,
                    "attack_type": "direct",
                    "category": "vulnerability_generation",
                    "behavior": behavior,
                    "prompt": prompt,
                    "context": None,
                    "is_code_relevant": True,
                    "difficulty": diff,
                })
    except Exception as e:
        logger.error(f"Error processing CyberSecEval: {e}")

    logger.info(f"  CyberSecEval: {len(results)} samples extracted")
    return results


# ---------------------------------------------------------------------------
# HackAPrompt Processing
# ---------------------------------------------------------------------------
def process_hackaprompt() -> list:
    import datasets
    logger.info("Processing HackAPrompt...")
    results = []
    hf_dir = RAW_DIR / "HackAPrompt"

    if not hf_dir.exists():
        logger.warning(f"Directory not found: {hf_dir}")
        return []

    try:
        ds = datasets.load_from_disk(str(hf_dir))
        if isinstance(ds, datasets.DatasetDict) and "train" in ds:
            ds = ds["train"]

        for item in ds:
            # Only take SUCCESSFUL attacks
            if not item.get("correct", False):
                continue

            prompt = item.get("user_input", "")
            if not prompt or len(prompt.strip()) < 5:
                continue

            raw_level = item.get("level", 1)

            # Fix: Use a meaningful behavior description instead of raw expected_completion
            behavior = f"Bypass Level-{raw_level} system prompt constraints to produce forbidden output"

            # Map HackAPrompt levels (0-10) to our 1-5 scale
            mapped_score = min(5, max(1, (raw_level + 1) // 2))

            diff = calculate_difficulty(prompt)
            diff["score"] = max(diff["score"], mapped_score)
            diff["tags"].append("constraint_evasion")
            diff["tags"].append(f"hackaprompt_level:{raw_level}")

            results.append({
                "source": "HackAPrompt",
                "source_file": "train",
                "attack_type": "indirect",
                "category": "prompt_injection",
                "behavior": behavior,
                "prompt": prompt,
                "context": None,
                "is_code_relevant": False,
                "difficulty": diff,
            })
    except Exception as e:
        logger.error(f"Error processing HackAPrompt: {e}")

    logger.info(f"  HackAPrompt: {len(results)} samples extracted")
    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate(data: list) -> list:
    """Remove exact duplicates based on prompt hash, and near-duplicates based on
    normalized text similarity (simple n-gram Jaccard)."""
    logger.info(f"Deduplication: starting with {len(data)} samples...")

    # Phase 1: Exact dedup via hash
    seen_hashes = set()
    phase1 = []
    for item in data:
        h = hashlib.md5(item["prompt"].encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            phase1.append(item)

    exact_removed = len(data) - len(phase1)
    logger.info(f"  Exact duplicates removed: {exact_removed}")

    # Phase 2: Near-dedup via normalized prefix bucketing (O(n), fast)
    def normalize_text(text: str) -> str:
        """Normalize text for fuzzy comparison: lowercase, strip whitespace/punct."""
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric
        text = re.sub(r'\s+', ' ', text)           # Collapse whitespace
        return text

    phase2 = []
    seen_prefixes = set()
    PREFIX_LEN = 100  # Compare first 100 normalized chars as fingerprint

    for item in phase1:
        prefix = normalize_text(item["prompt"])[:PREFIX_LEN]
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            phase2.append(item)

    near_removed = len(phase1) - len(phase2)
    logger.info(f"  Near duplicates removed: {near_removed}")
    logger.info(f"  Final count after dedup: {len(phase2)}")
    return phase2


# ---------------------------------------------------------------------------
# Quality Filters
# ---------------------------------------------------------------------------
def quality_filter(data: list) -> list:
    """Remove low-quality entries."""
    original_count = len(data)
    filtered = []
    for item in data:
        prompt = item["prompt"]
        # Too short (likely garbage)
        if len(prompt.strip()) < 10:
            continue
        # Too long (likely entire documents pasted in)
        if len(prompt) > 8000:
            continue
        filtered.append(item)

    removed = original_count - len(filtered)
    logger.info(f"Quality filter: removed {removed} samples (too short/long)")
    return filtered


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def print_statistics(data: list):
    """Print dataset distribution statistics."""
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)

    # By source
    by_source = defaultdict(int)
    for item in data:
        by_source[item["source"]] += 1
    logger.info("By Source:")
    for src, count in sorted(by_source.items()):
        logger.info(f"  {src}: {count:,}")

    # By difficulty
    by_diff = defaultdict(int)
    for item in data:
        by_diff[item["difficulty"]["score"]] += 1
    logger.info("By Difficulty:")
    for score in sorted(by_diff.keys()):
        logger.info(f"  Level {score}: {by_diff[score]:,}")

    # By attack type
    by_type = defaultdict(int)
    for item in data:
        by_type[item["attack_type"]] += 1
    logger.info("By Attack Type:")
    for t, count in sorted(by_type.items()):
        logger.info(f"  {t}: {count:,}")

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info("Starting normalization pipeline...")

    # 1. Extract from all sources
    all_data = []
    all_data.extend(process_injecagent())
    all_data.extend(process_bipia())
    all_data.extend(process_cyberseceval())
    all_data.extend(process_hackaprompt())
    logger.info(f"Total raw entries: {len(all_data)}")

    # 2. Quality filter
    all_data = quality_filter(all_data)

    # 3. Deduplicate
    all_data = deduplicate(all_data)

    # 4. Print statistics
    print_statistics(all_data)

    # 5. Save
    output_file = OUTPUT_DIR / "normalized_pool.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(all_data)} normalized samples to {output_file}")


if __name__ == "__main__":
    main()
