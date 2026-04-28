"""Shared helpers for Blue Team defense-memory loading and retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_/-]{2,}")
_STOPWORDS = {
    "about",
    "after",
    "against",
    "algorithm",
    "assistant",
    "before",
    "between",
    "block",
    "code",
    "coding",
    "complete",
    "correct",
    "description",
    "final",
    "following",
    "function",
    "generate",
    "ignore",
    "include",
    "instructions",
    "model",
    "please",
    "python",
    "response",
    "return",
    "should",
    "solution",
    "task",
    "tests",
    "this",
    "tool",
    "using",
    "with",
    "write",
    "your",
}
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def tokenize_text(text: str) -> list[str]:
    tokens = [
        tok.lower()
        for tok in _TOKEN_RE.findall(normalize_text(text))
        if tok.lower() not in _STOPWORDS
    ]
    return tokens


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    counts: dict[str, int] = {}
    for token in tokenize_text(text):
        counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ordered[:limit]]


def stable_memory_key(parts: list[str]) -> str:
    payload = "|".join(part.strip().lower() for part in parts if part and part.strip())
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def risk_rank(risk_level: str) -> int:
    return _RISK_ORDER.get(str(risk_level or "").lower(), 0)


def load_defense_memory(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []

    memory_path = Path(path)
    if not memory_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with open(memory_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.lstrip("\ufeff").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def retrieve_relevant_memories(
    query_text: str,
    memory_rows: list[dict[str, Any]],
    *,
    top_k: int = 3,
    static_pattern_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    query_norm = normalize_text(query_text)
    query_tokens = set(tokenize_text(query_text))
    pattern_ids = set(static_pattern_ids or [])
    scored: list[tuple[float, dict[str, Any]]] = []

    for row in memory_rows:
        attack_signature = row.get("attack_signature") or {}
        entry_keywords = set(attack_signature.get("keywords") or [])
        high_risk_phrases = attack_signature.get("high_risk_phrases") or []
        matched_patterns = set(row.get("matched_patterns") or [])

        overlap = query_tokens & entry_keywords
        phrase_hits = [
            phrase for phrase in high_risk_phrases if normalize_text(phrase) in query_norm
        ]
        pattern_overlap = pattern_ids & matched_patterns

        score = float(len(overlap)) + 2.0 * len(phrase_hits) + 2.5 * len(pattern_overlap)
        if risk_rank(str(row.get("risk_level", "low"))) >= 2:
            score += 0.5

        if score <= 0:
            continue

        compact = {
            "memory_key": row.get("memory_key"),
            "risk_level": row.get("risk_level", "low"),
            "score": round(score, 2),
            "attack_signature": attack_signature,
            "matched_patterns": sorted(matched_patterns),
            "failure_mode": row.get("failure_mode"),
            "counter_strategy": row.get("counter_strategy"),
            "source_episode_keys": (row.get("source_episode_keys") or [])[:5],
        }
        scored.append((score, compact))

    scored.sort(
        key=lambda item: (
            -item[0],
            -risk_rank(str(item[1].get("risk_level", "low"))),
            str(item[1].get("memory_key") or ""),
        )
    )
    return [row for _, row in scored[: max(int(top_k), 0)]]
