"""I/O helpers: JSONL, CSV, text normalization, and token counting."""

from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSONL / CSV
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(
    text: str,
    lowercase: bool = False,
    collapse_whitespace: bool = True,
    strip_wrapping_quotes: bool = True,
    normalize_unicode: bool = True,
) -> str:
    if text is None:
        return ""
    value = text
    if normalize_unicode:
        value = unicodedata.normalize("NFKC", value)
    if strip_wrapping_quotes:
        value = value.strip().strip("\"'`")
    if collapse_whitespace:
        value = re.sub(r"\s+", " ", value).strip()
    if lowercase:
        value = value.lower()
    return value


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


class MetricTokenizer:
    """Wraps a HF tokenizer for metrics; falls back to char-based estimate."""

    def __init__(self, model_id: str | None):
        self.model_id = model_id
        self.tokenizer = None
        if model_id:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=True,
                )
            except Exception:
                self.tokenizer = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is None:
            return approx_token_count(text)
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return approx_token_count(text)
