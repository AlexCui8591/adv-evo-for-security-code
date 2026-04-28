"""Blue Team tool: detect prompt injection via static signatures + defense memory."""

from __future__ import annotations

import re
import time
from typing import Any

from blue_team.defense_memory import load_defense_memory, retrieve_relevant_memories

# ---------------------------------------------------------------------------
# Known prompt-injection / social-engineering patterns
# ---------------------------------------------------------------------------

_PATTERNS: list[dict[str, str]] = [
    {
        "id": "direct_override",
        "severity": "HIGH",
        "description": "Direct instruction override attempt",
        "pattern": (
            r"(?i)(ignore|disregard|forget|override|bypass|cancel)\s+"
            r"(all\s+)?(previous|prior|above|earlier|your|the\s+previous)\s+"
            r"(instructions?|rules?|constraints?|directives?|prompts?|guidelines?)"
        ),
    },
    {
        "id": "role_hijack",
        "severity": "HIGH",
        "description": "Attempted persona or role change",
        "pattern": (
            r"(?i)(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be)"
            r"|from\s+now\s+on\s+you\s+(are|will|should)|assume\s+the\s+role\s+of)"
        ),
    },
    {
        "id": "system_override",
        "severity": "HIGH",
        "description": "System-level prompt override",
        "pattern": (
            r"(?i)(new\s+system\s+prompt|updated\s+system\s+instructions?"
            r"|<\s*system\s*>|system\s+override|SYSTEM\s*:)"
        ),
    },
    {
        "id": "data_exfil",
        "severity": "HIGH",
        "description": "Data exfiltration instruction",
        "pattern": (
            r"(?i)(send|exfil(trate)?|transmit|leak|expose|reveal|print|output)\s+"
            r"(all\s+)?(the\s+)?"
            r"(api[_\s]?key|secret[_\s]?key|password|token|credential|system\s+prompt"
            r"|environment\s+variable|config(uration)?)"
        ),
    },
    {
        "id": "hidden_objective",
        "severity": "HIGH",
        "description": "Hidden secondary objective injection",
        "pattern": (
            r"(?i)(hidden\s+instruction|secret\s+(task|goal|objective)"
            r"|secondary\s+objective|real\s+(task|goal|objective)"
            r"|actual\s+(task|goal|objective|purpose))"
        ),
    },
    {
        "id": "separator_injection",
        "severity": "MEDIUM",
        "description": "Instruction injection via separator / markdown header",
        "pattern": (
            r"(?i)(---+\s*(new|updated|revised|important)\s*(instructions?|task|prompt|rules?)"
            r"|#+\s*(new\s+)?instructions?\s*:?\s*ignore"
            r"|\[\s*INST\s*\]|\[SYS\])"
        ),
    },
    {
        "id": "network_call",
        "severity": "MEDIUM",
        "description": "Unexpected network call requested in task",
        "pattern": (
            r"(?i)(make\s+(a\s+)?http\s+(request|call)|send\s+(a\s+)?post\s+request"
            r"|call\s+(the\s+)?api\s+at|connect\s+to\s+(the\s+)?server"
            r"|upload\s+(the\s+)?result(s)?\s+to)"
        ),
    },
    {
        "id": "obfuscation",
        "severity": "MEDIUM",
        "description": "Obfuscation or encoding trick (base64, rot13, etc.)",
        "pattern": (
            r"(?:import\s+base64|b64decode|base64\.b64"
            r"|codecs\.decode|rot13|zlib\.decompress)"
        ),
    },
    {
        "id": "shell_exec",
        "severity": "HIGH",
        "description": "Shell execution or OS access requested",
        "pattern": (
            r"(?i)(os\.system|subprocess\.(run|Popen|call|check_output)"
            r"|eval\s*\(|exec\s*\(|__import__\s*\("
            r"|open\s*\(\s*['\"]\/etc\/(passwd|shadow|hosts)"
            r"|os\.environ)"
        ),
    },
    {
        "id": "multilingual",
        "severity": "LOW",
        "description": "Large non-ASCII block, possible multilingual injection",
        "pattern": r"[\u4e00-\u9fff\u0600-\u06ff\u0400-\u04ff]{15,}",
    },
    {
        "id": "meta_injection",
        "severity": "MEDIUM",
        "description": "Meta-level injection language detected",
        "pattern": (
            r"(?i)(jailbreak|prompt\s+injection|DAN\s+mode"
            r"|developer\s+mode|god\s+mode|unrestricted\s+mode"
            r"|do\s+anything\s+now)"
        ),
    },
]


class MemoryRetrievalTool:
    """Scan text for known prompt-injection patterns and historical attack memories."""

    name = "memory_retrieval"
    description = (
        "Search both static prompt-injection signatures and a dynamic defense "
        "memory built from prior attacks. Returns matched static patterns plus "
        "historical defense memories with risk levels and counter-strategies."
    )

    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "memory_retrieval",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": (
                            "Text to search, typically the full injected task description."
                        ),
                    }
                },
                "required": ["query_text"],
            },
        },
    }

    def __init__(
        self,
        extra_patterns: list[dict[str, str]] | None = None,
        *,
        defense_memory_path: str | None = None,
        defense_retrieval_top_k: int = 3,
        enable_static_scan: bool = True,
        enable_defense_retrieval: bool = True,
    ):
        all_patterns = _PATTERNS + (extra_patterns or [])
        self._compiled: list[tuple[dict[str, str], re.Pattern[str]]] = []
        for pattern in all_patterns:
            try:
                self._compiled.append((pattern, re.compile(pattern["pattern"])))
            except re.error:
                continue

        self.defense_memory_path = defense_memory_path
        self.defense_retrieval_top_k = max(int(defense_retrieval_top_k), 0)
        self.enable_static_scan = bool(enable_static_scan)
        self.enable_defense_retrieval = bool(enable_defense_retrieval)
        self._defense_memory_rows = load_defense_memory(defense_memory_path)

    def reload_defense_memory(self) -> None:
        if self.enable_defense_retrieval:
            self._defense_memory_rows = load_defense_memory(self.defense_memory_path)
        else:
            self._defense_memory_rows = []

    def query_static(self, query_text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        if not self.enable_static_scan:
            return {
                "matches": [],
                "match_count": 0,
                "static_match_count": 0,
                "high_severity": 0,
                "static_high_severity_count": 0,
                "medium_severity": 0,
                "low_severity": 0,
                "suspicious": False,
                "static_enabled": False,
                "summary": "Static prompt-injection scanning is disabled.",
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        matches: list[dict[str, str]] = []

        for pattern_def, regex in self._compiled:
            matched = regex.search(query_text)
            if matched:
                matches.append(
                    {
                        "id": pattern_def["id"],
                        "severity": pattern_def["severity"],
                        "description": pattern_def["description"],
                        "matched_text": matched.group(0)[:120],
                    }
                )

        high = sum(1 for row in matches if row["severity"] == "HIGH")
        medium = sum(1 for row in matches if row["severity"] == "MEDIUM")
        low = sum(1 for row in matches if row["severity"] == "LOW")

        return {
            "matches": matches,
            "match_count": len(matches),
            "static_match_count": len(matches),
            "high_severity": high,
            "static_high_severity_count": high,
            "medium_severity": medium,
            "low_severity": low,
            "suspicious": len(matches) > 0,
            "static_enabled": True,
            "summary": (
                f"Detected {len(matches)} static pattern(s): "
                f"{high} HIGH, {medium} MEDIUM, {low} LOW."
                if matches
                else "No known static injection patterns detected."
            ),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def query(self, query_text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        static_result = self.query_static(query_text)
        static_pattern_ids = [match["id"] for match in static_result["matches"]]
        if self.enable_defense_retrieval:
            retrieved_memories = retrieve_relevant_memories(
                query_text,
                self._defense_memory_rows,
                top_k=self.defense_retrieval_top_k,
                static_pattern_ids=static_pattern_ids,
            )
        else:
            retrieved_memories = []

        dynamic_high_risk_count = sum(
            1 for row in retrieved_memories if str(row.get("risk_level", "")).lower() == "high"
        )
        suspicious = static_result["suspicious"] or bool(retrieved_memories)
        summary = self._build_summary(static_result, retrieved_memories)
        memory_scan = {
            "static_enabled": bool(static_result.get("static_enabled", False)),
            "dynamic_enabled": self.enable_defense_retrieval,
            "static_match_count": int(static_result.get("static_match_count", 0) or 0),
            "static_high_severity_count": int(
                static_result.get("static_high_severity_count", 0) or 0
            ),
            "dynamic_match_count": len(retrieved_memories),
            "dynamic_high_risk_count": dynamic_high_risk_count,
            "suspicious": suspicious,
        }

        return {
            **static_result,
            "retrieved_memories": retrieved_memories,
            "dynamic_match_count": len(retrieved_memories),
            "dynamic_high_risk_count": dynamic_high_risk_count,
            "dynamic_enabled": self.enable_defense_retrieval,
            "memory_scan": memory_scan,
            "suspicious": suspicious,
            "summary": summary,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def _build_summary(
        self,
        static_result: dict[str, Any],
        retrieved_memories: list[dict[str, Any]],
    ) -> str:
        parts = [str(static_result.get("summary", ""))]
        if not self.enable_defense_retrieval:
            parts.append("Historical defense-memory retrieval is disabled.")
        elif not retrieved_memories:
            parts.append("No historical defense memories matched.")
        else:
            risk_levels = [str(row.get("risk_level", "low")).lower() for row in retrieved_memories]
            high = sum(1 for level in risk_levels if level == "high")
            medium = sum(1 for level in risk_levels if level == "medium")
            low = sum(1 for level in risk_levels if level == "low")
            parts.append(
                "Retrieved "
                f"{len(retrieved_memories)} defense memory entr"
                f"{'y' if len(retrieved_memories) == 1 else 'ies'} "
                f"({high} high, {medium} medium, {low} low risk)."
            )
        return " ".join(part for part in parts if part)
