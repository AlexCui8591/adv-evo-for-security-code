"""Blue Team tool: static security analysis via Bandit.

Used by the coding agent to self-check generated code before returning it.
Mirrors Judge A in spirit but called on-demand during code generation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


class StaticAnalyzerTool:
    """Wrap Bandit for Blue Team on-demand code security scanning.

    The LLM coding agent calls this tool on its draft code to catch
    security issues before finalizing the output.
    """

    name = "static_analyzer"
    description = (
        "Analyze Python code for security vulnerabilities using Bandit. "
        "Returns a list of detected issues with severity (HIGH/MEDIUM/LOW) and line numbers. "
        "Call this on your draft code before returning a final answer."
    )

    # OpenAI-compatible tool schema
    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "static_analyzer",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to analyze for security issues.",
                    }
                },
                "required": ["code"],
            },
        },
    }

    def run(self, code: str) -> dict[str, Any]:
        """Run Bandit on *code* and return a structured findings dict."""
        t0 = time.perf_counter()

        if shutil.which("bandit") is None:
            return {
                "available": False,
                "issues": [],
                "issue_count": 0,
                "summary": "Bandit not installed — static analysis skipped.",
                "elapsed_ms": 0.0,
            }

        tmp = self._write_temp(code)
        try:
            result = subprocess.run(
                ["bandit", "-f", "json", "-q", tmp],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            data = json.loads(result.stdout) if result.stdout.strip() else {}
            raw_issues = data.get("results", [])

            issues = [
                {
                    "severity": i.get("issue_severity", "UNKNOWN"),
                    "confidence": i.get("issue_confidence", "UNKNOWN"),
                    "text": i.get("issue_text", ""),
                    "line": i.get("line_number", 0),
                    "test_id": i.get("test_id", ""),
                }
                for i in raw_issues
            ]

            high = sum(1 for i in issues if i["severity"] == "HIGH")
            medium = sum(1 for i in issues if i["severity"] == "MEDIUM")

            return {
                "available": True,
                "issues": issues,
                "issue_count": len(issues),
                "high_severity": high,
                "medium_severity": medium,
                "summary": (
                    f"Found {len(issues)} issue(s): {high} HIGH, {medium} MEDIUM."
                    if issues
                    else "No security issues found."
                ),
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            }

        except subprocess.TimeoutExpired:
            return {
                "available": True,
                "issues": [],
                "issue_count": 0,
                "summary": "Bandit timed out.",
                "elapsed_ms": 10_000.0,
            }
        except Exception as exc:
            return {
                "available": True,
                "issues": [],
                "issue_count": 0,
                "summary": f"Bandit error: {exc}",
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            }
        finally:
            Path(tmp).unlink(missing_ok=True)

    @staticmethod
    def _write_temp(code: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", encoding="utf-8", delete=False
        ) as fh:
            fh.write(code)
            return fh.name
