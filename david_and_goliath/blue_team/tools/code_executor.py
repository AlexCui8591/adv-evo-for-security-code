"""Blue Team tool: execute Python code in a sandboxed subprocess.

Used by the coding agent to verify that its draft solution actually runs
and produces expected output before finalizing.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Hard cap: never let the agent wait more than this, regardless of what the
# LLM requests.
_MAX_TIMEOUT_S = 15


class CodeExecutorTool:
    """Execute Python code and capture stdout/stderr.

    Runs code in a child subprocess so any crash or infinite loop is
    isolated from the main process.  Does NOT provide kernel-level
    sandboxing; use infra/sandbox.py on top if you need that.
    """

    name = "code_executor"
    description = (
        "Execute a Python code snippet in a subprocess and return stdout/stderr. "
        "Use this to verify that your solution produces the correct output. "
        "Execution is time-limited to prevent infinite loops."
    )

    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "code_executor",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds (default 5, max 15).",
                        "default": 5,
                    },
                },
                "required": ["code"],
            },
        },
    }

    def __init__(self, default_timeout: int = 5):
        self.default_timeout = default_timeout

    def run(self, code: str, timeout: int | None = None) -> dict[str, Any]:
        """Execute *code*, return stdout/stderr/returncode."""
        effective_timeout = min(timeout or self.default_timeout, _MAX_TIMEOUT_S)
        tmp = self._write_temp(code)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, tmp],
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "stdout": proc.stdout[:2000],
                "stderr": proc.stderr[:1000],
                "returncode": proc.returncode,
                "elapsed_ms": round(elapsed_ms, 1),
                "timed_out": False,
                "success": proc.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {effective_timeout}s.",
                "returncode": -1,
                "elapsed_ms": effective_timeout * 1000,
                "timed_out": True,
                "success": False,
            }
        except Exception as exc:
            return {
                "stdout": "",
                "stderr": str(exc),
                "returncode": -1,
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
                "timed_out": False,
                "success": False,
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
