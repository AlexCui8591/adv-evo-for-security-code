"""Blue Team tool: execute Python code in a best-effort isolated subprocess."""

from __future__ import annotations

from typing import Any

from infra.sandbox import run_python_in_sandbox

_MAX_TIMEOUT_S = 15


class CodeExecutorTool:
    """Execute Python code and capture stdout/stderr."""

    name = "code_executor"
    description = (
        "Execute a Python code snippet in an isolated subprocess and return "
        "stdout/stderr. Use this to verify that your solution runs correctly. "
        "Execution is time-limited and screened for dangerous patterns."
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
        effective_timeout = min(timeout or self.default_timeout, _MAX_TIMEOUT_S)
        return run_python_in_sandbox(code, effective_timeout, label="code_executor")
