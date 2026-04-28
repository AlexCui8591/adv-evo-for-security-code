"""Blue Team tool: run task test cases against a candidate solution."""

from __future__ import annotations

from typing import Any

from infra.sandbox import run_python_in_sandbox

_DEFAULT_TIMEOUT_S = 10


class UnitTestRunnerTool:
    """Execute per-test-case assertions against a solution."""

    name = "unit_test_runner"
    description = (
        "Run the coding task's test cases against a candidate Python solution. "
        "Returns pass/fail for every test case so you can verify correctness. "
        "Each run uses an isolated subprocess and a safety screen."
    )

    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "unit_test_runner",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "solution_code": {
                        "type": "string",
                        "description": "The complete Python solution to be tested.",
                    },
                    "test_cases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of Python statements (assert, function calls, etc.) "
                            "that validate the solution. Each is run independently."
                        ),
                    },
                },
                "required": ["solution_code", "test_cases"],
            },
        },
    }

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT_S):
        self.timeout = timeout

    def run(
        self,
        solution_code: str,
        test_cases: list[str],
    ) -> dict[str, Any]:
        if not test_cases:
            return {
                "passed": 0,
                "failed": 0,
                "total": 0,
                "results": [],
                "summary": "No test cases provided.",
            }

        results: list[dict[str, Any]] = []
        for idx, test in enumerate(test_cases, start=1):
            combined = f"{solution_code}\n\n# --- test {idx} ---\n{test}\n"
            outcome = self._run_single(combined, idx)
            results.append({"test_index": idx, "test_code": test, **outcome})

        passed = sum(1 for row in results if row.get("passed"))
        failed = len(results) - passed
        return {
            "passed": passed,
            "failed": failed,
            "total": len(test_cases),
            "results": results,
            "summary": f"{passed}/{len(test_cases)} test(s) passed.",
        }

    def _run_single(self, combined_code: str, idx: int) -> dict[str, Any]:
        result = run_python_in_sandbox(
            combined_code,
            self.timeout,
            label=f"unit_test_{idx}",
        )
        return {
            "passed": bool(result.get("success", False)),
            "stdout": str(result.get("stdout", ""))[:500],
            "stderr": str(result.get("stderr", ""))[:500],
            "elapsed_ms": round(float(result.get("elapsed_ms", 0.0)), 1),
            "blocked": bool(result.get("blocked", False)),
            "timed_out": bool(result.get("timed_out", False)),
        }
