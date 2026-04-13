"""Blue Team tool: run task test cases against a candidate solution.

The coding agent calls this after drafting a solution to confirm it passes
the tests embedded in the CodingTask before returning its final answer.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_DEFAULT_TIMEOUT_S = 10


class UnitTestRunnerTool:
    """Execute per-test-case assertions against a solution.

    Each test case is a short Python snippet (typically an assert statement
    or a call that should not raise).  The solution code is prepended so the
    test can call its functions.
    """

    name = "unit_test_runner"
    description = (
        "Run the coding task's test cases against a candidate Python solution. "
        "Returns pass/fail for every test case so you can verify correctness. "
        "Call this after writing your solution to confirm it is correct."
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
                            "that validate the solution.  Each is run independently."
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
        """Run every test case against *solution_code*.

        Returns aggregated pass/fail counts and per-test details.
        """
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

        passed = sum(1 for r in results if r.get("passed"))
        failed = len(results) - passed
        return {
            "passed": passed,
            "failed": failed,
            "total": len(test_cases),
            "results": results,
            "summary": f"{passed}/{len(test_cases)} test(s) passed.",
        }

    # ------------------------------------------------------------------

    def _run_single(self, combined_code: str, idx: int) -> dict[str, Any]:
        tmp = self._write_temp(combined_code)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, tmp],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            return {
                "passed": proc.returncode == 0,
                "stdout": proc.stdout[:500],
                "stderr": proc.stderr[:500],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"Test {idx} timed out after {self.timeout}s.",
                "elapsed_ms": self.timeout * 1000,
            }
        except Exception as exc:
            return {
                "passed": False,
                "stdout": "",
                "stderr": str(exc),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
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
