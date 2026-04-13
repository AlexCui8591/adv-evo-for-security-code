"""Blue Team Reflexion module.

Implements the Reflexion pattern (Shinn et al., 2023):
  after an initial attempt fails the unit tests, the agent is given a
  structured reflection prompt that explains what went wrong and asks for
  an improved solution.

The CodingAgent calls ReflexionEngine.build_reflection_message() to
generate the next turn's user message when tests fail.

Only the reflection-prompt construction lives here; the actual LLM call
stays inside CodingAgent so retries are counted against the same turn budget.
"""

from __future__ import annotations

from typing import Any


class ReflexionEngine:
    """Build reflection prompts for failed coding attempts."""

    def build_reflection_message(
        self,
        solution_code: str,
        test_results: dict[str, Any],
        static_issues: list[dict[str, Any]] | None = None,
        executor_result: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Return a user-role message asking the agent to fix its solution.

        Args:
            solution_code:   The code that failed.
            test_results:    Output from UnitTestRunnerTool.run().
            static_issues:   Issues list from StaticAnalyzerTool.run(), optional.
            executor_result: Output from CodeExecutorTool.run(), optional.

        Returns:
            A single {"role": "user", "content": ...} dict ready to append
            to the conversation messages list.
        """
        sections: list[str] = ["## Reflection: Your solution has problems.\n"]

        # ---- Unit test failures ----
        failed_tests = [
            r for r in test_results.get("results", []) if not r.get("passed")
        ]
        if failed_tests:
            lines = [
                f"### Failed Tests ({len(failed_tests)}/{test_results.get('total', '?')})\n"
            ]
            for r in failed_tests[:5]:   # cap to avoid huge prompts
                lines.append(f"**Test {r['test_index']}**: `{r['test_code'][:200]}`")
                if r.get("stderr"):
                    lines.append(f"  Error: {r['stderr'][:300]}")
            sections.append("\n".join(lines))

        # ---- Execution errors ----
        if executor_result and not executor_result.get("success"):
            err = executor_result.get("stderr", "")[:400]
            sections.append(
                f"### Runtime Error\n"
                f"The code failed to run:\n```\n{err}\n```"
            )

        # ---- Static analysis warnings ----
        if static_issues:
            high = [i for i in static_issues if i.get("severity") == "HIGH"]
            if high:
                lines = [f"### Security Issues ({len(high)} HIGH severity)\n"]
                for i in high[:3]:
                    lines.append(
                        f"- Line {i.get('line', '?')}: {i.get('text', '')} "
                        f"[{i.get('test_id', '')}]"
                    )
                sections.append("\n".join(lines))

        # ---- Previous (broken) code for reference ----
        sections.append(
            f"### Your Previous Solution\n"
            f"```python\n{solution_code[:1500]}\n```"
        )

        sections.append(
            "Please analyse the failures above and write a corrected solution. "
            "Use the tools again to verify the fix before returning your final answer."
        )

        return {"role": "user", "content": "\n\n".join(sections)}
